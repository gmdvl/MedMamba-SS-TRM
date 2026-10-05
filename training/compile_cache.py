# -*- coding: utf-8 -*-
"""
training/compile_cache.py
================================
MedMamba-SS-TRM v17, Stage 12b. Keep `torch.compile`'s PERSISTENT caches off a RAM-backed `/tmp`.

Why
---
A run died at gate G7, two minutes in, after the model was built and G1/G2/G9 had passed:

    File ".../triton/backends/nvidia/compiler.py", line 465, in make_cubin
        fsrc.write(src)
    OSError: [Errno 28] No space left on device

There was space: `df /tmp` reported 2.6 GB free. The trap is that **`/tmp` on this machine
is a 3.1 GB tmpfs**, and tmpfs pages come out of the same pool as anonymous memory - so its
`Avail` column is a size cap, not free storage. With ~21 GB of 30 GB resident, a write into
tmpfs can fail with ENOSPC while `df` still shows gigabytes. `/tmp/torchinductor_<user>`
had already grown to 164 MB of RAM, and Inductor compiles Triton kernels in parallel
subprocesses that all write there.

What this sets, and what it deliberately does NOT
-------------------------------------------------
    TORCHINDUCTOR_CACHE_DIR   generated Python modules + kernel cache   <- redirected
    TRITON_CACHE_DIR          Triton's compiled-kernel cache            <- redirected
    TMPDIR                    process-wide scratch for `tempfile`       <- NOT by default

**TMPDIR is not redirected by default, and the reason is a bug this module shipped with.**
Pointing `TMPDIR` at a cache directory inside this repository broke every DataLoader worker:

    OSError: AF_UNIX path too long

`multiprocessing`'s resource sharer binds a Unix-domain socket under `TMPDIR`, appending
`/pymp-XXXXXXXX/listener-XXXXXXXX` - 32 bytes - and `sun_path` holds at most 108. The
repository path is 63 characters, so `<repo>/.cache/compile/tmp` came to 79 and the socket
address to 111. The DataLoader, not the compiler, paid for it.

That makes `TMPDIR` the wrong lever anyway: it is process-wide and affects everything that
opens a temporary file, while the thing that actually filled tmpfs is the two PERSISTENT
caches above. Redirecting those keeps `/tmp` nearly empty, which is what the small,
short-lived `make_cubin` writes need. Pass `tmpdir=True` to redirect it as well; the
`_AF_UNIX_BUDGET` guard below refuses any path that would not leave room for the socket.

Each variable is checked independently: one the caller has already exported is left alone
rather than suppressing the others. Auto-fixing rather than warning follows
`training/fd_limit.raise_open_file_limit`, which raises RLIMIT_NOFILE for the same
reason - a run should not die of an environment default nobody chose.

The cache is shared across runs on purpose: Inductor compilation costs 1-2 minutes on the
first step, and a per-run cache would pay that every run.

Never raises. On any failure the caller keeps torch's defaults.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional

# Filesystem types whose "free space" is really free MEMORY. `df` reports their size cap,
# so a capacity check alone cannot tell you a write will succeed.
_MEMORY_BACKED = {"tmpfs", "ramfs", "devtmpfs"}

# Below this, one parallel Inductor compile can plausibly fill the filesystem. The observed
# failure had 2.6 GB "free"; the margin is for the memory-pressure case, not the disk one.
_MIN_SAFE_GB = 8.0

# `sun_path` is 108 bytes including the NUL. multiprocessing appends
# `/pymp-XXXXXXXX` (14) + `/listener-XXXXXXXX` (18) = 32. Keep headroom for a longer
# prefix in another Python version: a TMPDIR that violates this breaks DataLoader workers
# with "AF_UNIX path too long", which points at multiprocessing and not at the real cause.
_AF_UNIX_BUDGET = 60


def _fstype(path: str) -> Optional[str]:
    """Filesystem type backing `path`, from /proc/mounts. None if undeterminable."""
    try:
        target = os.path.realpath(path)
        best, best_type = "", None
        with open("/proc/mounts") as f:
            for line in f:
                parts = line.split()
                if len(parts) < 3:
                    continue
                mount, fstype = parts[1], parts[2]
                # Longest matching mount point wins, so /tmp beats / for /tmp/x.
                if (target == mount or target.startswith(mount.rstrip("/") + "/")) \
                        and len(mount) > len(best):
                    best, best_type = mount, fstype
        return best_type
    except Exception:
        return None


def _free_gb(path: str) -> Optional[float]:
    try:
        return shutil.disk_usage(path).free / 1024 ** 3
    except Exception:
        return None


def _usable(path: Path) -> bool:
    """Writable, on real storage, with room. A write probe, because mkdir succeeding on a
    memory-backed filesystem proves nothing - the whole point here is that `df` lies."""
    try:
        path.mkdir(parents=True, exist_ok=True)
        if _fstype(str(path)) in _MEMORY_BACKED:
            return False
        free = _free_gb(str(path))
        if free is not None and free < _MIN_SAFE_GB:
            return False
        probe = path / ".probe"
        probe.write_bytes(b"\0" * (1 << 20))
        probe.unlink()
        return True
    except Exception:
        return False


def describe_compile_scratch() -> Dict[str, object]:
    """What torch.compile would use right now, and whether it is memory-backed."""
    current = os.environ.get("TORCHINDUCTOR_CACHE_DIR") or os.environ.get("TMPDIR") or "/tmp"
    fstype = _fstype(current)
    return {
        "path": current,
        "fstype": fstype,
        "free_gb": _free_gb(current),
        "memory_backed": fstype in _MEMORY_BACKED if fstype else None,
        "already_set": [v for v in ("TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR", "TMPDIR")
                        if v in os.environ],
    }


def _short_tmpdir_candidates(root: Path) -> List[Path]:
    """Directories short enough to hold a Unix socket, nearest the repo first.

    Walks UP from the repository rather than down, because the repository path is itself
    63 characters here - the thing that made the original redirect impossible.
    """
    out = []
    for base in [root] + list(root.parents):
        cand = base / ".medmamba-ss-trm-tmp"
        if len(str(cand)) + _AF_UNIX_BUDGET <= 107:
            out.append(cand)
    return out


def ensure_compile_scratch_on_disk(preferred: Optional[str] = None,
                                    tmpdir: bool = False,
                                    verbose: bool = True) -> Optional[str]:
    """Redirect Inductor/Triton caches to disk if the default is memory-backed.

    `preferred` defaults to `<repo>/.cache/compile`, on whatever disk the repository lives
    on - the same disk that already holds `experiments/` and `data/`.
    `tmpdir=True` additionally redirects `TMPDIR`, subject to `_AF_UNIX_BUDGET`.

    Returns the directory now in use, or None if nothing changed.
    """
    state = describe_compile_scratch()
    free_gb = state["free_gb"]
    risky = bool(state["memory_backed"]) or (free_gb is not None and free_gb < _MIN_SAFE_GB)
    if not risky:
        return None

    root = Path(__file__).resolve().parent.parent
    target = Path(preferred) if preferred else root / ".cache" / "compile"
    if not _usable(target):
        if verbose:
            print(f"[compile-cache] WARNING: {state['path']} is {state['fstype']} "
                  f"({free_gb:.1f} GB) but {target} is unusable. Keeping torch's defaults; "
                  f"a Triton compile may fail with ENOSPC.", flush=True)
        return None

    changed = []
    for var, sub in (("TORCHINDUCTOR_CACHE_DIR", "inductor"), ("TRITON_CACHE_DIR", "triton")):
        if var in os.environ:                       # per-variable, not all-or-nothing
            continue
        os.environ[var] = str(target / sub)
        changed.append(var)

    if tmpdir and "TMPDIR" not in os.environ:
        for cand in _short_tmpdir_candidates(root):
            if _usable(cand):
                os.environ["TMPDIR"] = str(cand)
                changed.append("TMPDIR")
                break
        else:
            if verbose:
                print("[compile-cache] TMPDIR left alone: no directory near this repository "
                      "is short enough for a Unix socket (AF_UNIX sun_path is 108 bytes and "
                      "multiprocessing appends ~32). Redirecting it anyway would break the "
                      "DataLoader, not help the compiler.", flush=True)

    if verbose and changed:
        why = ("memory-backed " + str(state["fstype"])) if state["memory_backed"] \
            else f"only {free_gb:.1f} GB free"
        kept = [v for v in state["already_set"] if v not in changed]
        print(f"[compile-cache] {state['path']} is {why} - tmpfs reports a size cap, not free "
              f"memory, so a parallel Triton compile can hit ENOSPC there. Set "
              f"{', '.join(changed)} -> {target} ({_free_gb(str(target)):.0f} GB free)."
              + (f" Left your {', '.join(kept)} untouched." if kept else ""), flush=True)
    return str(target)


def warn_if_tmpdir_breaks_worker_ipc(verbose: bool = True) -> Optional[str]:
    """Warn when the INHERITED `TMPDIR` is too long for multiprocessing's Unix socket.

    The mirror image of the bug in this module's history. There, a long `TMPDIR` was set by
    this code; here it arrives from the caller's environment - and the symptom is identical
    and equally misdirecting:

        OSError: AF_UNIX path too long

    raised from `multiprocessing/connection.py` inside a DataLoader worker's feeder thread,
    naming nothing that has anything to do with the data pipeline it just killed. Checking
    costs one string length, and the message is the difference between a five-minute fix and
    an afternoon. Returns the offending path, or None.

    Not auto-fixed: `TMPDIR` is the caller's to choose, and silently relocating everything
    the process writes to a temporary file is a bigger side effect than the problem.
    """
    tmp = os.environ.get("TMPDIR")
    if not tmp:
        return None
    if len(tmp.rstrip("/")) + _AF_UNIX_BUDGET <= 107:
        return None
    if verbose:
        print(f"[compile-cache] WARNING: TMPDIR={tmp!r} is {len(tmp)} characters. "
              f"multiprocessing binds a Unix socket under it and appends ~32 bytes against a "
              f"108-byte sun_path limit, so DataLoader workers will die with 'AF_UNIX path "
              f"too long' - a traceback that names multiprocessing and not TMPDIR. Unset it, "
              f"or point it somewhere shorter (<= {107 - _AF_UNIX_BUDGET} characters).",
              flush=True)
    return tmp
