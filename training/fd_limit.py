# -*- coding: utf-8 -*-
"""
training/fd_limit.py
===========================
MedMamba-SS-TRM v17, note 3. Raise this process's open-file soft limit to its hard limit.

Why
---
A 2-hour HSI run reached gate G6 and died there:

    [test-eval] 348894 held-out test patches from .../X_test.npy
    WARNING: test-split evaluation failed: RuntimeError: Too many open files.
    Communication with the workers is no longer possible.

The test split is the LARGEST loader in the run - validation was subsampled to 10% and
training to 10%, but the test split is evaluated whole - and it is built LAST, after the
training and validation loaders have already consumed descriptors. PyTorch's default
`file_descriptor` sharing strategy passes one FD per shared tensor between workers, so the
ceiling is the process's `RLIMIT_NOFILE`, whose SOFT value is commonly 1024 while the hard
value is 1048576. Nothing in the run needs a 1024-descriptor budget; it was simply never
raised.

This raises soft to hard, which is a plain request for headroom the kernel has already
granted the user. It is not the same as `torch.multiprocessing.set_sharing_strategy(
"file_system")`, which trades the FD ceiling for shared-memory segments that leak on a
hard kill - a bad trade in a repo whose v12/v13 plans are largely about `/dev/shm`
exhaustion and worker SIGBUS. If the raised limit still is not enough, the caller's
0-worker retry is the backstop.

No-op on platforms without `resource` (Windows), and never raises.
"""

from __future__ import annotations

from typing import Optional, Tuple


def raise_open_file_limit(verbose: bool = True) -> Optional[Tuple[int, int]]:
    """Raise RLIMIT_NOFILE soft -> hard. Returns `(old_soft, new_soft)`, or None."""
    try:
        import resource
    except ImportError:                                       # pragma: no cover - Windows
        return None
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        if soft >= hard:
            return (soft, soft)
        resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
        new_soft = resource.getrlimit(resource.RLIMIT_NOFILE)[0]
        if verbose and new_soft > soft:
            print(f"[fd-limit] open files: soft {soft} -> {new_soft} (hard {hard}). DataLoader "
                  f"workers pass one descriptor per shared tensor; the default soft limit is "
                  f"what gate G6 ran out of on the full test split.", flush=True)
        return (soft, new_soft)
    except Exception as e:                                    # pragma: no cover
        if verbose:
            print(f"[fd-limit] could not raise the open-file limit ({type(e).__name__}: {e}); "
                  f"continuing at the inherited soft limit.", flush=True)
        return None


def is_fd_exhaustion(exc: BaseException) -> bool:
    """True when `exc` is the DataLoader's descriptor-exhaustion failure.

    Matched on the message because PyTorch raises a bare `RuntimeError` for it - there is
    no dedicated exception type to catch, and an `OSError` with `errno.EMFILE` is the
    other spelling the same condition takes.
    """
    text = str(exc).lower()
    if "too many open files" in text:
        return True
    return isinstance(exc, OSError) and getattr(exc, "errno", None) == 24   # EMFILE
