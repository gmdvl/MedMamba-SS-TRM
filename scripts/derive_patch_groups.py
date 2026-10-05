# -*- coding: utf-8 -*-
"""
scripts/derive_patch_groups.py
================================
MedMamba-SS-TRM v16 plan, Stage 0.3 - read-only. No re-prep. Unblocks A-4, A-5 and
C-1 TODAY on datasets that are already on disk.

Row order in the unified `X_*.npy` is deterministic and sorted by item key
(`training/prep/manifest.py:collect_shard_entries(deterministic=True)`,
called from `training/prep/core.py:_unify_all` -> `core.py:208`). So a
cumulative sum over `sorted(_progress.json["completed_items"])`, filtered by
split, using each item's `counts[modality]`, reconstructs which raw row of
`X_{split}.npy` came from which capture/image, in order, without touching a
single pixel.

Emits, alongside the modality dir (for the `histology_hsi_bc` adapter):

  groups_{split}.npy    - patient id per patch (int32)
  captures_{split}.npy  - capture index per patch (int32)
  capture_index.json    - index -> {key, patient, tissue, capture, split, n}

For the `pad_ufes_20` adapter (flat_output, one modality: "image"), emits:

  images_{split}.npy    - clinical image id per patch (int32)

Patient/tissue/capture are recovered from the item KEY, by re-applying the
SAME regex the frozen HSI adapter already used to build that key
(`prepare_histologyhsi_bc.CAPTURE_RE`, imported not reimplemented) - the
key is `os.path.relpath(capture_dir, hsi_root).replace(os.sep, "_")` or the
flat capture directory name, and in both cases the pattern
`HSI?_VNIR_<patient>_<tissue>_x10_C<capture>` appears somewhere in it.

Self-check, mandatory: `sum(counts) == len(y_{split})` per split, and every
capture's derived patches carry a single label. If either fails, NO sidecar
is written for that split - a silently misaligned group vector is worse than
none.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.prep.manifest import load_manifest  # frozen, imported not edited
from training.prep.core import _split_out_name, _modality_dir  # frozen, imported not edited
from prepare_histologyhsi_bc import CAPTURE_RE

_SPLITS = ("train", "validation", "test")


def _parse_capture_key(key: str):
    m = CAPTURE_RE.search(key)
    if not m:
        return None
    return m["patient"], m["tissue"], m["capture"]


def _sorted_items_for_split(manifest: dict, split: str):
    items = manifest.get("completed_items", {})
    return [(k, items[k]) for k in sorted(items) if items[k].get("split") == split]


def _derive_hsi(out_dir: str, manifest: dict, modality: str, dry_run: bool) -> Dict[str, dict]:
    m_dir = _modality_dir(out_dir, _FlatOutputShim(False), modality)
    reports = {}

    # capture_index.json is global (spans every split) so a single index means
    # the same thing wherever it appears.
    all_keys = sorted(manifest.get("completed_items", {}))
    capture_index: Dict[int, dict] = {}
    key_to_idx: Dict[str, int] = {}
    for i, key in enumerate(all_keys):
        parsed = _parse_capture_key(key)
        item = manifest["completed_items"][key]
        patient, tissue, capture = parsed if parsed else (None, None, None)
        capture_index[i] = {
            "key": key, "patient": patient, "tissue": tissue, "capture": capture,
            "split": item.get("split"), "n": int((item.get("counts") or {}).get(modality, 0)),
        }
        key_to_idx[key] = i

    for split in _SPLITS:
        out_name = _split_out_name(split)
        y_path = os.path.join(m_dir, f"y_{out_name}.npy")
        if not os.path.isfile(y_path):
            continue  # split not active for this dataset (e.g. no validation)
        y = np.load(y_path, mmap_mode="r")

        entries = _sorted_items_for_split(manifest, split)
        patient_ids: List[int] = []
        capture_ids: List[int] = []
        cursor = 0
        ok = True
        for key, item in entries:
            n = int((item.get("counts") or {}).get(modality, 0))
            if n <= 0:
                continue
            parsed = _parse_capture_key(key)
            if parsed is None:
                print(f"  [warn] key {key!r} did not match CAPTURE_RE - skipping split {split}'s sidecars "
                      f"entirely (a partial group vector is worse than none).", flush=True)
                ok = False
                break
            patient, _tissue, _capture = parsed
            try:
                patient_int = int(patient)
            except ValueError:
                patient_int = abs(hash(patient)) % (2 ** 31 - 1)
            idx = key_to_idx[key]

            batch_labels = np.asarray(y[cursor:cursor + n])
            if batch_labels.size and len(np.unique(batch_labels)) != 1:
                print(f"  [FAIL] capture {key!r} (rows {cursor}:{cursor + n} of split={split}) has "
                      f"{len(np.unique(batch_labels))} distinct labels - the row-order reconstruction "
                      f"is wrong. Aborting sidecar generation for split={split}.", flush=True)
                ok = False
                break

            patient_ids.extend([patient_int] * n)
            capture_ids.extend([idx] * n)
            cursor += n

        if not ok:
            reports[split] = {"written": False, "reason": "self-check failed"}
            continue
        if cursor != len(y):
            print(f"  [FAIL] split={split} modality={modality}: reconstructed {cursor} rows, "
                  f"y_{out_name}.npy has {len(y)}. Not writing sidecars for this split.", flush=True)
            reports[split] = {"written": False, "reason": f"count mismatch {cursor} != {len(y)}"}
            continue

        groups = np.asarray(patient_ids, dtype=np.int32)
        captures = np.asarray(capture_ids, dtype=np.int32)
        if not dry_run:
            np.save(os.path.join(m_dir, f"groups_{out_name}.npy"), groups)
            np.save(os.path.join(m_dir, f"captures_{out_name}.npy"), captures)
        reports[split] = {"written": not dry_run, "n": int(cursor), "n_patients": int(len(set(patient_ids))),
                          "n_captures": int(len(set(capture_ids)))}
        print(f"  [{modality}/{split}] {cursor} patches, {reports[split]['n_patients']} patient(s), "
              f"{reports[split]['n_captures']} capture(s) -> "
              f"groups_{out_name}.npy / captures_{out_name}.npy", flush=True)

    if not dry_run:
        with open(os.path.join(m_dir, "capture_index.json"), "w") as f:
            json.dump(capture_index, f, indent=2)
    return reports


class _FlatOutputShim:
    """`_modality_dir` only reads `.flat_output`; avoids importing the real
    adapter classes (and their heavy deps) just for that one bool."""
    def __init__(self, flat_output: bool):
        self.flat_output = flat_output


def _derive_pad(out_dir: str, manifest: dict, modality: str, dry_run: bool) -> Dict[str, dict]:
    m_dir = out_dir  # flat_output
    all_keys = sorted(manifest.get("completed_items", {}))
    key_to_idx = {k: i for i, k in enumerate(all_keys)}
    reports = {}

    for split in _SPLITS:
        out_name = _split_out_name(split)
        y_path = os.path.join(m_dir, f"y_{out_name}.npy")
        if not os.path.isfile(y_path):
            continue
        y = np.load(y_path, mmap_mode="r")

        entries = _sorted_items_for_split(manifest, split)
        image_ids: List[int] = []
        cursor = 0
        ok = True
        for key, item in entries:
            n = int((item.get("counts") or {}).get(modality, 0))
            if n <= 0:
                continue
            batch_labels = np.asarray(y[cursor:cursor + n])
            if batch_labels.size and len(np.unique(batch_labels)) != 1:
                print(f"  [FAIL] image {key!r} (rows {cursor}:{cursor + n} of split={split}) has "
                      f"{len(np.unique(batch_labels))} distinct labels. Aborting for split={split}.",
                      flush=True)
                ok = False
                break
            image_ids.extend([key_to_idx[key]] * n)
            cursor += n

        if not ok or cursor != len(y):
            reports[split] = {"written": False, "reason": "self-check failed"}
            continue

        images = np.asarray(image_ids, dtype=np.int32)
        if not dry_run:
            np.save(os.path.join(m_dir, f"images_{out_name}.npy"), images)
        reports[split] = {"written": not dry_run, "n": int(cursor), "n_images": int(len(set(image_ids)))}
        print(f"  [{modality}/{split}] {cursor} patches, {reports[split]['n_images']} clinical image(s) "
              f"-> images_{out_name}.npy", flush=True)

    if not dry_run:
        with open(os.path.join(m_dir, "image_index.json"), "w") as f:
            json.dump({i: k for k, i in key_to_idx.items()}, f, indent=2)
    return reports


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out_dir", required=True,
                     help="the prep output root that holds _progress.json "
                          "(e.g. data/hsi_v7-80_10_10_importance or data/pad_v6)")
    ap.add_argument("--dry_run", action="store_true", help="report what would be written; write nothing")
    args = ap.parse_args(argv)

    manifest = load_manifest(args.out_dir)
    if manifest is None:
        raise SystemExit(f"no _progress.json found under {args.out_dir}")

    adapter = manifest.get("adapter")
    modalities = manifest.get("modalities") or []
    print(f"adapter={adapter!r} modalities={modalities}", flush=True)

    if adapter.startswith("histology_hsi_bc"):
        for modality in modalities:
            print(f"[{modality}]", flush=True)
            _derive_hsi(args.out_dir, manifest, modality, args.dry_run)
    elif adapter.startswith("pad_ufes_20"):
        for modality in modalities:
            print(f"[{modality}]", flush=True)
            _derive_pad(args.out_dir, manifest, modality, args.dry_run)
    else:
        raise SystemExit(f"unsupported adapter {adapter!r} - derive_patch_groups.py only knows "
                          f"histology_hsi_bc and pad_ufes_20")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
