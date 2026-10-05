# -*- coding: utf-8 -*-
"""
prepare_pad_ufes_20_v5.py
=========================
Turns a raw PAD-UFES-20 download (https://data.mendeley.com/datasets/zr7vgbcyr2/1)
into flat ``X_train.npy`` / ``y_train.npy`` (+ optional ``X_val.npy`` /
``y_val.npy``) / ``X_test.npy`` / ``y_test.npy`` files via a memory-bounded,
resumable, batch-streamed pipeline. Optional sliding-window patch extraction.

What v5 adds over ``prepare_pad_ufes_20_v4.py`` (v4 is frozen for reproducing
existing datasets):

  * **Crash-safe finalize** - the final ``X_*.npy`` is built into ``<name>.tmp``
    and ``fsync``+``os.replace``-d (``training.dataset_prep_unify``); the source
    shards are deleted only after the produced arrays are re-read end-to-end
    (``--verify_level deep`` by default). A SIGKILL / OOM / disk-full during
    unify can no longer leave a right-sized-but-corrupt / all-zeros
    ``X_train.npy`` with the shards already gone.
  * **``dataset_manifest.json``** (sha256 + provenance) is written next to the
    arrays, so ``train_example_v13.py --manifest_check`` can cross-check them.
  * **Stratified, patient-level, coverage-checked splits** - the same
    ``training.dataset_prep_common`` utilities the HistologyHSI script uses.
    ``--split`` presets replace ``--test_size`` (kept as a deprecated alias);
    a real validation split is produced for three-way presets; an uncoverable
    class aborts *here*, before any patch extraction, not later at train time.
  * **Fail-soft ingestion** - one unreadable / corrupt source image is
    recorded under ``skipped_images`` and the run continues, instead of
    aborting or being retried forever.

Usage::

    python prepare_pad_ufes_20_v5.py --root /path/to/PAD-UFES-20 --out_dir ./data_pad \\
        --patch_size 11 --stride 11 --split 80_10_10

    # if all images finished but the unify step didn't:
    python prepare_pad_ufes_20_v5.py --out_dir ./data_pad --finalize_only
"""

import argparse
import datetime as _dt
import gc
import json
import os
import subprocess

import numpy as np
import pandas as pd
from PIL import Image

from training.dataset_prep_common import (
    SPLIT_PRESETS, SPLIT_ALGO_VERSION, SkipRegistry, check_prep_class_coverage,
    patient_split_three, resolve_split_fractions, stratified_patient_split_three,
)
from training.dataset_prep_unify import (
    dataset_is_finalized_and_intact, finalize_dataset, unify_split,
)
from training.npy_atomic import save_json_atomic, save_npy_atomic
from training.npy_integrity import DatasetStorageError

PROGRESS_FILE = "_progress.json"

# Standard class mappings for PAD-UFES-20
DIAGNOSTIC_TO_ID = {'ACK': 0, 'BCC': 1, 'MEL': 2, 'NEV': 3, 'SCC': 4, 'SEK': 5}

def SPLITS_FOR(has_val):
    return ("train", "validation", "test") if has_val else ("train", "test")


def _split_out_name(split: str) -> str:
    return "val" if split == "validation" else split


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=os.path.dirname(os.path.abspath(__file__)),
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


# ============================================================================
# Streaming shard writer - bounds memory to ~one shard at a time
# ============================================================================

class ShardWriter:
    def __init__(self, batches_dir, split, shard_counter_start, batch_size):
        self.batches_dir = batches_dir
        self.split = split
        self.batch_size = batch_size
        self.shard_id = shard_counter_start
        self.buf_x, self.buf_y = [], []
        self.written_shards = []
        os.makedirs(os.path.join(batches_dir, split), exist_ok=True)

    def add(self, image_array, label):
        self.buf_x.append(image_array)
        self.buf_y.append(label)
        if len(self.buf_x) >= self.batch_size:
            self._flush()

    def _flush(self):
        if not self.buf_x:
            return
        name = f"shard_{self.shard_id:06d}"
        x_path = os.path.join(self.batches_dir, self.split, name + "_X.npy")
        y_path = os.path.join(self.batches_dir, self.split, name + "_y.npy")
        # atomic + fsync (training.npy_atomic) so a kill mid-write never leaves
        # a final-named shard with an unwritten tail.
        save_npy_atomic(x_path, np.stack(self.buf_x).astype(np.float32))
        save_npy_atomic(y_path, np.array(self.buf_y, dtype=np.int64))
        self.written_shards.append((x_path, y_path, len(self.buf_x)))
        self.shard_id += 1
        self.buf_x, self.buf_y = [], []
        gc.collect()

    def finalize(self):
        self._flush()
        return self.written_shards, self.shard_id


# ============================================================================
# Resumable progress manifest
# ============================================================================

def manifest_path(out_dir):
    return os.path.join(out_dir, PROGRESS_FILE)


def load_manifest(out_dir):
    p = manifest_path(out_dir)
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return None


def save_manifest(out_dir, manifest):
    save_json_atomic(manifest_path(out_dir), manifest)


_COMPAT_KEYS = ("img_size", "patch_size", "stride", "seed", "batch_size", "split", "split_strategy")


def new_manifest(args, class_names, train_patients, val_patients, test_patients):
    return {
        "args": {
            "img_size": args.img_size, "patch_size": args.patch_size, "stride": args.stride,
            "seed": args.seed, "batch_size": args.batch_size,
            "split": args.split, "split_strategy": args.split_strategy,
            "split_algo_version": SPLIT_ALGO_VERSION,
        },
        "class_names": class_names,
        "train_patients": sorted(train_patients),
        "validation_patients": sorted(val_patients),
        "test_patients": sorted(test_patients),
        "next_shard_id": {s: 0 for s in ("train", "validation", "test")},
        "completed_images": {},
        "skipped_images": {},
        "finalized": False,
    }


def check_manifest_compatible(manifest, args):
    saved = manifest.get("args", {})
    current = {k: getattr(args, k) for k in _COMPAT_KEYS}
    mismatches = {k: (saved.get(k), v) for k, v in current.items() if saved.get(k) != v}
    if mismatches:
        raise ValueError(
            f"Found an existing progress file with DIFFERENT settings: {mismatches}. "
            f"Use a fresh --out_dir, delete _progress.json to start over, or pass --fresh.")


def cleanup_orphan_shards(out_dir, manifest):
    referenced = set()
    for img in manifest["completed_images"].values():
        for entry in (img.get("shards") or []):
            referenced.add(entry[0])
            referenced.add(entry[1])
    batches_dir = os.path.join(out_dir, "_batches")
    if not os.path.isdir(batches_dir):
        return
    for split in ("train", "validation", "test"):
        split_dir = os.path.join(batches_dir, split)
        if not os.path.isdir(split_dir):
            continue
        for fname in os.listdir(split_dir):
            fpath = os.path.join(split_dir, fname)
            if fname.endswith(".tmp") or fpath not in referenced:
                os.remove(fpath)


# ============================================================================
# Discovery and Processing
# ============================================================================

def discover_images(root, exclude_dir=None):
    meta_path = os.path.join(root, "metadata.csv")
    if not os.path.exists(meta_path):
        found = [f for f in os.listdir(root) if f.lower().endswith('.csv')]
        if found:
            meta_path = os.path.join(root, found[0])
        else:
            raise FileNotFoundError(f"Could not find metadata.csv (or any .csv file) in {root}")

    df = pd.read_csv(meta_path)
    col_map = {col.lower().strip(): col for col in df.columns}
    for req in ['img_id', 'patient_id', 'diagnostic']:
        if req not in col_map:
            raise ValueError(f"metadata.csv missing column '{req}'. Found: {list(df.columns)}")
    img_col, patient_col, diag_col = col_map['img_id'], col_map['patient_id'], col_map['diagnostic']

    exclude_dir = os.path.abspath(exclude_dir) if exclude_dir else None
    image_files, n_collision = {}, 0
    for dirpath, _, filenames in os.walk(root):
        if exclude_dir and os.path.abspath(dirpath).startswith(exclude_dir):
            continue  # never ingest a previous --out_dir nested under --root
        for f in filenames:
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                stem = os.path.splitext(f)[0].strip()
                full = os.path.join(dirpath, f)
                if stem in image_files and image_files[stem] != full:
                    n_collision += 1
                    print(f"  [warn] duplicate image stem {stem!r}: keeping {image_files[stem]}, "
                          f"ignoring {full}")
                    continue
                image_files[stem] = full
    if n_collision:
        print(f"  [warn] {n_collision} image(s) ignored due to stem collisions.")

    captures, skipped = [], 0
    for _, row in df.iterrows():
        clean_stem = os.path.splitext(str(row[img_col]).strip())[0].strip()
        if clean_stem not in image_files:
            skipped += 1
            continue
        diag = str(row[diag_col]).strip().upper()
        if diag not in DIAGNOSTIC_TO_ID:
            continue
        captures.append({"key": clean_stem, "path": image_files[clean_stem],
                         "patient": str(row[patient_col]).strip(), "label": DIAGNOSTIC_TO_ID[diag]})
    if skipped:
        print(f"  [warn] Skipped {skipped} images listed in metadata that were not found on disk.")
    return captures


def process_one_image(img_path, target_size, writer, label, patch_size=None, stride=None):
    img = Image.open(img_path).convert("RGB")
    if target_size is not None and target_size > 0:
        img = img.resize((target_size, target_size), Image.Resampling.BILINEAR)
    img_array = np.asarray(img, dtype=np.float32) / 255.0

    if patch_size is not None and patch_size > 0:
        step = stride if (stride is not None and stride > 0) else patch_size
        h, w, _ = img_array.shape
        count = 0
        for y in range(0, h - patch_size + 1, step):
            for x in range(0, w - patch_size + 1, step):
                writer.add(img_array[y:y + patch_size, x:x + patch_size, :], label)
                count += 1
        return count
    writer.add(img_array, label)
    return 1


# ============================================================================
# Unify / finalize
# ============================================================================

def unify_all(out_dir, manifest, args):
    roles_present = {
        "X_train": os.path.join(out_dir, "X_train.npy"), "y_train": os.path.join(out_dir, "y_train.npy"),
        "X_val": os.path.join(out_dir, "X_val.npy"), "y_val": os.path.join(out_dir, "y_val.npy"),
        "X_test": os.path.join(out_dir, "X_test.npy"), "y_test": os.path.join(out_dir, "y_test.npy"),
    }
    has_val = bool(manifest.get("validation_patients"))
    splits = SPLITS_FOR(has_val)
    active_roles = {r: p for r, p in roles_present.items()
                    if (has_val or "val" not in r)}

    if manifest.get("finalized") and dataset_is_finalized_and_intact(out_dir, active_roles):
        print("  Already finalized and the X/y .npy files still verify - nothing to do.")
        return None

    shards_by_split = {s: [] for s in splits}
    for img in manifest["completed_images"].values():
        entries = img.get("shards")
        if not entries:
            continue
        split = img.get("split", "train")
        if split not in shards_by_split:
            continue
        shards_by_split[split].extend(entries)

    counts = {}
    for split in splits:
        out_name = _split_out_name(split)
        counts[split], _ = unify_split(
            shards_by_split[split], os.path.join(out_dir, f"X_{out_name}.npy"),
            os.path.join(out_dir, f"y_{out_name}.npy"))
    print("  Unified: " + " ".join(f"{s}={counts[s]}" for s in splits) + f" -> {out_dir}/")

    manifest_extra = {
        "prep_script": "prepare_pad_ufes_20_v5.py", "prep_version": 5,
        "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "git_sha": _git_sha(), "cli_args": vars(args),
        "split_algo_version": manifest["args"].get("split_algo_version"),
        "split": manifest["args"].get("split"), "class_names": manifest.get("class_names"),
    }
    try:
        finalize_dataset(out_dir, active_roles, manifest_extra=manifest_extra,
                         verify=not args.no_verify_output, verify_level=args.verify_level)
    except DatasetStorageError as e:
        print(f"\nFAILURE_CLASS=DATASET_STORAGE_ERROR  ({e})", flush=True)
        print("  The unified arrays did NOT verify. Source shards were kept; fix the cause "
              "(disk space / disk health) and re-run with --finalize_only.", flush=True)
        raise SystemExit(2)

    if not args.keep_batches:
        batches_dir = os.path.join(out_dir, "_batches")
        if os.path.isdir(batches_dir):
            for split in splits:
                split_dir = os.path.join(batches_dir, split)
                if os.path.isdir(split_dir):
                    for f in os.listdir(split_dir):
                        os.remove(os.path.join(split_dir, f))
                    os.rmdir(split_dir)
            if not os.listdir(batches_dir):
                os.rmdir(batches_dir)
    return counts


# ============================================================================
# Main
# ============================================================================

def build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", help="PAD-UFES-20 directory with metadata.csv and images "
                                  "(not required with --finalize_only)")
    p.add_argument("--out_dir", default="./data")
    p.add_argument("--img_size", type=int, default=224, help="resize H/W for standard RGB images")
    p.add_argument("--patch_size", type=int, default=None, help="square patch size (e.g. 11)")
    p.add_argument("--stride", type=int, default=None, help="patch stride (default: patch_size)")
    p.add_argument("--split", choices=sorted(SPLIT_PRESETS), default="80_20",
                    help="patient-level split preset. Two-way (no validation): 80_20, 70_30. "
                         "Three-way: 80_10_10, 70_15_15, 60_20_20.")
    p.add_argument("--split_strategy", choices=["stratified", "legacy_random"], default="stratified",
                    help="'stratified' (default): every class with enough distinct patients is "
                         "guaranteed in every split. 'legacy_random': plain shuffle-and-slice.")
    p.add_argument("--allow_missing_classes", action="store_true",
                    help="warn instead of aborting when a split has zero samples of a class")
    p.add_argument("--test_size", type=float, default=None,
                    help="DEPRECATED - use --split. If given (and --split left at its default) "
                         "the closest two-way preset is used.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--keep_batches", action="store_true")
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--finalize_only", action="store_true")
    p.add_argument("--no_verify_output", action="store_true",
                    help="skip the end-to-end read-back of the produced arrays (NOT recommended)")
    p.add_argument("--verify_level", choices=["structural", "deep"], default="deep",
                    help="'deep' (default): full sequential read of every produced X_*.npy.")
    return p


def _resolve_deprecated_test_size(args):
    if args.test_size is None:
        return
    if args.split != "80_20":
        print(f"  [warn] both --test_size and --split given; ignoring --test_size={args.test_size}")
        return
    frac = args.test_size
    choice = min((("80_20", 0.20), ("70_30", 0.30)), key=lambda kv: abs(kv[1] - frac))[0]
    print(f"  [deprecation] --test_size {frac} -> --split {choice} (use --split directly in future)")
    args.split = choice


def main():
    args = build_arg_parser().parse_args()
    _resolve_deprecated_test_size(args)
    os.makedirs(args.out_dir, exist_ok=True)

    if args.finalize_only:
        manifest = load_manifest(args.out_dir)
        if manifest is None:
            raise RuntimeError(f"No {PROGRESS_FILE} found in {args.out_dir}")
        # tolerate an older/plain-v4 manifest missing the newer keys
        manifest.setdefault("skipped_images", {})
        manifest["args"].setdefault("split", "80_20")
        print("Finalizing (unifying shards) only ...")
        counts = unify_all(args.out_dir, manifest, args)
        manifest["finalized"] = True
        save_manifest(args.out_dir, manifest)
        if counts is not None:
            print("Done.")
        return

    if not args.root:
        raise SystemExit("--root is required unless --finalize_only is passed")

    print(f"Scanning {args.root} ...")
    captures = discover_images(args.root, exclude_dir=args.out_dir)
    class_names = list(DIAGNOSTIC_TO_ID.keys())
    n_patients = len(sorted({c['patient'] for c in captures}))
    print(f"Found {len(captures)} labeled images across {n_patients} patients.")
    if not captures:
        raise RuntimeError("No matching images found. Ensure the dataset is unzipped under --root.")

    manifest = None if args.fresh else load_manifest(args.out_dir)
    if manifest is not None:
        check_manifest_compatible(manifest, args)
        manifest.setdefault("skipped_images", {})
        train_patients = set(manifest["train_patients"])
        val_patients = set(manifest["validation_patients"])
        test_patients = set(manifest["test_patients"])
        print(f"Resuming: {len(manifest['completed_images'])} image(s) already completed.")
    else:
        patient_to_classes = {}
        for c in captures:
            patient_to_classes.setdefault(c["patient"], set()).add(c["label"])
        if args.split_strategy == "legacy_random":
            train_patients, val_patients, test_patients, warns = patient_split_three(
                list(patient_to_classes), args.split, args.seed)
        else:
            train_patients, val_patients, test_patients, warns = stratified_patient_split_three(
                patient_to_classes, args.split, args.seed)
        for w in warns:
            print(f"  [split-warning] {w}")

        def _assign(pat):
            return "test" if pat in test_patients else "validation" if pat in val_patients else "train"

        item_labels = [(c["label"], c["patient"]) for c in captures]
        coverage = check_prep_class_coverage(item_labels, _assign, num_classes=len(class_names),
                                             class_names=class_names,
                                             fail_hard=not args.allow_missing_classes)
        save_json_atomic(os.path.join(args.out_dir, "class_coverage_report.json"), coverage)
        if coverage["coverage_ok"]:
            print("  [class-coverage] every class present in every split - OK")
        else:
            print(f"  [WARNING] class coverage: missing {coverage['missing']} "
                  f"(continuing because --allow_missing_classes)")

        manifest = new_manifest(args, class_names, train_patients, val_patients, test_patients)
        tv = f" Val={len(val_patients)}" if val_patients else ""
        print(f"Starting fresh. Train={len(train_patients)}{tv} Test={len(test_patients)} "
              f"patients (split={args.split}, strategy={args.split_strategy}).")

    cleanup_orphan_shards(args.out_dir, manifest)
    skips = SkipRegistry(manifest.get("skipped_images"))

    def assign_split(pat):
        return "test" if pat in test_patients else "validation" if pat in val_patients else "train"

    n_printed = 0
    for cap in captures:
        if cap["key"] in manifest["completed_images"] or cap["key"] in skips:
            continue
        split = assign_split(cap["patient"])
        writer = ShardWriter(os.path.join(args.out_dir, "_batches"), split,
                             manifest["next_shard_id"][split], args.batch_size)
        try:
            n_patches = process_one_image(cap["path"], args.img_size, writer, cap["label"],
                                          patch_size=args.patch_size, stride=args.stride)
        except Exception as e:  # noqa: BLE001 - one bad image must not kill a resumable run
            skips.add(cap["key"], reason=repr(e), stage="process_one_image")
            manifest["skipped_images"] = skips.as_dict()
            save_manifest(args.out_dir, manifest)
            print(f"  [skip] {cap['key']}: {e!r}")
            continue

        shards, next_id = writer.finalize()
        manifest["next_shard_id"][split] = next_id
        manifest["completed_images"][cap["key"]] = {"shards": shards, "split": split,
                                                    "n_patches": n_patches}
        save_manifest(args.out_dir, manifest)
        if n_printed < 5:
            print(f"  [{cap['key']}] processed ({n_patches} patches, split={split})")
            n_printed += 1
        elif n_printed == 5:
            print("  ... (further per-image logs suppressed)")
            n_printed += 1

    if len(skips):
        print(f"  [ingest] {skips.summary()}")

    # Fail-soft skips happen after the pre-extraction coverage check - re-verify
    # on the images that actually made it to disk before unifying.
    done = set(manifest["completed_images"])
    realized_items = [(c["label"], c["patient"]) for c in captures if c["key"] in done]
    realized_cov = check_prep_class_coverage(realized_items, assign_split, num_classes=len(class_names),
                                             class_names=class_names,
                                             fail_hard=not args.allow_missing_classes)
    save_json_atomic(os.path.join(args.out_dir, "class_coverage_report.json"), realized_cov)
    if not realized_cov["coverage_ok"]:
        print(f"  [WARNING] after skips, class coverage is incomplete: {realized_cov['missing']}")

    print("All images processed. Unifying shards into final arrays ...")
    counts = unify_all(args.out_dir, manifest, args)
    manifest["finalized"] = True
    save_manifest(args.out_dir, manifest)
    if counts is not None:
        _write_dataset_statistics(args.out_dir, manifest, counts, skips)
    print("Done.")


def _write_dataset_statistics(out_dir, manifest, counts, skips):
    has_val = bool(manifest.get("validation_patients"))
    tr, va, te = resolve_split_fractions(manifest["args"]["split"])
    total = sum(counts.values()) or 1
    stats = {
        "split": {
            "preset": manifest["args"]["split"], "strategy": manifest["args"]["split_strategy"],
            "split_algo_version": manifest["args"].get("split_algo_version"),
            "target_fractions": {"train": tr, "validation": va, "test": te},
            "patient_counts": {"train": len(manifest["train_patients"]),
                               "validation": len(manifest["validation_patients"]) if has_val else 0,
                               "test": len(manifest["test_patients"])},
            "realized_patch_counts": counts,
            "realized_patch_fractions": {k: round(v / total, 4) for k, v in counts.items()},
        },
        "class_names": manifest.get("class_names"),
        "patch_size": manifest["args"]["patch_size"], "stride": manifest["args"]["stride"],
        "img_size": manifest["args"]["img_size"], "seed": manifest["args"]["seed"],
        "skipped_images": skips.as_dict(),
    }
    save_json_atomic(os.path.join(out_dir, "dataset_statistics.json"), stats)
    print(f"  Wrote {out_dir}/dataset_statistics.json")


if __name__ == "__main__":
    main()
