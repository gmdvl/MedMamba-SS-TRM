# -*- coding: utf-8 -*-
"""
prepare_pad_ufes_20_v2.py
=========================
Turns a raw PAD-UFES-20 dataset download into flat X_train.npy / y_train.npy
and X_test.npy / y_test.npy files via a memory-bounded, resumable, 
batch-streamed processing pipeline. Supports patch extraction via sliding windows.
"""

import argparse
import gc
import json
import os
import random

import numpy as np
import pandas as pd
from PIL import Image

PROGRESS_FILE = "_progress.json"

# Standard class mappings for PAD-UFES-20
DIAGNOSTIC_TO_ID = {
    'ACK': 0, 
    'BCC': 1, 
    'MEL': 2, 
    'NEV': 3, 
    'SCC': 4, 
    'SEK': 5
}

# ============================================================================
# Streaming shard writer - bounds memory to ~one shard at a time
# ============================================================================

class ShardWriter:
    def __init__(self, batches_dir: str, split: str, shard_counter_start: int, batch_size: int):
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
        
        x_tmp, y_tmp = x_path + ".tmp", y_path + ".tmp"
        with open(x_tmp, "wb") as f:
            np.save(f, np.stack(self.buf_x).astype(np.float32))
        with open(y_tmp, "wb") as f:
            np.save(f, np.array(self.buf_y, dtype=np.int64))
            
        os.replace(x_tmp, x_path)
        os.replace(y_tmp, y_path)
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
    p = manifest_path(out_dir)
    tmp = p + ".tmp"
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=2)
    os.replace(tmp, p)

def new_manifest(args, class_names, test_patients):
    return {
        "args": {
            "img_size": args.img_size, 
            "patch_size": args.patch_size,
            "stride": args.stride,
            "test_size": args.test_size, 
            "seed": args.seed,
        },
        "class_names": class_names,
        "test_patients": sorted(test_patients),
        "next_shard_id": {"train": 0, "test": 0},
        "completed_images": {},
        "finalized": False,
    }

def check_manifest_compatible(manifest, args):
    saved = manifest.get("args", {})
    current = {
        "img_size": args.img_size, 
        "patch_size": args.patch_size,
        "stride": args.stride,
        "test_size": args.test_size, 
        "seed": args.seed,
    }
    mismatches = {k: (saved.get(k), v) for k, v in current.items() if saved.get(k) != v}
    if mismatches:
        raise ValueError(
            f"Found an existing progress file with DIFFERENT settings: {mismatches}. "
            f"Use a fresh --out_dir, delete _progress.json to start over, or pass --fresh."
        )

def cleanup_orphan_shards(out_dir):
    manifest = load_manifest(out_dir)
    if manifest is None:
        return
    referenced = set()
    for img in manifest["completed_images"].values():
        for entry in (img.get("shards") or []):
            referenced.add(entry[0])
            referenced.add(entry[1])

    batches_dir = os.path.join(out_dir, "_batches")
    if not os.path.isdir(batches_dir):
        return
    for split in ("train", "test"):
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

def discover_images(root):
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

    img_col = col_map['img_id']
    patient_col = col_map['patient_id']
    diag_col = col_map['diagnostic']

    image_files = {}
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                stem = os.path.splitext(f)[0].strip()
                image_files[stem] = os.path.join(dirpath, f)

    captures = []
    skipped = 0
    for _, row in df.iterrows():
        raw_img_id = str(row[img_col]).strip()
        clean_stem = os.path.splitext(raw_img_id)[0].strip()
        
        if clean_stem not in image_files:
            skipped += 1
            continue
            
        diag = str(row[diag_col]).strip().upper()
        if diag not in DIAGNOSTIC_TO_ID:
            continue
            
        captures.append({
            "key": clean_stem,
            "path": image_files[clean_stem],
            "patient": str(row[patient_col]).strip(),
            "label": DIAGNOSTIC_TO_ID[diag]
        })
        
    if skipped > 0:
        print(f"  [warn] Skipped {skipped} images listed in metadata that were not found on disk.")
        
    return captures

def patient_split(captures, test_size, seed):
    patients = sorted({c["patient"] for c in captures})
    rng = random.Random(seed)
    rng.shuffle(patients)
    n_test = max(1, int(round(len(patients) * test_size)))
    return set(patients[:n_test])

def process_one_image(img_path, target_size, writer, label, patch_size=None, stride=None):
    img = Image.open(img_path).convert("RGB")
    if target_size is not None and target_size > 0:
        img = img.resize((target_size, target_size), Image.Resampling.BILINEAR)
    
    img_array = np.asarray(img, dtype=np.float32) / 255.0  # Scale RGB to [0, 1]

    # Sliding window patch extraction mode
    if patch_size is not None and patch_size > 0:
        step = stride if (stride is not None and stride > 0) else patch_size
        h, w, _ = img_array.shape
        count = 0
        for y in range(0, h - patch_size + 1, step):
            for x in range(0, w - patch_size + 1, step):
                patch = img_array[y : y + patch_size, x : x + patch_size, :]
                writer.add(patch, label)
                count += 1
        return count
    else:
        # Standard full-image mode
        writer.add(img_array, label)
        return 1

# ============================================================================
# Unify
# ============================================================================

def unify_split(shard_entries, out_x_path, out_y_path):
    if not shard_entries:
        return 0
    total_n = sum(n for _, _, n in shard_entries)
    sample = np.load(shard_entries[0][0], mmap_mode="r")
    patch_shape = sample.shape[1:]
    dtype = sample.dtype
    del sample

    out_x = np.lib.format.open_memmap(out_x_path, mode="w+", dtype=dtype, shape=(total_n,) + patch_shape)
    all_y = np.empty((total_n,), dtype=np.int64)

    offset = 0
    for x_path, y_path, n in shard_entries:
        xb = np.load(x_path, mmap_mode="r")
        out_x[offset:offset + n] = xb[:]
        all_y[offset:offset + n] = np.load(y_path)
        offset += n
        del xb
    out_x.flush()
    del out_x
    np.save(out_y_path, all_y)
    return total_n

def unify_all(out_dir, manifest, delete_batches):
    if manifest.get("finalized"):
        print("Already finalized. Shards are deleted. Exiting.")
        return

    train_shards, test_shards = [], []
    for img in manifest["completed_images"].values():
        entries = img.get("shards")
        if not entries:
            continue
        
        split = img.get("split")
        if not split and len(entries) > 0:
            split = "train" if "/train/" in entries[0][0] else "test"

        (train_shards if split == "train" else test_shards).extend(entries)

    n_train = unify_split(train_shards, os.path.join(out_dir, "X_train.npy"), os.path.join(out_dir, "y_train.npy"))
    n_test = unify_split(test_shards, os.path.join(out_dir, "X_test.npy"), os.path.join(out_dir, "y_test.npy"))
    
    print(f"  Unified: train={n_train} test={n_test} samples/patches -> {out_dir}/")

    if delete_batches:
        batches_dir = os.path.join(out_dir, "_batches")
        if os.path.isdir(batches_dir):
            for split in ("train", "test"):
                split_dir = os.path.join(batches_dir, split)
                if os.path.isdir(split_dir):
                    for f in os.listdir(split_dir):
                        os.remove(os.path.join(split_dir, f))
                    os.rmdir(split_dir)
            if not os.listdir(batches_dir):
                os.rmdir(batches_dir)

# ============================================================================
# Main
# ============================================================================

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True, help="Path to PAD-UFES-20 directory containing metadata.csv and images")
    p.add_argument("--out_dir", default="./data")
    p.add_argument("--img_size", type=int, default=224, help="Target H/W dimension to resize standard RGB images")
    p.add_argument("--patch_size", type=int, default=None, help="Size of square patches to extract (e.g., 11)")
    p.add_argument("--stride", type=int, default=None, help="Stride step for patch extraction (default: patch_size)")
    p.add_argument("--test_size", type=float, default=0.2, help="Fraction of PATIENTS held out for test")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--keep_batches", action="store_true")
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--finalize_only", action="store_true")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.finalize_only:
        manifest = load_manifest(args.out_dir)
        if manifest is None:
            raise RuntimeError(f"No {PROGRESS_FILE} found in {args.out_dir}")
        print("Finalizing (unifying shards) only ...")
        unify_all(args.out_dir, manifest, delete_batches=not args.keep_batches)
        manifest["finalized"] = True
        save_manifest(args.out_dir, manifest)
        print("Done.")
        return

    print(f"Scanning {args.root} ...")
    captures = discover_images(args.root)
    class_names = list(DIAGNOSTIC_TO_ID.keys())
    print(f"Found {len(captures)} labeled images across {len(sorted({c['patient'] for c in captures}))} patients.")

    if len(captures) == 0:
        raise RuntimeError("No matching images found. Ensure your dataset images are unzipped and located under --root.")

    manifest = None if args.fresh else load_manifest(args.out_dir)
    if manifest is not None:
        check_manifest_compatible(manifest, args)
        test_patients = set(manifest["test_patients"])
        n_done = len(manifest["completed_images"])
        print(f"Resuming: {n_done} image(s) already completed.")
    else:
        test_patients = patient_split(captures, args.test_size, args.seed)
        manifest = new_manifest(args, class_names, test_patients)

    cleanup_orphan_shards(args.out_dir)

    n_printed = 0
    for cap in captures:
        if cap["key"] in manifest["completed_images"]:
            continue

        split = "test" if cap["patient"] in test_patients else "train"
        writer = ShardWriter(os.path.join(args.out_dir, "_batches"), split,
                             manifest["next_shard_id"][split], args.batch_size)

        n_patches = process_one_image(
            cap["path"], 
            args.img_size, 
            writer, 
            cap["label"],
            patch_size=args.patch_size,
            stride=args.stride
        )
        shards, next_id = writer.finalize()

        manifest["next_shard_id"][split] = next_id
        manifest["completed_images"][cap["key"]] = {"shards": shards, "split": split, "n_patches": n_patches}
        save_manifest(args.out_dir, manifest) 

        if n_printed < 5:
            print(f"  [{cap['key']}] processed ({n_patches} patches, split={split})")
            n_printed += 1
        elif n_printed == 5:
            print("  ... (further per-image logs suppressed)")
            n_printed += 1

    print(f"All images processed. Unifying shards into final arrays ...")
    unify_all(args.out_dir, manifest, delete_batches=not args.keep_batches)
    manifest["finalized"] = True
    save_manifest(args.out_dir, manifest)
    print("Done.")

if __name__ == "__main__":
    main()