# -*- coding: utf-8 -*-
"""
prepare_pad_ufes_20_v6.py
=========================
PAD-UFES-20 (https://data.mendeley.com/datasets/zr7vgbcyr2/1) -> flat
``X_{train,val,test}.npy`` / ``y_*.npy`` via the generic
``training.prep`` core (parallel extraction, spanning shards, crash-safe
unify, flexible splits).

``prepare_pad_ufes_20_v5.py`` stays FROZEN for reproducing existing
datasets. v6 = a thin :class:`training.prep.DatasetAdapter` + the shared
core; new here vs v5:

  * multi-process decode (``--num_workers``), shards that span images
    (no more one 1-row shard per whole image), batched manifest
    checkpoints, a much cheaper finalize (folded sha256 + probe verify).
  * EXIF-orientation fix (``ImageOps.exif_transpose``) and fast JPEG
    down-scale decode (``Image.draft``).
  * flexible ``--split`` (``75/15/10`` as well as presets), grouped
    ``--kfold``, reusable ``--split_file``; ``class_names.json`` is written.
  * ``--store_dtype {float32,float16,uint8}`` (default float32).

Usage::

    python prepare_pad_ufes_20_v6.py --root /path/to/PAD-UFES-20 --out_dir ./data_pad \\
        --patch_size 11 --stride 11 --split 80/10/10 --num_workers 8
"""

from __future__ import annotations

import os
from typing import Iterator, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from training.prep import DatasetAdapter, ModalitySpec, PrepItem, run_prep

DIAGNOSTIC_TO_ID = {"ACK": 0, "BCC": 1, "MEL": 2, "NEV": 3, "SCC": 4, "SEK": 5}
_IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
_JPEG_EXT = (".jpg", ".jpeg")


class PadUfes20Adapter(DatasetAdapter):
    name = "pad_ufes_20"
    flat_output = True

    @property
    def class_names(self) -> List[str]:
        return list(DIAGNOSTIC_TO_ID)

    def modalities(self) -> List[str]:
        return ["image"]

    def add_cli_args(self, p) -> None:
        p.add_argument("--img_size", type=int, default=224,
                        help="resize H/W (0 = keep native; only valid with --patch_size)")
        p.add_argument("--patch_size", type=int, default=None, help="square patch side (e.g. 11)")
        p.add_argument("--stride", type=int, default=None, help="patch stride (default: patch_size)")
        p.add_argument("--test_size", type=float, default=None,
                        help="DEPRECATED - use --split. Mapped to the nearest 2-way preset "
                             "only if --split is left at its default.")

    def preprocess_args(self, args) -> None:
        if args.test_size is None:
            return
        if args.split != "80_20":
            print(f"  [warn] both --test_size and --split given; ignoring --test_size={args.test_size}")
            return
        choice = min((("80_20", 0.20), ("70_30", 0.30)),
                     key=lambda kv: abs(kv[1] - args.test_size))[0]
        print(f"  [deprecation] --test_size {args.test_size} -> --split {choice}")
        args.split = choice

    def compat_keys(self, args) -> dict:
        return {"img_size": args.img_size, "patch_size": args.patch_size, "stride": args.stride}

    # ---- discovery -----------------------------------------------------
    def discover(self, args) -> Iterator[PrepItem]:
        root = args.root
        meta_path = os.path.join(root, "metadata.csv")
        if not os.path.exists(meta_path):
            csvs = [f for f in os.listdir(root) if f.lower().endswith(".csv")]
            if not csvs:
                raise FileNotFoundError(f"no metadata.csv (or any .csv) in {root}")
            meta_path = os.path.join(root, csvs[0])
        df = pd.read_csv(meta_path)
        cmap = {c.lower().strip(): c for c in df.columns}
        for req in ("img_id", "patient_id", "diagnostic"):
            if req not in cmap:
                raise ValueError(f"metadata.csv missing column {req!r}. Found: {list(df.columns)}")
        img_col, pat_col, diag_col = cmap["img_id"], cmap["patient_id"], cmap["diagnostic"]

        exclude = os.path.abspath(args.out_dir) if args.out_dir else None
        stem_to_path, ncol = {}, 0
        for dp, _dn, fns in os.walk(root):
            if exclude and os.path.abspath(dp).startswith(exclude):
                continue
            for f in fns:
                if f.lower().endswith(_IMG_EXT):
                    stem = os.path.splitext(f)[0].strip()
                    full = os.path.join(dp, f)
                    if stem in stem_to_path and stem_to_path[stem] != full:
                        ncol += 1
                        continue
                    stem_to_path[stem] = full
        if ncol:
            print(f"  [warn] {ncol} image(s) ignored due to filename-stem collisions.")

        missing = 0
        for _i, row in df.iterrows():
            stem = os.path.splitext(str(row[img_col]).strip())[0].strip()
            path = stem_to_path.get(stem)
            if path is None:
                missing += 1
                continue
            diag = str(row[diag_col]).strip().upper()
            if diag not in DIAGNOSTIC_TO_ID:
                continue
            yield PrepItem(key=stem, group_id=str(row[pat_col]).strip(),
                            label=DIAGNOSTIC_TO_ID[diag], payload={"path": path})
        if missing:
            print(f"  [warn] {missing} metadata row(s) had no image on disk.")

    # ---- per-item extraction ----------------------------------------
    def modality_spec(self, modality, args, state) -> ModalitySpec:
        as_uint8 = (args.store_dtype == "uint8")
        side = args.patch_size if args.patch_size else (args.img_size if args.img_size > 0 else None)
        shape = (side, side, 3) if side else None
        return ModalitySpec(store_dtype="float32", value_clip=None if as_uint8 else (0.0, 1.0),
                             sample_shape=shape)

    def expected_sample_count(self, item, modality, args, state):
        if not args.patch_size:
            return 1
        if args.img_size and args.img_size > 0:
            step = args.stride or args.patch_size
            n = (args.img_size - args.patch_size) // step + 1
            return max(0, n) ** 2
        return None  # native size unknown without decoding

    def load_item(self, item, modality, args, state) -> Iterator[Tuple[np.ndarray, int]]:
        path = item.payload["path"]
        im = Image.open(path)
        if path.lower().endswith(_JPEG_EXT) and args.img_size and args.img_size > 0:
            im.draft("RGB", (args.img_size, args.img_size))     # fast down-scale decode
        im = ImageOps.exif_transpose(im).convert("RGB")
        if args.img_size and args.img_size > 0:
            im = im.resize((args.img_size, args.img_size), Image.Resampling.BILINEAR)
        arr = np.asarray(im, dtype=np.float32)
        if args.store_dtype != "uint8":
            arr = arr / 255.0

        ps = args.patch_size
        if not ps:
            yield arr, item.label
            return
        step = args.stride or ps
        h, w, _ = arr.shape
        for y in range(0, h - ps + 1, step):
            for x in range(0, w - ps + 1, step):
                yield arr[y:y + ps, x:x + ps, :], item.label


def main() -> int:
    return run_prep(PadUfes20Adapter())


if __name__ == "__main__":
    raise SystemExit(main())
