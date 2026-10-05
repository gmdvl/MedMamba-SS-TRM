# -*- coding: utf-8 -*-
"""
prepare_hmi_lusc_v20.py
=======================
HMI-LUSC (histological hyperspectral imaging of lung squamous cell carcinoma)
-> the flat NHWC .npy layout every trainer in this repo reads. It is the
SECOND hyperspectral histology dataset of manuscript v18: a different sensor
(61 bands, 450-750 nm, 5 nm steps) from HistologyHSI-BC-Recurrence (740 bands,
400-938 nm), a different organ, and pixel-level tumour masks.

Raw layout (one folder per patient, zips extracted in place):

    P<k>/whiteReference(.hdr)  P<k>/darkReference(.hdr)       per-patient references
    P<k>/LUSC_ROI_<r>/Raw(.hdr)                               uint8 BSQ, 61 x 2064 x 3088
    P<k>/LUSC_ROI_<r>/Label.png                               0 = non-tumour, 255 = tumour

Two sub-commands:

  extract   calibrate every ROI, (raw - dark) / (white - dark) clipped to [0, 1.5];
            tile it into P x P patches on a stride; label a patch TUMOUR if at
            least --purity of its mask pixels are tumour and NON-TUMOUR if at most
            1 - --purity are; drop mixed patches and, in both classes, glass
            patches whose mean reflectance exceeds --glass_max (otherwise
            "non-tumour" would largely mean "empty slide"); keep at most
            --per_class_cap patches per class per ROI (seeded draw).
            -> <pool>/{X_pool.npy (float16 N,P,P,61), y_pool, patient_pool,
               roi_pool, pos_pool, tumour_frac_pool, wavelengths.npy, pool_report.json}
            No label enters any statistic except the per-patch class assignment.

  split     patient-level k-fold: patients are permuted once (--seed); fold f
            holds out patients [2f, 2f+1] (10 patients, k = 5) as TEST and the
            next patient in the permutation as VALIDATION; the other seven train.
            -> <out>/hsi/{X,y,groups,captures}_{train,val,test}.npy,
               wavelengths.npy, class_names.json, split_assignment.json

Usage (host):
    python prepare_hmi_lusc_v20.py extract --root "<HMI-LUSC dir>" --pool data/lusc_v20/pool
    for f in 0 1 2 3 4; do
      python prepare_hmi_lusc_v20.py split --pool data/lusc_v20/pool --out data/lusc_v20-f$f --fold $f
    done
"""
from __future__ import annotations

import argparse
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

CLASS_NAMES = ["non-tumour", "tumour"]


# --------------------------------------------------------------------------- ENVI (no `spectral` needed)
def read_envi_header(hdr: Path) -> dict:
    text = hdr.read_text(errors="replace")
    out = {}
    for key in ("samples", "lines", "bands", "data type", "header offset", "byte order"):
        m = re.search(rf"^{key}\s*=\s*(\d+)", text, re.M)
        if m:
            out[key] = int(m.group(1))
    m = re.search(r"^interleave\s*=\s*(\w+)", text, re.M)
    out["interleave"] = m.group(1).lower() if m else "bsq"
    m = re.search(r"^wavelength\s*=\s*\{([^}]*)\}", text, re.M | re.S)
    out["wavelength"] = [float(v) for v in m.group(1).split(",")] if m else None
    return out


def open_bsq(path: Path) -> np.ndarray:
    h = read_envi_header(Path(str(path) + ".hdr"))
    if h.get("data type") != 1 or h["interleave"] != "bsq":
        raise ValueError(f"{path}: expected uint8 BSQ, got type {h.get('data type')} / {h['interleave']}")
    return np.memmap(path, dtype=np.uint8, mode="r", offset=h.get("header offset", 0),
                     shape=(h["bands"], h["lines"], h["samples"]))


def box_mean(img: np.ndarray, p: int, ys: np.ndarray, xs: np.ndarray) -> np.ndarray:
    """Mean of img over the p x p boxes with top-left corners (ys, xs), via an integral image."""
    ii = np.pad(img.astype(np.float64), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    s = ii[ys + p, xs + p] - ii[ys, xs + p] - ii[ys + p, xs] + ii[ys, xs]
    return s / (p * p)


# --------------------------------------------------------------------------- extract
def discover(root: Path):
    rois = []
    for pdir in sorted(root.glob("P*"), key=lambda d: int(re.sub(r"\D", "", d.name) or 0)):
        if not pdir.is_dir():
            continue
        pid = int(re.sub(r"\D", "", pdir.name))
        for rdir in sorted(pdir.glob("LUSC_ROI_*"), key=lambda d: int(d.name.rsplit("_", 1)[-1])):
            if (rdir / "Raw").is_file() and (rdir / "Label.png").is_file():
                rois.append((pid, pdir, rdir))
    return rois


def extract_patient(task):
    pid, pdir, rdirs, a = task
    from PIL import Image
    white = np.asarray(open_bsq(pdir / "whiteReference"), dtype=np.float32)
    dark = np.asarray(open_bsq(pdir / "darkReference"), dtype=np.float32)
    denom = np.maximum(white - dark, 1.0)
    del white
    rng = np.random.default_rng(a["seed"] * 1000 + pid)
    out = []
    for rdir in rdirs:
        roi = int(rdir.name.rsplit("_", 1)[-1])
        raw = np.asarray(open_bsq(rdir / "Raw"), dtype=np.float32)
        refl = np.clip((raw - dark) / denom, 0.0, 1.5)                      # C,H,W
        del raw
        mask = (np.asarray(Image.open(rdir / "Label.png").convert("L")) > 127).astype(np.float32)
        _, H, W = refl.shape
        p, s = a["patch"], a["stride"]
        gy, gx = np.meshgrid(np.arange(0, H - p + 1, s), np.arange(0, W - p + 1, s), indexing="ij")
        ys, xs = gy.ravel(), gx.ravel()
        frac = box_mean(mask, p, ys, xs)
        bright = box_mean(refl.mean(0), p, ys, xs)
        tissue = bright <= a["glass_max"]
        keep = []
        for cls, sel in ((1, (frac >= a["purity"]) & tissue), (0, (frac <= 1 - a["purity"]) & tissue)):
            idx = np.flatnonzero(sel)
            if idx.size > a["per_class_cap"]:
                idx = np.sort(rng.choice(idx, a["per_class_cap"], replace=False))
            keep += [(cls, i) for i in idx]
        patches = np.stack([refl[:, ys[i]:ys[i] + p, xs[i]:xs[i] + p].transpose(1, 2, 0)
                            for _, i in keep]).astype(np.float16) if keep else None
        out.append({"patient": pid, "roi": roi, "X": patches,
                    "y": np.array([c for c, _ in keep], np.int64),
                    "pos": np.array([(ys[i], xs[i]) for _, i in keep], np.int32).reshape(-1, 2),
                    "frac": np.array([frac[i] for _, i in keep], np.float32),
                    "n_candidates": {"tumour": int(((frac >= a["purity"]) & tissue).sum()),
                                     "non_tumour": int(((frac <= 1 - a["purity"]) & tissue).sum()),
                                     "glass_dropped": int((~tissue).sum())}})
        print(f"  P{pid} ROI {roi}: {len(keep)} patches "
              f"(tumour {int(sum(c for c, _ in keep))})", flush=True)
    return out


def cmd_extract(args):
    root, pool = Path(args.root), Path(args.pool)
    pool.mkdir(parents=True, exist_ok=True)
    rois = discover(root)
    if not rois:
        raise SystemExit(f"no P*/LUSC_ROI_*/Raw under {root}: extract the patient zips first")
    wl = read_envi_header(rois[0][2] / "Raw.hdr")["wavelength"]
    by_patient = {}
    for pid, pdir, rdir in rois:
        by_patient.setdefault(pid, (pdir, []))[1].append(rdir)
    a = {k: getattr(args, k) for k in ("patch", "stride", "purity", "glass_max", "per_class_cap", "seed")}
    tasks = [(pid, pdir, rdirs, a) for pid, (pdir, rdirs) in sorted(by_patient.items())]
    print(f"[extract] {len(rois)} ROIs from {len(tasks)} patients, {len(wl)} bands "
          f"{wl[0]:g}-{wl[-1]:g} nm", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        results = [r for rs in ex.map(extract_patient, tasks) for r in rs]
    results = [r for r in results if r["X"] is not None]
    n = sum(len(r["y"]) for r in results)
    X = np.lib.format.open_memmap(pool / "X_pool.npy", mode="w+", dtype=np.float16,
                                  shape=(n, args.patch, args.patch, len(wl)))
    cur = 0
    for r in results:
        X[cur:cur + len(r["y"])] = r["X"]
        cur += len(r["y"])
    X.flush()
    cat = lambda k: np.concatenate([r[k] for r in results])  # noqa: E731
    np.save(pool / "y_pool.npy", cat("y"))
    np.save(pool / "patient_pool.npy", np.concatenate([np.full(len(r["y"]), r["patient"], np.int32)
                                                        for r in results]))
    np.save(pool / "roi_pool.npy", np.concatenate([np.full(len(r["y"]), r["patient"] * 100 + r["roi"],
                                                            np.int32) for r in results]))
    np.save(pool / "pos_pool.npy", cat("pos"))
    np.save(pool / "tumour_frac_pool.npy", cat("frac"))
    np.save(pool / "wavelengths.npy", np.asarray(wl, np.float32))
    report = {"n_patches": int(n), "bands": len(wl), "wavelength_nm": [wl[0], wl[-1]],
              "params": a, "class_names": CLASS_NAMES,
              "per_roi": [{"patient": r["patient"], "roi": r["roi"], "n": int(len(r["y"])),
                           "tumour": int(r["y"].sum()), **r["n_candidates"]} for r in results]}
    (pool / "pool_report.json").write_text(json.dumps(report, indent=1))
    y = cat("y")
    print(f"[extract] {n} patches, tumour {int(y.sum())}, non-tumour {int((y == 0).sum())} -> {pool}")
    return 0


# --------------------------------------------------------------------------- split
def fold_assignment(patients, fold: int, k: int, seed: int):
    perm = [int(p) for p in np.random.default_rng(seed).permutation(sorted(patients))]
    n_test = len(perm) // k
    test = perm[fold * n_test:(fold + 1) * n_test]
    val = [perm[((fold + 1) * n_test) % len(perm)]]
    train = [p for p in perm if p not in test and p not in val]
    return train, val, test


def cmd_split(args):
    pool, out = Path(args.pool), Path(args.out) / "hsi"
    out.mkdir(parents=True, exist_ok=True)
    y = np.load(pool / "y_pool.npy")
    pat = np.load(pool / "patient_pool.npy")
    roi = np.load(pool / "roi_pool.npy")
    X = np.load(pool / "X_pool.npy", mmap_mode="r")
    train, val, test = fold_assignment(np.unique(pat), args.fold, args.kfold, args.seed)
    for name, members in (("train", train), ("val", val), ("test", test)):
        idx = np.flatnonzero(np.isin(pat, members))
        dst = np.lib.format.open_memmap(out / f"X_{name}.npy", mode="w+", dtype=np.float16,
                                        shape=(len(idx),) + X.shape[1:])
        for i in range(0, len(idx), 8192):
            dst[i:i + 8192] = X[idx[i:i + 8192]]
        dst.flush()
        np.save(out / f"y_{name}.npy", y[idx])
        np.save(out / f"groups_{name}.npy", pat[idx].astype(np.int32))
        np.save(out / f"captures_{name}.npy", roi[idx].astype(np.int32))
        print(f"  {name:5s} patients {members}: {len(idx)} patches, tumour {int(y[idx].sum())}")
    np.save(out / "wavelengths.npy", np.load(pool / "wavelengths.npy"))
    (out / "class_names.json").write_text(json.dumps(CLASS_NAMES))
    (out.parent / "split_assignment.json").write_text(json.dumps(
        {"strategy": "patient_kfold", "kfold": args.kfold, "fold": args.fold, "seed": args.seed,
         "groups": {**{str(p): "train" for p in train}, **{str(p): "validation" for p in val},
                    **{str(p): "test" for p in test}}}, indent=1))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--root", required=True)
    e.add_argument("--pool", required=True)
    e.add_argument("--patch", type=int, default=11)
    e.add_argument("--stride", type=int, default=11)
    e.add_argument("--purity", type=float, default=0.9)
    e.add_argument("--glass_max", type=float, default=0.85)
    e.add_argument("--per_class_cap", type=int, default=3000)
    e.add_argument("--seed", type=int, default=42)
    e.add_argument("--workers", type=int, default=2,
                   help="patients in parallel; each holds ~6 GB (dark, white - dark, one ROI in float32)")
    s = sub.add_parser("split")
    s.add_argument("--pool", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--fold", type=int, required=True)
    s.add_argument("--kfold", type=int, default=5)
    s.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)
    return cmd_extract(args) if args.cmd == "extract" else cmd_split(args)


if __name__ == "__main__":
    raise SystemExit(main())
