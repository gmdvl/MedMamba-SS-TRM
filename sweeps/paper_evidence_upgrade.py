# -*- coding: utf-8 -*-
"""
sweeps/paper_evidence_upgrade.py - every GPU/host job of
plan/manuscript_v18_evidence_upgrade.plan.md (`sweeps/v20.py` until 2026-10-02;
ported from scripts/v20_queue.sh, deleted 2026-10-01; see archive/README.md).
The run tags keep their `v20-` prefix: finished runs recorded them, and a job
counts as done when a run with its tag exists.

    python run_experiments.py sweeps/paper_evidence_upgrade.py p1                 # all priority-1 stages, in order
    python run_experiments.py sweeps/paper_evidence_upgrade.py p2
    python run_experiments.py sweeps/paper_evidence_upgrade.py headline heldout   # individual stages
    python run_experiments.py sweeps/paper_evidence_upgrade.py --status

Environment overrides: PY, RAW_BC, RAW_LUSC, LUSC_ROOT, MEDMAMBA_DIR,
MM_SEEDS (default "1 7 42"), CV_CLEAN=1 (delete a fold's X_*.npy once both of
its runs are done).
"""

import glob
import os
from pathlib import Path

from run_experiments import PY, ROOT, cmd, needs_run, run_dir, train

RAW_BC = os.environ.get("RAW_BC", "/home/dante/Downloads/downloads/HistologyHSI-BC-Recurrence/")
RAW_LUSC = os.environ.get("RAW_LUSC", "/home/dante/Downloads/downloads/HMI-LUSC_ A Histological "
                          "Hyperspectral Imaging Dataset for Lung Squamous Cell Carcinoma - 30188080")
LUSC_ROOT = os.environ.get("LUSC_ROOT", "/data/dante_data/hmi_lusc_raw")   # /home has no room
MEDMAMBA_DIR = os.environ.get("MEDMAMBA_DIR", str(ROOT.parent / "medmamba-original/MedMamba"))
MM_SEEDS = [int(s) for s in os.environ.get("MM_SEEDS", "1 7 42").split()]

BUILD = "data/hsi_v9-trainsel"
SPLIT_V8 = "data/hsi_v8-80_10_10_importance-new/split_assignment.json"
SEEDS, CSEEDS = (1, 7, 13, 23, 42), (1, 7, 42)
REF = dict(profile="paper_recipe", batch_size=256, lambda_sam=0.1,
           train_subsample_frac=0.102, val_subsample_frac=0.0986)
NOSAM = {k: v for k, v in REF.items() if k != "lambda_sam"}
PREP_FLAGS = ["--modality", "both", "--label_source", "tissue", "--patch_size", "11", "--stride", "11",
              "--roi_min_frac", "0.8", "--band_selection", "importance", "--num_bands", "32",
              "--band_min_gap", "8", "--band_max_corr", "0.95", "--band_min_coverage", "0.30",
              "--capture_gain", "median_ratio", "--hsi_value_scale", "auto", "--rgb_source",
              "synthetic", "--seed", "42", "--num_workers", "8", "--verify_level", "deep"]
BASELINE = "scripts/train_hsi_baseline.py"


def baseline(arch, tag, data, name, **flags):
    return train(tag, data, name=name, script=BASELINE, match_arch=arch, arch=arch, **flags)


def _prep():
    return [
        cmd("prep", [PY, "prepare_histologyhsi_bc_trainsel.py", "--root", RAW_BC, "--out_dir",
                     f"./{BUILD}", "--split", "80_10_10", "--split_strategy", "stratified",
                     "--split_file", SPLIT_V8, *PREP_FLAGS],
            done_if=lambda: all(Path(f"{BUILD}/{m}/y_test.npy").exists() for m in ("hsi", "rgb"))),
        cmd("prep-val-subsets", [PY, "scripts/export_val_subset_indices.py", "--data_dir",
                                 f"{BUILD}/hsi", "--seeds", *map(str, SEEDS)],
            done_if=f"{BUILD}/hsi/val_subset_s42.npy"),
        cmd("prep-report", [PY, "-c", (
            "import json, numpy as np; "
            f"s = json.load(open('{BUILD}/pass1_scope_report.json')); "
            f"wl = np.load('{BUILD}/selected_wavelengths.npy'); "
            "print(f\"band selection saw {s['n_train_captures']} training captures, 0 of "
            "{s['n_heldout_captures']} held-out; {len(wl)} bands {wl.min():.1f}-{wl.max():.1f} nm, "
            "largest gap {np.diff(np.sort(wl)).max():.1f} nm\")")],
            requires=lambda: None if Path(f"{BUILD}/pass1_scope_report.json").exists() else "no prep yet"),
    ]


def _evalmode():
    return [cmd(f"evalmode-{Path(d).name[:15]}", [PY, "scripts/check_eval_mode.py", "--run_dir", d],
                done_if=f"{d}/eval_mode_check_v20.json")
            for pattern in ("experiments/20260915_083810_*arch-split",
                            "experiments/20260915_152009_*arch-fullchannel")
            for d in glob.glob(pattern)]


def _heldout():
    jobs = []
    for s in SEEDS:
        for m in ("hsi", "rgb"):
            tag, data = f"v20-head-s{s}", f"{BUILD}/{m}"
            jobs.append(cmd(
                f"heldout-s{s}-{m}",
                lambda tag=tag, data=data, s=s: [PY, "scripts/heldout_eval.py", "--run_dirs",
                                                 str(run_dir(tag, data))] + (["--xai"] if s == 42 else []),
                done_if=lambda tag=tag, data=data: bool(run_dir(tag, data)) and (
                    run_dir(tag, data) / "heldout_v19/validation/test_predictions.npz").exists(),
                requires=needs_run(tag, data)))
    return jobs


def _zeroshot():
    tag, data = "v20-head-s42", f"{BUILD}/hsi"
    jobs = []
    for name, script, out, bands in (
            ("zeroshot", "scripts/eval_band_decimation.py", "band_decimation", "32,16,8,4,2"),
            ("zeroshot-fixed", "scripts/eval_band_decimation_fixed_eta.py",
             "band_decimation_fixed_eta", "32,16,8,4,2")):
        jobs.append(cmd(
            name, lambda script=script, bands=bands: [PY, script, "--run_dir", str(run_dir(tag, data)),
                                                      "--keep_bands", bands],
            done_if=lambda out=out: bool(run_dir(tag, data)) and (
                run_dir(tag, data) / out / "summary.json").exists(),
            requires=needs_run(tag, data)))
    return jobs


def _cv():
    jobs = []
    for f in range(5):
        cv = f"data/hsi_v9-cv5-f{f}"
        jobs.append(cmd(f"cv-prep-f{f}", [PY, "prepare_histologyhsi_bc_trainsel.py", "--root", RAW_BC,
                                          "--out_dir", f"./{cv}", "--kfold", "5", "--fold", str(f),
                                          "--cv_val_frac", "0.125", *PREP_FLAGS],
                        done_if=f"{cv}/hsi/y_test.npy"))
        jobs += [train(f"v20-cv-f{f}", f"{cv}/{m}", name=f"cv-f{f}-{m}", **REF, epochs=20, seed=42)
                 for m in ("hsi", "rgb")]
        if os.environ.get("CV_CLEAN") == "1":
            jobs.append(cmd(
                f"cv-clean-f{f}",
                [PY, "-c", f"import glob, os; [os.remove(p) for p in glob.glob('{cv}/*/X_*.npy')]"],
                done_if=lambda cv=cv: not glob.glob(f"{cv}/*/X_*.npy"),
                requires=lambda f=f, cv=cv: None if all(
                    run_dir(f"v20-cv-f{f}", f"{cv}/{m}") for m in ("hsi", "rgb")) else "runs not done"))
    return jobs


def _lusc_prep():
    jobs = []
    for z in sorted(glob.glob(f"{RAW_LUSC}/P*.zip")):
        p = Path(z).stem
        dest = f"{LUSC_ROOT}/{p}"
        if Path(f"{RAW_LUSC}/{p}/LUSC_ROI_1").is_dir():          # already extracted in place
            argv = [PY, "-c", f"import os; os.makedirs('{LUSC_ROOT}', exist_ok=True); "
                              f"os.symlink('{RAW_LUSC}/{p}', '{dest}')"]
        else:
            argv = ["unzip", "-q", z, "-d", dest]
        jobs.append(cmd(f"lusc-unzip-{p}", argv, done_if=dest))
    jobs.append(cmd("lusc-extract", [PY, "prepare_hmi_lusc_v20.py", "extract", "--root", LUSC_ROOT,
                                     "--pool", "data/lusc_v20/pool", "--workers", "2"],
                    done_if="data/lusc_v20/pool/y_pool.npy"))
    jobs += [cmd(f"lusc-split-f{f}", [PY, "prepare_hmi_lusc_v20.py", "split", "--pool", "data/lusc_v20/pool",
                                      "--out", f"data/lusc_v20-f{f}", "--fold", str(f)],
                 done_if=f"data/lusc_v20-f{f}/hsi/y_test.npy") for f in range(5)]
    return jobs


def _lusc():
    jobs = []
    for f in range(5):
        d = f"data/lusc_v20-f{f}/hsi"
        ready = lambda d=d: None if Path(f"{d}/y_test.npy").exists() else "run lusc-prep"  # noqa: E731
        run = train(f"v20-lusc-f{f}", d, name=f"lusc-f{f}-trm", profile="paper_recipe", batch_size=256,
                    lambda_sam=0.1, train_subsample_frac=1.0, val_subsample_frac=0.25, epochs=20, seed=42)
        run.requires = ready
        jobs.append(run)
        for a in ("hybridsn", "spectralformer"):
            b = baseline(a, f"v20-lusc-f{f}", d, f"lusc-f{f}-{a}", seed=42,
                         train_subsample_frac=1.0, val_subsample_frac=0.25)
            b.requires = ready
            jobs.append(b)
    return jobs


STAGES = {
    "prep": _prep(),
    "evalmode": _evalmode(),
    "headline": [train(f"v20-head-s{s}", f"{BUILD}/{m}", name=f"head-s{s}-{m}", **REF, epochs=20, seed=s)
                 for s in SEEDS for m in ("hsi", "rgb")],
    "heldout": _heldout(),
    "probes": [cmd(f"probes-{m}", [PY, "scripts/shallow_probe_heldout.py", "--data_dir", f"{BUILD}/{m}"],
                   done_if=f"{BUILD}/{m}/shallow_probe_heldout_v19.json") for m in ("hsi", "rgb")],
    "baselines": [baseline(a, f"v20-s{s}", f"{BUILD}/hsi", f"{a}-s{s}", seed=s)
                  for a in ("hybridsn", "spectralformer") for s in SEEDS],
    "medmamba": [cmd(f"medmamba-s{s}",
                     [PY, "train_hsi_v20.py", "--data", str(ROOT / BUILD / "hsi"), "--epochs", "5",
                      "--batch_size", "256", "--lr", "1e-4", "--amp", "bf16", "--class_weights",
                      "--normalize", "global_zscore", "--norm_stats_sample_cap", "5000",
                      "--checkpoint_metric", "f1_macro", "--patch_size", "1", "--dims", "64,128,256,512",
                      "--depths", "1,1,2,1", "--seed", str(s), "--run_name", f"v20_s{s}",
                      "--val_subset_indices", str(ROOT / BUILD / f"hsi/val_subset_s{s}.npy")],
                     done_if=f"{MEDMAMBA_DIR}/runs_hsi/v20_s{s}/predictions_val_full.npz", cwd=MEDMAMBA_DIR)
                 for s in MM_SEEDS],
    "controls": [job for s in CSEEDS for job in (
        train(f"v20-abl-base-s{s}", f"{BUILD}/hsi", name=f"abl-base-s{s}", **REF, epochs=12, seed=s),
        train(f"v20-k2-s{s}", f"{BUILD}/hsi", name=f"k2-s{s}", **REF, epochs=12, seed=s,
              trm_n_latent=1, trm_n_improve=1, trm_deep_supervision_steps=1))],
    "ablations": [job for s in CSEEDS for job in (
        train(f"v20-recon-off-s{s}", f"{BUILD}/hsi", name=f"recon-off-s{s}", **NOSAM, epochs=12, seed=s,
              recon_mode="none"),
        train(f"v20-enc-index-s{s}", f"{BUILD}/hsi", name=f"enc-index-s{s}", **REF, epochs=12, seed=s,
              no_use_wavelengths=True))],
    "zeroshot": _zeroshot(),
    "bands": [job for n in (16, 8) for job in (
        cmd(f"bands{n}-build", [PY, "scripts/make_band_subset_build.py", "--src", f"{BUILD}/hsi",
                                "--keep", str(n), "--out", f"data/hsi_v9-bands{n}"],
            done_if=f"data/hsi_v9-bands{n}/y_test.npy"),
        train(f"v20-bands{n}", f"data/hsi_v9-bands{n}", name=f"bands{n}", **REF, epochs=12, seed=42))],
    "depth": [train(f"v20-depth-n{t}", f"{BUILD}/hsi", name=f"depth-n{t}", **REF, epochs=12, seed=42,
                    trm_n_improve=t) for t in (1, 2, 4)],
    "hier": [train(f"v20-arch-{a}", f"{BUILD}/hsi", name=f"arch-{a}", **REF, epochs=12, seed=42,
                   architecture=a) for a in ("split", "fullchannel")],
    "cv": _cv(),
    "lusc-prep": _lusc_prep(),
    "lusc": _lusc(),
    "analysis": [cmd("analysis", [PY, "scripts/analysis_v20.py"])],
}

GROUPS = {
    "p1": ["prep", "evalmode", "headline", "heldout", "probes", "baselines", "medmamba", "controls",
           "cv", "analysis"],
    "p2": ["ablations", "zeroshot", "bands", "depth", "hier", "lusc-prep", "lusc", "analysis"],
}
