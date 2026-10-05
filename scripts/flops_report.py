#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/flops_report.py
=============================
v18 F2 - write the FLOP accounting that no run in this repository has ever
produced, for finished runs and for bare architecture presets.

Why this exists
---------------
`training/trainerg_v11.py:717` passes `sample_x.shape[1:]` - a 3-D
`(C, H, W)` - to `training.flops_counter.count_flops`, which builds
`torch.randn(*input_shape)` and hands a 3-D tensor to a model that needs 4.
The resulting `ValueError` is caught and stored, so EVERY `test_report.json`
in `experiments/` carries `"flops": {"error": "ValueError: not enough values
to unpack (expected 4, got 3)"}`. That is the sole reason manuscript section
7.6 - the compute-cost inversion, the paper's most transferable result - is
tagged *(prior)* instead of *(verified)*.

`trainerg_v11.py` and `flops_counter.py` are both FROZEN. Neither is edited:
the arithmetic is imported and reused, and the corrected call lives in
`training/flops_report.py`, shared with `TrainerG_v18` so new runs and old
runs are counted by one implementation.

Read-only with respect to a run: it writes `<run_dir>/flops_report.json` and,
only with `--write_into_test_report`, replaces the `flops` error block inside
`test_report.json`. Nothing else in the run directory is touched.

Usage
-----
    # one finished run
    python scripts/flops_report.py --run_dir experiments/<run>

    # every finished run
    for r in experiments/2026*/; do
        python scripts/flops_report.py --run_dir "$r"
    done

    # bare presets - the hierarchical-vs-recursive comparison ON THE SAME
    # TASK that section 7.6 has never had, and the numbers behind section 3.6
    python scripts/flops_report.py \
        --architectures split,fullchannel,efficient,recursive \
        --modality hsi --num_classes 3 --in_channels 32 \
        --out findings/flops_architectures_hsi.json
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.config_presets import _PRESETS, build_model, v15_config_overrides
from training.flops_report import build_flops_report

# Reuse the model reconstruction the test-split script already solved, rather
# than keeping a second copy of `train_example_v16._main`'s construction in
# sync with it.
from scripts.eval_test_split import _rebuild_model  # noqa: E402

# The representation settings every v15+ run uses. Only the bare-preset mode
# needs these; the run mode takes them from the run's own `config.json`.
_V15_BASE_OVERRIDES = {
    "spectral_token_fusion": "concat_mlp", "spectral_pe_gain": 0.1,
    "spectral_value_init_std": 0.5, "spectral_ctx_norm": True,
    "classifier_init": "fan_in", "wavelength_encoding_scale": None,
    "spectral_chunk_size": 1024,
}
_TRM_REPORTED = dict(
    trm_dim=128, trm_core_layers=2, trm_n_latent=6, trm_n_improve=3,
    trm_deep_supervision_steps=3, trm_mixer="mlp", trm_ema_rate=0.9995,
    trm_act_halting=False, trm_halt_threshold=None, trm_halt_exploration_prob=0.1,
    trm_spatial_pe_gain=0.1, trm_checkpoint_core=True, trm_mixer_channel_mlp=True,
    trm_drop_path=0.0, trm_dropout=0.0,
)


def _patch_hw(data_dir: Path, fallback=(11, 11)):
    """`(H, W)` from the dataset's own arrays - `(N, H, W, C)` - or the
    fallback when the build is not reachable from here."""
    for name in ("X_test.npy", "X_val.npy", "X_train.npy"):
        path = data_dir / name
        if path.is_file():
            shape = np.load(path, mmap_mode="r").shape
            if len(shape) == 4:
                return int(shape[1]), int(shape[2])
    return fallback


def _rebuild_model_lenient(cfg: dict):
    """`_rebuild_model`, with the leniency the training entry point had.

    `OPTIMAL_DEFAULTS` sets `classifier_dropout: 0.1` for every architecture, and
    `train_example_v16_optimal.build_model_optimal` DROPS it with a printed
    warning on split/fullchannel (whose `medmamba_ss_trm.ClassificationHead` hardcodes
    its own rate). `scripts/eval_test_split._rebuild_model` predates that and
    passes the value straight through, so rebuilding any hierarchical run from its
    own config.json raises

        ValueError: --classifier_dropout only applies to --architecture efficient
        or recursive

    even though the run trained perfectly well. Mirror the entry point rather than
    fail on a run we are only measuring.
    """
    a = cfg.get("cli_args", {})
    if a.get("classifier_dropout") is not None and \
            cfg.get("architecture") not in ("efficient", "recursive"):
        cfg = copy.deepcopy(cfg)
        cfg["cli_args"]["classifier_dropout"] = None

    # `_rebuild_model` builds `cfg_overrides` from the representation flags only
    # and never looks at `--patch_size`, so it silently uses the PRESET's stem
    # stride. The PAD whole-image runs resolve theirs from `--target_token_grid
    # 28` and record `patch_size: 8` against the rgb preset's 4 - a 2x stride is
    # 4x the tokens, and the FLOP count comes out ~3.8x too high (9.09 GF against
    # the true 2.40 GF). Parameter counts are unaffected, which is exactly why it
    # goes unnoticed. Patch the preset entry for the duration of the build rather
    # than forking `_rebuild_model`, so there stays one construction path.
    patch_size = a.get("patch_size")
    key = (cfg.get("architecture"), cfg.get("modality"))
    if patch_size and key in _PRESETS and _PRESETS[key].get("patch_size") != patch_size:
        saved = dict(_PRESETS[key])
        _PRESETS[key] = {**saved, "patch_size": int(patch_size)}
        try:
            return _rebuild_model(cfg)
        finally:
            _PRESETS[key] = saved
    return _rebuild_model(cfg)


def report_for_run(run_dir: Path, device: str, data_dir_override=None) -> dict:
    cfg = json.load(open(run_dir / "config.json"))
    a = cfg["cli_args"]
    data_dir = Path(data_dir_override or a["data_dir"])

    model = _rebuild_model_lenient(cfg)
    height, width = _patch_hw(data_dir)

    wavelengths = None
    wl_path = data_dir / "wavelengths.npy"
    if a.get("use_wavelengths") and wl_path.is_file():
        wavelengths = np.load(wl_path)

    report = build_flops_report(
        model, in_channels=cfg["in_channels"], patch_hw=(height, width),
        cli_args=a, device=device, wavelengths=wavelengths,
        sensor_range=(tuple(a["sensor_range"]) if a.get("sensor_range") else None),
        backbone_num_params=cfg.get("backbone_num_params"))

    report["run"] = run_dir.name
    report["data_dir"] = str(data_dir)
    report["modality"] = cfg.get("modality")
    report["num_classes"] = cfg.get("classes")

    # Carry across the measured numbers that belong beside the counted ones,
    # so section 7.6's panel can be assembled from one file per run.
    test_report_path = run_dir / "test_report.json"
    if test_report_path.is_file():
        eff = json.load(open(test_report_path)).get("efficiency", {})
        report["measured_efficiency"] = {
            k: eff.get(k) for k in
            ("latency_bs1_ms_mean", "latency_bs16_ms_mean",
             "throughput_images_per_sec", "peak_gpu_memory_mb", "param_memory_mb")}
    return report


def report_for_preset(architecture: str, modality: str, num_classes: int, in_channels: int,
                      patch_hw, device: str, wavelengths=None, patch_size=None) -> dict:
    cfg_overrides = dict(v15_config_overrides(architecture))
    cfg_overrides.update(_V15_BASE_OVERRIDES)
    # The preset's own `patch_size` is not always what a run used: the PAD
    # whole-image runs resolve it from `--target_token_grid` and record
    # `patch_size: 8` (a 28x28 token grid from a 224x224 image), against the
    # recursive preset's 1. Counting at the preset value would overstate the
    # recursive model's PAD cost by the square of the ratio - 64x here - so the
    # caller must be able to pass the value the run actually used.
    if patch_size is not None:
        cfg_overrides["patch_size"] = int(patch_size)
    is_recursive = architecture == "recursive"
    model = build_model(
        architecture, modality, num_classes,
        drop_path_rate=0.0,
        # 'se_gate' is registered only on MedMambaSSEfficient's own fusion list and
        # is NOT valid at build_model, despite appearing in older config.json files.
        fusion_type=("gated" if architecture == "efficient" else None),
        # split/fullchannel use medmamba_ss_trm.ClassificationHead, whose dropout is
        # hardcoded; passing the flag to them raises.
        classifier_dropout=(0.1 if architecture in ("efficient", "recursive") else None),
        trm_kwargs=(dict(_TRM_REPORTED) if is_recursive else None),
        cfg_overrides=cfg_overrides)

    cli_args = {"architecture": architecture}
    if is_recursive:
        cli_args.update({k: _TRM_REPORTED[k] for k in
                         ("trm_dim", "trm_core_layers", "trm_n_latent", "trm_n_improve",
                          "trm_deep_supervision_steps", "trm_mixer")})
    n_params = sum(p.numel() for p in model.parameters())
    report = build_flops_report(model, in_channels=in_channels, patch_hw=patch_hw,
                               cli_args=cli_args, device=device, wavelengths=wavelengths,
                               backbone_num_params=n_params)
    report.update({"preset": architecture, "modality": modality,
                   "num_classes": num_classes, "in_channels": in_channels})
    return report


def _print_summary(report: dict, label: str) -> None:
    print(f"  {label:14s} params {report['wrapped_num_params']:>12,}   "
          f"conv/linear {report['conv_linear_flops_approx']/1e6:>9.1f} M   "
          f"scan {report['selective_scan_flops']/1e6:>7.1f} M   "
          f"total {report['total_gflops_approx']:>8.4f} GF   "
          f"core_app {report['core_applications']}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="FLOP accounting for finished runs and bare presets.")
    p.add_argument("--run_dir", default=None)
    p.add_argument("--data_dir", default=None, help="override config.json's data_dir (e.g. if it moved)")
    p.add_argument("--write_into_test_report", action="store_true",
                   help="also replace the `flops` error block inside test_report.json")
    p.add_argument("--architectures", default=None,
                   help="comma-separated presets to count instead of a run, "
                        "e.g. split,fullchannel,efficient,recursive")
    p.add_argument("--modality", choices=["hsi", "rgb"], default="hsi")
    p.add_argument("--num_classes", type=int, default=3)
    p.add_argument("--in_channels", type=int, default=32)
    p.add_argument("--patch_hw", default="11,11")
    p.add_argument("--patch_size", type=int, default=None,
                   help="override the preset's stem patch_size, e.g. 8 for the PAD whole-image "
                        "runs, which resolve it from --target_token_grid 28 and record patch_size 8")
    p.add_argument("--wavelengths", default=None, help="path to a wavelengths.npy for preset mode")
    p.add_argument("--out", default=None, help="write the preset-mode report here")
    p.add_argument("--device", default="cpu",
                   help="cpu is correct and sufficient: medmamba_ss_trm.py:81 ships a pure-PyTorch "
                        "selective scan, so no CUDA kernel is needed to count.")
    args = p.parse_args(argv)

    if not args.run_dir and not args.architectures:
        p.error("one of --run_dir or --architectures is required")

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[flops] CUDA unavailable - counting on CPU (the count is device-independent).")
        device = "cpu"

    if args.architectures:
        height, width = (int(v) for v in args.patch_hw.split(","))
        wavelengths = np.load(args.wavelengths) if args.wavelengths else None
        out = {"modality": args.modality, "num_classes": args.num_classes,
               "in_channels": args.in_channels, "patch_hw": [height, width],
               "patch_size_override": args.patch_size, "presets": {}}
        print(f"[presets] {args.modality} {args.in_channels}ch {args.num_classes}-class "
              f"{height}x{width}")
        for architecture in [s.strip() for s in args.architectures.split(",") if s.strip()]:
            report = report_for_preset(architecture, args.modality, args.num_classes,
                                       args.in_channels, (height, width), device, wavelengths,
                                       patch_size=args.patch_size)
            out["presets"][architecture] = report
            _print_summary(report, architecture)
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            with open(args.out, "w") as f:
                json.dump(out, f, indent=2, default=str)
            print(f"[written] {args.out}")
        return 0

    run_dir = Path(args.run_dir).resolve()
    if not (run_dir / "config.json").is_file():
        print(f"ERROR: {run_dir}/config.json not found.")
        return 2

    report = report_for_run(run_dir, device, data_dir_override=args.data_dir)
    with open(run_dir / "flops_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"[run] {run_dir.name}")
    _print_summary(report, report.get("architecture") or "?")
    print(f"[written] {run_dir/'flops_report.json'}")

    if args.write_into_test_report:
        test_report_path = run_dir / "test_report.json"
        if test_report_path.is_file():
            test_report = json.load(open(test_report_path))
            test_report["flops"] = report
            with open(test_report_path, "w") as f:
                json.dump(test_report, f, indent=2, default=str)
            print(f"[written] {test_report_path} (flops block replaced)")
        else:
            print(f"[skip] {test_report_path} does not exist")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
