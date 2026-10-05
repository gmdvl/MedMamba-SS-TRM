# -*- coding: utf-8 -*-
"""
train_example_v16_recon.py
============================
`train_example_v16.py` with reconstruction ARTIFACTS turned on - the
reconstructed cubes written to disk, the paper figures rendered, and the
per-epoch reconstruction metrics plotted.

Everything else is v16, unchanged and unedited. `_main()` is imported and
run as-is; only two module globals it resolves at call time are rebound:

    train_example_v16.build_arg_parser  ->  adds the artifact flags
    train_example_v16.TrainerG_v12      ->  TrainerG_v13

That is the whole diff. `train_example_v16.py` stays byte-identical, so the
v16 runs already in `experiments/` stay reproducible and
`test_frozen_files_untouched.py` is unaffected.

Usage - identical to `train_example_v16.py`, plus:

    python train_example_v16_recon.py \
        --data_dir data/hsi_v8-80_10_10_importance/hsi/ \
        --architecture recursive --recon_mode latent \
        --normalization global_zscore --epochs 30 \
        --recon_sample_per_class 3

`--recon_mode latent` is what makes any of this do anything: with
`--recon_mode none` (v16's usual setting) no decoder exists, and
`reconstruction/status.json` records exactly that instead of leaving an
empty directory behind.

Outputs added under `<run>/reconstruction/`:

    samples.npz                 x_true / x_recon cubes [N,C,H,W], reflectance
                                units, plus labels, split indices, tags,
                                wavelengths, class names
    samples_metrics.json        per-sample SAM / RMSE / PSNR / SID /
                                spectral SSIM / 2-D SSIM / peak-position error
    provenance.json             checkpoint, selection rule, band choice,
                                split statistics
    status.json                 written on every path, including skips
    reconstruction_samples.png  the contact sheet (paper figure)
    samples/NN_<tag>.png        one four-panel figure per sample
    metrics_curves.png          per-epoch SAM / RMSE / PSNR / spectral SSIM
"""

from __future__ import annotations

import train_example_v16 as v16
from training import trainerg_v13
from training.trainerg_v13 import TrainerG_v13

_v16_build_arg_parser = v16.build_arg_parser


def build_arg_parser():
    p = _v16_build_arg_parser()
    p.description = ("GMedMamba v16 training entry point + reconstruction artifacts "
                     "(cubes on disk, paper figures, metric curves)")
    g = p.add_argument_group("v16-recon-artifacts")
    g.add_argument("--recon_save_samples", choices=["on", "off"], default="on",
                    help="write the reconstructed cubes and figures at the end of training. "
                         "Ignored when --recon_mode none (there is no decoder to read).")
    g.add_argument("--recon_sample_per_class", type=int, default=3,
                    help="samples per class, taken at evenly spaced SAM quantiles - q=0 is that "
                         "class's best reconstruction, q=1 its worst, so the worst case is in the "
                         "figure by construction. 3 = best/median/worst; 5 adds the quartiles.")
    g.add_argument("--recon_rgb_bands", default=None,
                    help="three comma-separated values for the RGB composite: wavelengths in nm "
                         "('640,550,460', matched to nearest band) or band indices ('20,12,4'). "
                         "Default: nearest bands to 640/550/460 nm when wavelengths.npy is "
                         "present, otherwise evenly spaced indices. The choice is recorded in "
                         "provenance.json.")
    g.add_argument("--recon_render_png", choices=["on", "off"], default="on",
                    help="off writes samples.npz and the metrics only - useful on a headless "
                         "box where you would rather render the figures locally.")
    g.add_argument("--recon_max_cubes", type=int, default=64,
                    help="hard cap on retained cubes, a guard against a large --recon_sample_"
                         "per_class on a many-class dataset producing a multi-GB npz.")
    return p


def build_overrides() -> "v16.Overrides":
    """v17 S9 - what this entry point contributes, as a value instead of two assignments
    into `v16`'s namespace."""
    return v16.Overrides(parser_fn=build_arg_parser, trainer_cls=TrainerG_v13)


def main():
    trainerg_v13.ENTRY_POINT_OVERRIDE = "train_example_v16_recon.py"
    v16.main(build_overrides())


if __name__ == "__main__":
    main()
