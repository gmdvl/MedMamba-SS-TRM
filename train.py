# -*- coding: utf-8 -*-
"""
train.py
=========
One MedMamba-SS-TRM training run.

    python train.py --data_dir data/hsi_v8-80_10_10_importance-new/hsi   # default profile: paper_recipe
    python train.py --profile paper_recipe --data_dir data/hsi_v8-80_10_10_importance-new/hsi \\
        --batch_size 256 --epochs 12 --lambda_sam 0.1 \\
        --train_subsample_frac 0.102 --val_subsample_frac 0.0986 --run_tag abl-base

`--profile` picks the defaults (training/train_profiles.py); any flag overrides
them. `python train.py --help` lists every flag. Many runs: run_experiments.py.

Each profile is named for what it is for. A name ending in `_norecon` trains
with `recon_mode none`, so no reconstruction head and no MSE/SAM term:

    profile                      good for                                       reconstruction
    paper_recipe  (the default)  reproducing or extending the paper's numbers   latent, lambda_mse 0.1
    pad_ufes_best_norecon        best metrics on PAD-UFES-20 (RGB); --stage     NONE
    medmamba_protocol_norecon    comparing against MedMamba's own recipe        NONE
    pipeline_defaults            the bare pipeline, no tuning layered on top    latent, lambda_mse 1.0

They replace train_example_v16.py, _original, _optimal, _recon and
train_example_v18.py (now in archive/):

    train_example_v18.py                 --profile paper_recipe               (was v18)
    train_example_v16_optimal.py         --profile pad_ufes_best_norecon      (was optimal)
    train_example_v16_original.py        --profile medmamba_protocol_norecon  (was original)
    train_example_v16.py                 --profile pipeline_defaults          (was base)
    train_example_v16_recon.py           --profile pipeline_defaults  (artifact flags always exist)

The old names still work; the run prints the new one.
"""

import sys

import torch

from training.train_banner import print_banner
from training.train_cli import resolve
from training.train_pipeline import run


def main(argv=None) -> None:
    parsed, notes = resolve(argv)
    print_banner(parsed)
    try:
        run(parsed, notes)
    except torch.OutOfMemoryError:
        a = parsed.args
        print("\n" + "=" * 78, flush=True)
        print("CUDA OUT OF MEMORY - the recursive core's working set is (tokens x batch) and it "
              "is held through every core application.", flush=True)
        print(f"  --patch_size {(a.patch_size or 1) * 2}   halves the token grid, quartering "
              f"activations (costs spatial resolution)", flush=True)
        print(f"  --batch_size {max(1, a.batch_size // 2)}   halves them too (changes the "
              f"optimisation)", flush=True)
        print("=" * 78, flush=True)
        raise


if __name__ == "__main__":
    main(sys.argv[1:])
