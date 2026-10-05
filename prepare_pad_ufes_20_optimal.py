# -*- coding: utf-8 -*-
"""
prepare_pad_ufes_20_optimal.py
===============================
PAD-UFES-20 prepared for the BEST ACHIEVABLE MedMamba-SS-TRM (TRM) metrics,
rather than for reproducing a published protocol.

Companion to `train_example_v16_optimal.py`. Like
`prepare_pad_ufes_20_original.py`, this is a `training.prep.DatasetAdapter`
SUBCLASS of `prepare_pad_ufes_20_v6.PadUfes20Adapter`, which is on the v16
plan's FROZEN list (`test_frozen_files_untouched.py:FROZEN_GLOBS`) because
`data/pad_v6*` fed the runs the manuscript reports. v6 is imported and never
edited.

Why this file exists at all
---------------------------
Almost every extraction decision in v6 is already right; what was wrong was
the COMMAND LINE the manuscript's datasets were built with. Two defaults are
changed here, and one option is added. That is the honest size of the delta -
the larger half of the remediation lives in `train_example_v16_optimal.py`.

  1. `--split` default `70/15/15`, not the prep core's `80_20`.

     `80_20` is a TWO-WAY spec: `parse_split_spec` maps it to
     `(0.8, 0.0, 0.2)`, so no `X_val.npy` is written. `train_example_v16`'s
     `discover_data` then falls back to `X_test.npy` AS the validation split,
     and `_build_test_loader_v16` detects that and refuses to report a test
     number at all (`train_example_v16.py:300-305`) - gate G6 is unreachable
     and the only number the run can report is the one it selected on.

     Among three-way choices, `70/15/15` is a compromise, and the numbers
     it compromises between are worth seeing. PAD-UFES-20 is 2,298 images
     from 1,373 patients: BCC 845 (36.8%), ACK 730 (31.8%), NEV 244
     (10.6%), SEK 235 (10.2%), SCC 192 (8.4%), MEL **52 (2.3%)**. Measured
     with `--dry_run`, at `--split_strategy stratified` and patient
     grouping:

         split       train   val   test  | MEL train  val  test
         60/10/30    1,384   245    669  |     33      5     14
         70/15/15    1,626   328    344  |     39      6      7
         80/10/10    1,852   225    221  |     42      5      5

     MEL is the whole problem. 80/10/10 buys +14% training data and leaves
     FIVE melanomas in the validation split that every epoch's checkpoint
     decision is made on, and five in the split the thesis reports - one
     image flipping moves MEL F1 by ~0.1 and macro-F1 by ~0.017, so a large
     part of the reported number would be sampling noise. 60/10/30 has the
     most trustworthy test split (669 images, 14 MEL) and the least data to
     learn from. 70/15/15 keeps 92% of 80/10/10's melanoma training images
     while giving val and test half again as many samples.

     Whatever is chosen, macro-F1 on this dataset has an irreducibly noisy
     MEL cell; report balanced accuracy and per-class support alongside it.
     Pass `--split 80/10/10` to favour training data, or `--split 60/10/30`
     to match `prepare_pad_ufes_20_original.py` and get the most defensible
     test estimate.

  2. `--tiling` (new), default `whole` - i.e. one 224x224 sample per
     clinical image, which is v6's `--patch_size` default and NOT what
     `data/pad_v6` was built with.

     This is the single largest measured effect on this dataset. Both preps
     have a `scripts/shallow_baseline_v16.py` report on disk, computed by the
     same code on the same classes, and they disagree by a factor of two:

         input               probe  accuracy  balanced  macro-F1
         11x11 patches       flat     0.2128    0.2316    0.1846
         (data/pad_v6)       mean     0.2283    0.2373    0.1867
         whole 224x224       flat     0.4735    0.3365    0.3252
         (data/pad_original) mean     0.3020    0.2955    0.2327

     On 11x11 crops a linear model over all 363 pixels does WORSE than one
     over the 3 mean channel values: the crop carries colour and nothing
     else, because an 11x11 window is 0.24% of the frame and mostly misses
     the lesion the label refers to. Whole images nearly double every
     figure. No MedMamba-SS-TRM run on the 11x11 grid has ever exceeded
     balanced accuracy 0.271 / macro-F1 0.235
     (`experiments/20260905_133922_.../history.json`), which is close to the
     ceiling that probe implies rather than an architecture result.

     `--tiling mil9` / `mil4` are the middle ground, for when whole-image
     runs are data-starved: 112x112 tiles at stride 56 (9 per image, 14,634
     train samples at 70/15/15) or 112 at stride 112 (4 per image). Each tile is 25% of
     the frame instead of 0.24%, so it usually still contains lesion, and
     `train_example_v16 --eval_group_aggregation mean_prob` reassembles
     IMAGE-level test metrics from the tile posteriors. Those modes write the
     `images_{split}.npy` sidecar that aggregation needs (see below); `whole`
     does not need one, since one sample already is one image.

What is deliberately NOT changed from v6
----------------------------------------
  * `--img_size 224`, bilinear resize of the full frame, EXIF-corrected, RGB,
    float32 in [0, 1]. Normalization is applied at TRAINING time, so the
    stored array stays inspectable and the integrity gates' [0, 1] range
    check keeps meaning something.
  * The square resize distorts aspect ratio. A centre-crop-to-square would
    preserve geometry but can cut off an off-centre lesion; PAD's frames are
    already tight crops, so the distortion is mild and uniform across splits.
    Left as v6 has it, and named here so it is a known quantity.
  * `--balance_classes none`. The prep core can undersample the train split,
    but PAD's rarest class is MEL at 39 train images under 70/15/15;
    undersampling every other class down to it would keep 234 of 1,626
    images and discard 86% of the data. Imbalance is corrected in the LOSS
    instead - see
    `train_example_v16_optimal.OPTIMAL_DEFAULTS["class_weight_power"]`.
  * Patient-level grouping. `PrepItem.group_id` is the `patient_id` column,
    so `training.prep.splits` never places one patient in two splits, at any
    `--split`. This is not optional: the MedMamba paper's published PAD
    numbers use an image-level split, which shares patients across splits
    and inflates results.

Usage
-----
    python prepare_pad_ufes_20_optimal.py \\
        --root ../Dataset/PAD-UFES-20 --out_dir ./data/pad_optimal \\
        --num_workers 8

    # then, optionally, the linear-probe bar the network has to clear
    python scripts/shallow_baseline_v16.py --data_dir ./data/pad_optimal

    # patch-MIL variant (writes images_*.npy for --eval_group_aggregation)
    python prepare_pad_ufes_20_optimal.py --root ... --out_dir ./data/pad_optimal_mil9 \\
        --tiling mil9 --num_workers 8

Every `prepare_pad_ufes_20_v6.py` / `training.prep.core` flag still works and
overrides the defaults above.
"""

from __future__ import annotations

import os
import sys

from prepare_pad_ufes_20 import PadUfes20Adapter
from training.prep import run_prep
from training.prep_progress import install as install_prep_progress

# `dest -> (patch_size, stride)`. `None` patch_size is v6's "one sample per
# image" path in `load_item`; the tiled entries are square tiles of a 224
# frame, chosen so a tile is a quarter of the frame rather than a 500th.
TILINGS = {
    "whole": (None, None),      # 1 sample  / image, 224x224 (100% of frame)
    "mil4":  (112, 112),        # 4 samples / image, 112x112, no overlap
    "mil9":  (112, 56),         # 9 samples / image, 112x112, 50% overlap
}

# The prep core's `--split` default is a TWO-way spec and writes no X_val.npy;
# see the module docstring for why 70/15/15 and not 80/10/10.
DEFAULT_SPLIT = "70/15/15"


class PadUfes20OptimalAdapter(PadUfes20Adapter):
    """v6's adapter with a three-way default split and a `--tiling` preset.

    `name` keeps the `pad_ufes_20` prefix on purpose:
    `scripts/derive_patch_groups.py` dispatches on
    `manifest["adapter"].startswith("pad_ufes_20")`
    (`scripts/derive_patch_groups.py:232`), so the sidecar deriver works on
    this adapter's output without being taught about it - the same reason
    `prepare_pad_ufes_20_original.py` is called `pad_ufes_20_original`.
    """

    name = "pad_ufes_20_optimal"

    def add_cli_args(self, p) -> None:
        super().add_cli_args(p)
        p.add_argument("--tiling", choices=sorted(TILINGS), default="whole",
                        help="'whole' = one 224x224 sample per clinical image (the setting the "
                             "shallow-probe evidence in this file's docstring supports). "
                             "'mil9'/'mil4' cut 112x112 tiles instead, for use with "
                             "train_example_v16 --eval_group_aggregation mean_prob. Explicit "
                             "--patch_size/--stride override this.")
        p.add_argument("--no_group_sidecars", action="store_true",
                        help="skip the scripts/derive_patch_groups.py pass that a tiled run needs "
                             "for image-level evaluation. Inert for --tiling whole.")
        # Applied after v6's own add_cli_args so this wins; `--split` itself
        # belongs to the prep core's parser, which is built before either.
        p.set_defaults(split=DEFAULT_SPLIT)

    def preprocess_args(self, args) -> None:
        super().preprocess_args(args)
        # v17 S6 - keep the parsed namespace so `main` does not have to re-scan sys.argv.
        # `prepare_histologyhsi_bc_v7.py:93` already does exactly this; the PAD adapter
        # never did, which is why `main` grew a hand-rolled argv parser.
        self._args = args

        patch_size, stride = TILINGS[args.tiling]
        explicit = {a.split("=", 1)[0] for a in sys.argv[1:]}
        overridden = "--patch_size" in explicit or "--stride" in explicit
        if overridden:
            print(f"  [tiling] --patch_size/--stride given explicitly; --tiling {args.tiling} "
                  f"ignored (patch_size={args.patch_size}, stride={args.stride})")
        else:
            args.patch_size, args.stride = patch_size, stride

        if args.patch_size and args.patch_size <= 32:
            print(f"  [tiling] WARNING: --patch_size {args.patch_size} on a {args.img_size}px "
                  f"frame is {100.0 * args.patch_size ** 2 / max(1, args.img_size ** 2):.2f}% of "
                  f"the image per sample. At 11x11 a linear probe over all 363 pixels scored "
                  f"macro-F1 0.185 against 0.325 on whole images (data/pad_v6 vs "
                  f"data/pad_original shallow_baseline.json) - the label mostly does not apply "
                  f"to the crop. This is the setting this file exists to move away from.")

        self._describe(args, label=("explicit" if overridden else args.tiling))

    # ---- reporting -----------------------------------------------------
    def _describe(self, args, label: str = "whole") -> None:
        """Print what the resolved flags imply, before an hour of decoding.

        `label` is the tiling preset's name, or "explicit" when
        --patch_size/--stride overrode it - naming the ignored preset here
        would contradict the line above it."""
        per_image = 1
        if args.patch_size:
            step = args.stride or args.patch_size
            n = (args.img_size - args.patch_size) // step + 1
            per_image = max(0, n) ** 2
        side = args.patch_size or args.img_size
        print(f"  [tiling] {label}: {per_image} sample(s)/image of {side}x{side} "
              f"({100.0 * side ** 2 / max(1, args.img_size ** 2):.1f}% of the {args.img_size}px "
              f"frame); split={args.split}, patient-grouped")
        if per_image > 1:
            print(f"  [tiling] pass --eval_group_aggregation mean_prob to train_example_v16*.py "
                  f"to get IMAGE-level test metrics back out of the {per_image} tile posteriors.")

    def provenance(self, args, state) -> dict:
        out = dict(super().provenance(args, state) or {})
        out.update({"tiling": args.tiling, "prep_variant": "optimal",
                    "samples_per_image": self._samples_per_image(args)})
        return out

    @staticmethod
    def _samples_per_image(args) -> int:
        if not args.patch_size:
            return 1
        step = args.stride or args.patch_size
        return max(0, (args.img_size - args.patch_size) // step + 1) ** 2


def _write_group_sidecars(out_dir: str) -> None:
    """Run `scripts/derive_patch_groups.py` over the finished dataset.

    Tiled output needs `images_{split}.npy` before
    `--eval_group_aggregation` can do anything; the deriver reconstructs it
    from `_progress.json` row order alone and self-checks
    `sum(counts) == len(y_split)` before writing, so a misaligned sidecar is
    never produced. Best-effort: a failure here costs image-level reporting,
    not the dataset.
    """
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))
    try:
        import derive_patch_groups
        derive_patch_groups.main(["--out_dir", out_dir])
    except SystemExit as e:
        if e.code:
            print(f"  [sidecars] derive_patch_groups.py exited {e.code}; image-level evaluation "
                  f"(--eval_group_aggregation) will be unavailable for {out_dir}")
    except Exception as e:
        print(f"  [sidecars] WARNING: could not derive image sidecars: {type(e).__name__}: {e}. "
              f"Image-level evaluation (--eval_group_aggregation) will be unavailable; run "
              f"scripts/derive_patch_groups.py --out_dir {out_dir} by hand to retry.")


def main() -> int:
    # v17 S6 - rate + ETA on the extraction loop. training/prep/core.py is FROZEN
    # and discards the rate it already computes (core.py:400); this rebinds
    # run_extraction instead of editing it.
    install_prep_progress()
    adapter = PadUfes20OptimalAdapter()
    rc = run_prep(adapter)
    if rc:
        return rc

    # v17 S6 - read the parsed namespace `preprocess_args` kept, instead of re-scanning
    # sys.argv by hand. The old scan missed `--` , argparse prefix abbreviations
    # (`--til=mil9`) and any value that happened to look like a flag, and it silently fell
    # back to `./data` when it did - writing the sidecars next to the wrong dataset.
    a = getattr(adapter, "_args", None)
    if a is None:                    # run_prep returned before preprocess_args (e.g. --help)
        return 0
    tiling, out_dir = a.tiling, a.out_dir
    skip = bool(getattr(a, "no_group_sidecars", False))
    partial = bool(getattr(a, "dry_run", False) or getattr(a, "limit", None))
    if tiling != "whole" and not skip and not partial:
        print(f"\n  [sidecars] --tiling {tiling} produces several samples per clinical image; "
              f"deriving images_*.npy so image-level evaluation is possible.")
        _write_group_sidecars(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
