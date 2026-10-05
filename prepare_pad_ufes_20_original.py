# -*- coding: utf-8 -*-
"""
prepare_pad_ufes_20_original.py
================================
PAD-UFES-20 prepared under the ORIGINAL MedMamba INPUT protocol, so that
`medmamba_ss_trm.py`'s recursive (TRM) variant, trained by
`train_example_v16_original.py`, differs from a published-recipe MedMamba
run in as few places as possible.

Why a new file rather than a flag on v6
---------------------------------------
`prepare_pad_ufes_20_v6.py` is on the v16 plan's FROZEN list
(`test_frozen_files_untouched.py:FROZEN_GLOBS`) because `data/pad_v6/` and
`data/pad_v6-undersample/` - the inputs to the runs the manuscript reports -
were produced by it. It is imported here and never edited; this file is a
`training.prep.DatasetAdapter` SUBCLASS, exactly the relationship
`train_example_v16.py` has to `train_example_v15.py`.

What "the original setup" means, concretely
-------------------------------------------
Reference implementation: `../medmamba-original/MedMamba/train.py` and
`../medmamba-original/MedMamba/prepare_medmamba_dataset.py`, whose defaults
follow MedMamba (arXiv:2403.03849) for the non-MedMNIST datasets.

  * ONE sample per clinical image, resized to 224 x 224 x 3 - NOT the
    11 x 11 patch grid `pad_v6` uses. This is the single largest difference
    between the two protocols and the one the manuscript's Section VI-C
    identifies as the likely cause of both models barely clearing a
    six-feature colour baseline: an 11 x 11 crop of a skin photograph
    mostly does not contain the lesion the label refers to.
  * 60 / 10 / 30 train / val / test (the reference prep's
    DEFAULT_TRAIN_RATIO / VAL / TEST), against v6's `80/10/10`.
  * bilinear resize of the whole image, EXIF-corrected, RGB, stored as
    float32 in [0, 1]. The reference's
    `Normalize((0.5,)*3, (0.5,)*3)` - i.e. `(x - 0.5) / 0.5` - is applied at
    TRAINING time by `train_example_v16_original.py`, not baked in here, so
    the stored array stays inspectable and the integrity gates' [0, 1]
    range check still means something.
  * `--split_level` chooses between the reference prep's two modes:
    `patient` (its default; no patient appears in two splits) and `image`
    (what reproduces the per-split image counts published in the MedMamba
    paper, and what leaks patients across splits). `patient` is the default
    here too - the manuscript's numbers are patient-split throughout and a
    baseline comparison that silently changed that would not be comparable
    to anything else in the paper.

Everything else - parallel decode, spanning shards, crash-safe resume,
manifest/integrity artefacts, `class_names.json` - is v6's, unchanged,
because it is the prep half of the pipeline the training gates assume.

Usage
-----
    python prepare_pad_ufes_20_original.py \
        --root /path/to/PAD-UFES-20 \
        --out_dir ./data/pad_original \
        --num_workers 8

    # the MedMamba paper's split granularity (leaks patients; use only to
    # compare against the paper's published table, not against our own runs)
    python prepare_pad_ufes_20_original.py --root ... --out_dir ./data/pad_original_imgsplit \
        --split_level image
"""

from __future__ import annotations

import dataclasses
from typing import Iterator, List

from prepare_pad_ufes_20 import DIAGNOSTIC_TO_ID, PadUfes20Adapter
from training.prep import PrepItem, run_prep

# The reference prep's DEFAULT_TRAIN_RATIO / DEFAULT_VAL_RATIO /
# DEFAULT_TEST_RATIO (prepare_medmamba_dataset.py:137-139).
MEDMAMBA_SPLIT = "60/10/30"

# `train.py --image_size`, and the "224 x 224 x 3 input" line of the paper's
# protocol. Also v6's own `--img_size` default, so this is a re-statement
# rather than a change - it is spelled out because dropping `--patch_size`
# is what makes it take effect.
MEDMAMBA_IMAGE_SIZE = 224


class PadUfes20OriginalAdapter(PadUfes20Adapter):
    """v6's adapter with the reference prep's defaults and its
    `--split_level` switch. `discover`, `load_item`, `modality_spec` and
    `expected_sample_count` are all v6's: with `--patch_size` left unset,
    `load_item` already yields the whole resized image as a single sample
    (`prepare_pad_ufes_20_v6.py:156-158`) and `expected_sample_count`
    already returns 1, so the patch-free path needs no new code - only new
    defaults and a grouping choice."""

    name = "pad_ufes_20_original"

    def add_cli_args(self, p) -> None:
        super().add_cli_args(p)

        # `--split` belongs to the shared core parser (training/prep/core.py:58),
        # which runs BEFORE `add_cli_args`, so retarget its default here rather
        # than re-declaring the argument (which would need
        # `conflict_handler='resolve'` and lose its help text) - the same
        # technique `train_example_v16.build_arg_parser` uses to widen
        # `--normalization`.
        for action in p._actions:
            if action.dest == "split":
                action.default = MEDMAMBA_SPLIT
                action.help = (f"{action.help}  [original-protocol default: {MEDMAMBA_SPLIT}, "
                               f"the reference prep's 60/10/30]")
            elif action.dest == "img_size":
                action.default = MEDMAMBA_IMAGE_SIZE
            elif action.dest == "patch_size":
                action.help = ("square patch side. LEAVE UNSET for the original MedMamba "
                               "protocol: one 224x224 sample per clinical image. Setting it "
                               "reverts to the pad_v6 patch-grid protocol and this script "
                               "then differs from prepare_pad_ufes_20_v6.py only in --split.")

        p.add_argument("--split_level", choices=["patient", "image"], default="patient",
                        help="patient = no patient appears in more than one split (leak-free, "
                             "the reference prep's default and what every MedMamba-SS-TRM run in the "
                             "manuscript uses). image = split individual images, which matches "
                             "the granularity of the counts published in the MedMamba paper and "
                             "leaks patients across splits.")

    def compat_keys(self, args) -> dict:
        # Included so `--split_level` is part of the resume-compatibility
        # lock (training/prep/core.py:_locked_args): resuming an interrupted
        # patient-split prep into an image-split one would silently produce a
        # dataset whose split_assignment.json contradicts its shards.
        keys = dict(super().compat_keys(args))
        keys["split_level"] = getattr(args, "split_level", "patient")
        return keys

    def discover(self, args) -> Iterator[PrepItem]:
        """v6's discovery, with the split grouping key swapped for the image
        key under `--split_level image`. The core splits on `PrepItem.group_id`
        and nothing else, so one group per image IS an image-level split."""
        level = getattr(args, "split_level", "patient")
        for item in super().discover(args):
            yield item if level == "patient" else dataclasses.replace(item, group_id=item.key)

    def provenance(self, args, state) -> dict:
        prov = dict(super().provenance(args, state) or {})
        prov.update({
            "protocol": "medmamba_original",
            "protocol_reference": "arXiv:2403.03849 (non-MedMNIST) via "
                                   "medmamba-original/MedMamba/prepare_medmamba_dataset.py",
            "split_level": getattr(args, "split_level", "patient"),
            "sample_unit": "patch" if args.patch_size else "whole_image",
            "image_size": args.img_size,
        })
        return prov


def _print_banner(adapter: PadUfes20OriginalAdapter) -> None:
    print("=" * 78, flush=True)
    print("PAD-UFES-20  -  ORIGINAL MedMamba input protocol", flush=True)
    print("  sample unit   : one whole clinical image (no patch grid), unless --patch_size", flush=True)
    print(f"  image size    : {MEDMAMBA_IMAGE_SIZE}x{MEDMAMBA_IMAGE_SIZE}x3, bilinear, EXIF-corrected", flush=True)
    print(f"  split         : {MEDMAMBA_SPLIT} (train/val/test)", flush=True)
    print("  stored range  : float32 in [0, 1]; (x-0.5)/0.5 is applied at training time", flush=True)
    print(f"  classes       : {', '.join(adapter.class_names)}", flush=True)
    print("=" * 78, flush=True)


def main() -> int:
    adapter = PadUfes20OriginalAdapter()
    _print_banner(adapter)
    return run_prep(adapter)


if __name__ == "__main__":
    raise SystemExit(main())
