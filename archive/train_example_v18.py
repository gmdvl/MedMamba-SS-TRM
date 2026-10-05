# -*- coding: utf-8 -*-
"""
train_example_v18.py
======================
The v18 entry point: `train_example_v16_optimal_recon.py` plus the three things
the v18 experiment set needs and no existing entry point provides.

    1. --run_tag            so an ablation arm is identifiable from its
                            directory name rather than only its timestamp
    2. FLOPs that count     the corrected call `training/trainerg_v11.py:717`
                            never made
    3. a latent-space plot  that survives `--recon_mode latent`

Nothing frozen is edited. `training/trainerg_v11.py`, `training/plots.py`,
`training/flops_counter.py` and every `train_example_v{6,7,14,15}.py` are on
`tests/test_frozen_files_untouched.py:FROZEN_GLOBS` and stay byte-identical;
the two defects are repaired by overriding two methods in a subclass, and the
corrected logic lives in `training/flops_report_v18.py` and
`training/latent_space_v18.py` so that `scripts/flops_report_v16.py` and
`scripts/latent_space_v16.py` can retro-fit the finished runs from the same
implementation.

Composition
-----------
Exactly the pattern `train_example_v16_optimal_recon.py` uses on
`train_example_v16_optimal.py` - build the base `Overrides` and replace only
what changes, so the MRO is the design and there is no rebinding to track:

    TrainerG_v18
      -> TrainerG_v13Optimal     (optimal_recon: the composition itself)
         -> TrainerG_v12Optimal  (optimal:       _train_one_epoch, _trigger_stop)
         -> TrainerG_v13         (recon:         artifacts, export)
            -> TrainerG_v12      (everything else)

Why --run_tag exists
--------------------
`training/run_naming_v16.py:_NAME_ALLOWLIST` puts `loss`, `sampler`,
`normalization`, the two subsample fractions and `seed` into a run's directory
name. It does NOT include `trm_n_improve`, `recon_mode` or `use_wavelengths` -
which are precisely the three flags the v18 depth (E2), reconstruction (E4)
and encoding (E6) ablations vary. Without a tag those arms land in directory
names differing only by a timestamp, which is the failure the manuscript's
Appendix A opens by warning about ("directory names in the archived run tree
do not identify their contents") and which `build_run_name_optimal` and
`optimal_recon.build_run_name` already exist to prevent for their own
configurations. This is the same fix, made general.

The epoch budget, and why the banner prints it
-----------------------------------------------
`train_example_v16.py:709` computes `total_steps = args.epochs * steps_per_epoch`
and the cosine schedule decays over exactly that horizon. A 12-epoch run is
therefore a DIFFERENT learning-rate schedule, not a truncation of a 20-epoch
one, and an ablation arm may only be compared against a baseline that shares
its `--epochs`. The banner prints `epochs`/`total_steps` so a run's own log
records which block it belongs to.
"""

from __future__ import annotations

import argparse
import json
import re

import train_example_v16 as v16
import train_example_v16_optimal as optimal
import train_example_v16_optimal_recon as optimal_recon
from training import trainerg_v13
from training.trainerg_v13 import TrainerG_v13  # noqa: F401  (documents the MRO above)

_RUN_TAG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9-]*$")


# ============================================================================
# trainer
# ============================================================================

class TrainerG_v18(optimal_recon.TrainerG_v13Optimal):
    """`TrainerG_v13Optimal` with the two silent-artifact defects repaired.

    Both overrides are strictly additive: each calls `super()` first and only
    acts when the inherited path produced nothing. A run that already got a
    FLOP count or a latent-space plot is untouched, so this class cannot change
    any number an existing entry point would have produced.
    """

    # -- F2: FLOPs -----------------------------------------------------
    def evaluate_test_split(self, test_loader, *a, **kw):
        report = super().evaluate_test_split(test_loader, *a, **kw)
        if report is None:
            return report
        flops = report.get("flops")
        if not (isinstance(flops, dict) and flops.get("error")):
            return report          # already counted - leave it alone

        try:
            report["flops"] = self._recount_flops(test_loader)
            with open(self.exp_dir / "test_report.json", "w") as f:
                json.dump(report, f, indent=2, default=str)
            total = report["flops"].get("total_gflops_approx")
            self.logger.info(
                f"[flops-v18] recounted at {report['flops']['input_shape']}: "
                f"{total} GFLOPs, core_applications="
                f"{report['flops'].get('core_applications')} "
                f"(the inherited call passed a 3-D shape; see training/flops_report_v18.py)")
        except Exception as e:
            self.logger.warning(f"[flops-v18] recount failed: {type(e).__name__}: {e}")
        return report

    def _recount_flops(self, test_loader) -> dict:
        from training.flops_report_v18 import build_flops_report
        batch = next(iter(test_loader))
        sample_x = batch[0]
        _, in_channels, height, width = sample_x.shape
        cli_args = (self.config or {}).get("cli_args", {}) or {}
        return build_flops_report(
            self.model, in_channels=in_channels, patch_hw=(height, width),
            cli_args=cli_args, device=str(self.device),
            wavelengths=self.wavelengths, sensor_range=self.sensor_range,
            backbone_num_params=(self.config or {}).get("backbone_num_params"))

    # -- F3: latent space ----------------------------------------------
    def _generate_rich_diagnostics(self):
        super()._generate_rich_diagnostics()

        out_dir = self.dirs["latent_space"]
        try:
            if any(out_dir.iterdir()):
                return             # the inherited path worked - nothing to do
        except OSError:
            return

        try:
            from training.latent_space_v18 import write_latent_space
            written = write_latent_space(
                self.model, self.val_loader, str(self.device), str(out_dir),
                class_names=self.class_names, wavelengths=self.wavelengths,
                sensor_range=self.sensor_range)
            if written:
                self.logger.info(f"[latent-v18] wrote {sorted(written)} to {out_dir} "
                                  f"(the inherited extractor cannot unpack this loader's batches)")
            else:
                self.logger.warning("[latent-v18] no extraction strategy matched this model - "
                                    "latent_space/ intentionally left empty")
        except Exception as e:
            self.logger.warning(f"[latent-v18] failed: {type(e).__name__}: {e}")


# ============================================================================
# CLI
# ============================================================================

def _valid_run_tag(value: str) -> str:
    if not _RUN_TAG_RE.match(value or ""):
        raise argparse.ArgumentTypeError(
            f"--run_tag must be alphanumeric with dashes and start with a letter or digit, "
            f"got {value!r}. It becomes part of a directory name, so '_' is reserved as the "
            f"field separator run_naming_v16 uses.")
    return value


def build_arg_parser(argv=None):
    """`train_example_v16_optimal_recon.build_base_arg_parser` + `--run_tag`.

    The BASE parser, not `optimal_recon.build_arg_parser`: that one adds
    `RUN_DEFAULTS` (2026-09-28) and `--early_stopping_enabled`, and every
    documented v18/v20 command - and so every run behind the paper - relies on
    the defaults as they were before that layer existed."""
    p = optimal_recon.build_base_arg_parser(argv)
    p.description = ("GMedMamba v18 - the optimal + reconstruction configuration, with FLOPs and "
                     "latent-space artifacts that are actually produced, and a run tag so an "
                     "ablation arm is identifiable from its directory name")
    g = p.add_argument_group("v18")
    g.add_argument("--run_tag", type=_valid_run_tag, default=None,
                    help="appended to the run directory name. REQUIRED in practice for any "
                         "ablation that varies a flag run_naming_v16._NAME_ALLOWLIST does not "
                         "cover - --trm_n_improve, --recon_mode and --no_use_wavelengths are the "
                         "three the v18 set uses - or the arms differ only by timestamp.")
    return p


def build_run_name(args, *a, **kw) -> str:
    """`optimal_recon`'s name, plus `_<run_tag>` when one was given."""
    name = optimal_recon.build_run_name(args, *a, **kw)
    tag = getattr(args, "run_tag", None)
    return f"{name}_{tag}" if tag else name


def print_banner(args) -> None:
    optimal_recon.print_base_banner(args)
    tag = getattr(args, "run_tag", None)
    print(f"[v18] run_tag={tag!r}"
          + ("" if tag else "  <-- none given: this run is distinguishable from another with the "
                            "same flags only by its timestamp"), flush=True)
    print(f"[v18] epochs={args.epochs} -> the cosine schedule decays over epochs x steps_per_epoch "
          f"(train_example_v16.py:709). Only compare this run against one with the SAME --epochs.",
          flush=True)


def build_overrides(args) -> "v16.Overrides":
    base = optimal_recon.build_overrides(args)
    return v16.Overrides(parser_fn=build_arg_parser,
                         run_name_fn=build_run_name,
                         model_fn=base.model_fn,          # build_model_optimal, unchanged
                         trainer_cls=TrainerG_v18)


def main() -> None:
    args = build_arg_parser().parse_args()
    print_banner(args)
    trainerg_v13.ENTRY_POINT_OVERRIDE = "train_example_v18.py"
    v16.main(build_overrides(args))


if __name__ == "__main__":
    main()
