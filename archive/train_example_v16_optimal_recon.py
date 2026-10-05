# -*- coding: utf-8 -*-
"""
train_example_v16_optimal_recon.py
====================================
The BEST-METRICS configuration (`train_example_v16_optimal.py`) AND the
reconstruction artifacts (`train_example_v16_recon.py`), in one run.

Why this file has to exist
--------------------------
Both of those entry points contribute the SAME thing - the trainer class - so they are
mutually exclusive, and there was no way to get the tuned objective, normalization,
schedule and stopping rules together with a reconstructed cube on disk:

    train_example_v16_optimal:  trainer_cls = TrainerG_v12Optimal
    train_example_v16_recon:    trainer_cls = TrainerG_v13

(Before v17 S9 those were assignments to the module global `v16.TrainerG_v12`, and
"whichever runs last wins" was literal - this file had to re-assign three names AFTER
calling `optimal.install_overrides`, because that call had just set them to the
optimal-only versions. They are now values passed to `v16.main`, so the composition below
is an expression rather than an ordering. `install_overrides` survives as a deprecated
shim.) `--recon_mode latent` on
`train_example_v16_optimal.py` trains a decoder and then throws its output
away (`TrainerG_v12` keeps only `[N, C]` mean spectra); `--stage`, the focal
weighting and the EMA horizon do not exist on `train_example_v16_recon.py`.

The composition is still the whole file, and it is a CLASS composition - that part v17
did not touch, because it is the good half. `TrainerG_v12Optimal` and its base
`TrainerG_v12Original` override methods only and declare no `__init__`, while
`TrainerG_v13` adds one (it reads the `--recon_*` flags) - so a plain MRO
gives each side exactly what it contributes:

    TrainerG_v13Optimal
      -> TrainerG_v12Optimal    _train_one_epoch (3-tuple batches), _trigger_stop
      -> TrainerG_v12Original   _check_stopping_rules, gate-G5 metric alias
      -> TrainerG_v13           __init__, export_reconstruction_samples,
                                _generate_rich_diagnostics (writes the artifacts)
      -> TrainerG_v12           everything else

`TrainerG_v12Optimal._train_one_epoch` matters MORE here than in its own file:
it is the shim that lets a non-recursive architecture consume the 3-tuple
batches `--recon_mode latent` switches on. Without it `--architecture split
--recon_mode latent` dies unpacking the first batch.

What changes relative to train_example_v16_optimal.py
-----------------------------------------------------
Two layers of defaults on top of `OPTIMAL_DEFAULTS`, then `--stage`, then
whatever the command line says - later wins:

    OPTIMAL_DEFAULTS -> RECON_DEFAULTS -> RUN_DEFAULTS -> --stage -> explicit flags

`RECON_DEFAULTS` (recon_mode latent, lambda_mse 0.1) is the original layer:
a reconstruction run that reconstructs nothing is not worth the wall clock.

`RUN_DEFAULTS` (2026-09-28) makes the configuration below the one a bare
command runs. It is the v15/v16 recommended HSI command
(`documentations/10_commands_v15.md`, "HSI - recommended command") with
`global_zscore`, `--lambda_mse 0.1` and five samples per class, plus the three
guards that 64x64-patch runs need so gate G8 is not OOM-killed and the 64x64
latents do not become 4,096 tokens (`--drift_check_patches 500`,
`--norm_stats_sample_cap 1000`, `--patch_size 4`):

    --drift_check_patches 500 --norm_stats_sample_cap 1000 --patch_size 4
    --architecture recursive --normalization global_zscore
    --recon_mode latent --recon_out_activation auto
    --lambda_mse 0.1 --lambda_sam 0.1 --recon_sample_per_class 5
    --loss weighted_ce --class_weight_power 1.0 --sampler moderate_oversample
    --trm_drop_path 0.1 --trm_dropout 0.1 --classifier_dropout 0.2
    --augment_preset light --epochs 30 --lr 1e-4 --amp bf16
    --train_subsample_frac 0.1 --train_subsample_mode per_epoch
    --val_subsample_frac 0.2 --early_stop_patience 8 --min_patient_recall 0.5
    --compile on --loader_mode performance --dataset_storage mmap
    --eval_artifact_stride 5 --keep_last_n 2 --seed 42

It replaces three things `OPTIMAL_DEFAULTS` resolves from the dataset: the
stem stride (fixed at 4 - an explicit `--target_token_grid` still resolves
it), and both subsample fractions. The EMA rate is re-resolved for the new
`--train_subsample_frac`, so its 3-epoch horizon holds. `--batch_size` is not
in the list and stays at `OPTIMAL_DEFAULTS`' 32; the v15 command it comes from
ran at v15's default of 256.

`--stage` is applied AFTER `RUN_DEFAULTS`, so `--stage fit` and `--stage
balance` still switch every regulariser, the class weighting and EMA off.
Before this layer existed, `RECON_DEFAULTS` was applied after the stage and
turned reconstruction back on under `--stage fit`.

Early stopping (`--early_stopping_enabled`, default on)
-------------------------------------------------------
`--early_stop_patience` was recorded in every `optimal` / `optimal_recon` /
v18 run and did nothing. The only patience rule in the trainer chain is
`TrainerG_v12Original._check_stopping_rules`, gated on the class attribute
`early_stopping_enabled`, which only `train_example_v16_original.py` ever set;
`TrainerG_v12._check_stopping_rules` has no patience rule of its own. This
entry point now sets it from `--early_stopping_enabled` (default `on`), so
`--early_stop_patience 8` stops the run after 8 epochs without a new best
checkpoint metric. A patience of 0 or None leaves the rule off - `--stage fit`
sets 0, and the inherited rule would otherwise stop after epoch 1.

v18 is NOT affected
-------------------
`train_example_v18.py` produced every number in the paper, and every
documented v18/v20 command relies on the defaults it had then. It builds on
`build_base_arg_parser` and `print_base_banner` - `RECON_DEFAULTS` on top of
`OPTIMAL_DEFAULTS`, without `RUN_DEFAULTS` or `--early_stopping_enabled` - so
its parsed arguments, banner, run name and stopping rules are unchanged.

Outputs added under `<run>/reconstruction/` - see
`train_example_v16_recon.py` for the full list; the ones that answer "did it
reconstruct anything":

    samples.npz              x_true / x_recon cubes, reflectance units
    samples_metrics.json     per-sample SAM / RMSE / PSNR / SID / SSIM (2-D
                             and spectral) / peak-position error
    status.json              written on EVERY path, including the skips
    provenance.json          which checkpoint, which selection rule

Usage
-----
    # the RUN_DEFAULTS configuration above, as is
    python train_example_v16_optimal_recon.py --data_dir ./data/<dataset>

    # explicit flags still win; early stopping can be turned off
    python train_example_v16_optimal_recon.py \
        --data_dir ./data/hsi_v8-80_10_10_importance-new/hsi \
        --batch_size 256 --early_stopping_enabled off

A finished run WITHOUT artifacts is not a lost cause: any run trained with
`--recon_mode latent` still has its decoder in `best_model.pt`, and
`scripts/export_recon_samples_v16.py --run_dir <run>` writes the same files
after the fact - including `--split test`, which no training run can produce.
"""

from __future__ import annotations

import argparse

import train_example_v16 as v16
import train_example_v16_optimal as optimal
import train_example_v16_recon as recon
from training import trainerg_v13
from training.trainerg_v13 import TrainerG_v13

# `dest -> value`, applied on top of `OPTIMAL_DEFAULTS`. Kept as data, and
# echoed by `print_banner` below, so a run that silently reverted to
# `--recon_mode none` says so in its own log rather than leaving an empty
# `reconstruction/` directory to be discovered later.
RECON_DEFAULTS = {
    "recon_mode": "latent",
    "lambda_mse": 0.1,
}

# `dest -> value`, applied on top of `RECON_DEFAULTS` and below `--stage` - see
# "What changes relative to train_example_v16_optimal.py" in the module
# docstring. Every entry is listed even where it repeats a lower layer, so this
# dict alone says what a bare command runs, and `config_deviations` measures
# the command line against it.
RUN_DEFAULTS = {
    # --- memory / cost guards (64x64 patches: gate G8 OOM, 4,096-token grid)
    "drift_check_patches": 500,
    "norm_stats_sample_cap": 1000,
    "patch_size": 4,
    # --- model ------------------------------------------------------------
    "architecture": "recursive",
    "normalization": "global_zscore",
    # --- reconstruction ---------------------------------------------------
    "recon_mode": "latent",
    "recon_out_activation": "auto",
    "lambda_mse": 0.1,
    "lambda_sam": 0.1,
    "recon_sample_per_class": 5,
    # --- objective --------------------------------------------------------
    "loss": "weighted_ce",
    "class_weight_power": 1.0,
    "sampler": "moderate_oversample",
    # --- regularisation ---------------------------------------------------
    "trm_drop_path": 0.1,
    "trm_dropout": 0.1,
    "classifier_dropout": 0.2,
    "augment_preset": "light",
    # --- budget -----------------------------------------------------------
    "epochs": 30,
    "lr": 1e-4,
    "amp": "bf16",
    "train_subsample_frac": 0.1,
    "train_subsample_mode": "per_epoch",
    "val_subsample_frac": 0.2,
    # --- stopping / per-patient reporting ---------------------------------
    "early_stop_patience": 8,
    "min_patient_recall": 0.5,
    # --- throughput -------------------------------------------------------
    "compile_model": "on",              # --compile
    "loader_mode": "performance",
    "dataset_storage": "mmap",
    "eval_artifact_stride": 5,
    "keep_last_n": 2,
    "seed": 42,
}


class TrainerG_v13Optimal(optimal.TrainerG_v12Optimal, TrainerG_v13):
    """The optimal-configuration trainer that also writes the artifacts.

    Empty by construction: every method comes from one of the two bases, and
    the MRO in the module docstring is the whole design. A method defined here
    would be a third implementation to keep in sync, which is the drift this
    file's composition exists to avoid.
    """


def build_base_arg_parser(argv=None):
    """`train_example_v16_optimal.build_arg_parser` + the `--recon_*` flags.

    The parser `train_example_v18.py` builds on: `RECON_DEFAULTS` over
    `OPTIMAL_DEFAULTS`, and nothing from `RUN_DEFAULTS`. This entry point's own
    parser is `build_arg_parser`, below.

    `train_example_v16_recon.build_arg_parser` captured `v16.build_arg_parser`
    in a module global AT ITS OWN IMPORT, so it cannot be pointed at the
    optimal parser by rebinding `v16.build_arg_parser` - the name it calls is
    `recon._v16_build_arg_parser`. Rebinding that one composes the two parsers
    without restating the four artifact flags here, which would be a second
    copy of their help text and defaults to keep in sync.
    """
    previous = recon._v16_build_arg_parser
    recon._v16_build_arg_parser = lambda: optimal.build_arg_parser(argv)
    try:
        p = recon.build_arg_parser()
    finally:
        recon._v16_build_arg_parser = previous

    unknown = set(RECON_DEFAULTS) - {a.dest for a in p._actions}
    if unknown:
        raise RuntimeError(f"RECON_DEFAULTS names flag(s) the parser does not have: {sorted(unknown)}")
    p.set_defaults(**RECON_DEFAULTS)
    p.description = ("GMedMamba-R (TRM), best-metrics configuration WITH reconstruction artifacts "
                     "- the reconstructed cubes and their metrics written to disk")
    return p


def _default_patch_size(argv=None):
    """`RUN_DEFAULTS["patch_size"]`, unless `--target_token_grid` was typed.

    `OPTIMAL_DEFAULTS` resolves the stem stride from the input size and
    `--target_token_grid`. A fixed default would make that flag parse and do
    nothing, so an explicit grid still resolves the stride the way it does on
    `train_example_v16_optimal.py`."""
    if optimal._peek_int("--target_token_grid", 0, argv):
        return optimal.resolve_patch_size(optimal._peek_data_dir(argv), optimal._peek_target_grid(argv))
    return RUN_DEFAULTS["patch_size"]


def build_arg_parser(argv=None):
    """`build_base_arg_parser` + `RUN_DEFAULTS` + `--early_stopping_enabled`.

    Order matters, and is the one in the module docstring: `RUN_DEFAULTS`
    first, then the EMA rate re-resolved for the subsample fraction now in
    force, then `--stage` again so a diagnostic stage still wins over
    `RUN_DEFAULTS`. Explicit flags win over all of it, as argparse defaults
    always do.
    """
    p = build_base_arg_parser(argv)

    unknown = set(RUN_DEFAULTS) - {a.dest for a in p._actions}
    if unknown:
        raise RuntimeError(f"RUN_DEFAULTS names flag(s) the parser does not have: {sorted(unknown)}")
    p.set_defaults(**RUN_DEFAULTS)
    p.set_defaults(patch_size=_default_patch_size(argv))

    # `optimal.build_arg_parser` resolved the EMA rate for ITS subsample
    # fraction; this layer changed the fraction, so resolve it again - with the
    # fraction and batch size this run will actually use.
    p.set_defaults(trm_ema_rate=optimal.resolve_ema_rate(
        optimal._peek_data_dir(argv),
        optimal._peek_int("--batch_size", p.get_default("batch_size"), argv),
        subsample_frac=optimal._peek_float("--train_subsample_frac",
                                            p.get_default("train_subsample_frac"), argv)))

    stage_peek = argparse.ArgumentParser(add_help=False)
    stage_peek.add_argument("--stage", default="full")
    p.set_defaults(**optimal._STAGES.get(stage_peek.parse_known_args(argv)[0].stage, {}))

    g = p.add_argument_group("optimal-recon")
    g.add_argument("--early_stopping_enabled", choices=["on", "off"], default="on",
                    help="stop after --early_stop_patience epochs without a new best "
                         "--checkpoint_metric (default on). Without it --early_stop_patience is "
                         "recorded in config.json and ignored, as it is on train_example_v18.py. "
                         "A patience of 0 or None leaves the rule off either way.")
    p.description = ("GMedMamba-R (TRM) with reconstruction artifacts - RUN_DEFAULTS configuration "
                     "(see the module docstring); every default is overridable")
    return p


def early_stopping_active(args) -> bool:
    """True when the patience rule should run: switched on AND a patience to
    count. `getattr` defaults cover `train_example_v18.py`'s arguments, which
    have no `--early_stopping_enabled` and keep the rule off."""
    return (getattr(args, "early_stopping_enabled", "off") == "on"
            and bool(getattr(args, "early_stop_patience", None)))


def build_run_name(args, *a, **kw) -> str:
    """`..._optimal` from the optimal entry point, plus `_recon`, so the two
    configurations do not land in directory names differing only by
    timestamp."""
    return f"{optimal.build_run_name_optimal(args, *a, **kw)}_recon"


_OPTIMAL_CONFIG_DEVIATIONS = optimal.config_deviations


def base_config_deviations(args, argv=None) -> list:
    """`optimal.config_deviations`, with `RECON_DEFAULTS` folded into the
    baseline it compares against. What `train_example_v18.py`'s banner uses.

    Without this the banner lists `recon_mode='latent'` and `lambda_mse=0.1`
    under "DEVIATIONS from this configuration (explicit flags)" on every
    single run - flags the user did not type, reported as if they had. The
    heading's promise is that anything printed under it is the command line
    overriding a default, so the fix belongs in the expectation, not the
    heading. A user who really does pass `--recon_mode none` still gets it
    listed, because then it differs from `RECON_DEFAULTS` too.
    """
    out = [d for d in _OPTIMAL_CONFIG_DEVIATIONS(args, argv)
           if not any(d.startswith(f"{dest}=") for dest in RECON_DEFAULTS)]
    for dest, want in sorted(RECON_DEFAULTS.items()):
        got = getattr(args, dest, None)
        if got != want:
            out.append(f"{dest}={got!r}  (optimal-recon: {want!r})")
    return out


def config_deviations(args, argv=None) -> list:
    """`base_config_deviations`, measured against what `build_arg_parser`
    actually resolves: `RUN_DEFAULTS`, then the stage, with the stride and EMA
    rate resolved the same way the parser resolved them. Only a flag the
    command line changed is listed."""
    stage = optimal._STAGES.get(getattr(args, "stage", "full"), {})
    expected = {**RECON_DEFAULTS, **RUN_DEFAULTS, "patch_size": _default_patch_size(argv), **stage}
    if "trm_ema_rate" not in stage:
        expected["trm_ema_rate"] = optimal.resolve_ema_rate(
            optimal._peek_data_dir(argv), args.batch_size, subsample_frac=args.train_subsample_frac)
    out = [d for d in base_config_deviations(args, argv) if d.split("=", 1)[0] not in expected]
    for dest, want in sorted(expected.items()):
        got = getattr(args, dest, None)
        if got != want:
            out.append(f"{dest}={got!r}  (optimal-recon: {want!r})")
    if args.early_stopping_enabled != "on":
        out.append(f"early_stopping_enabled={args.early_stopping_enabled!r}  (optimal-recon: 'on')")
    return out


def _print_banner(args, deviations_fn) -> None:
    # `optimal.print_banner` reads `config_deviations` as a module global, so this is the
    # one place the substitution has to happen. v17 S9 removed the OTHER four rebindings
    # (see `build_overrides`); this one stays because `optimal.print_banner` is a plain
    # function that looks the name up at call time, and threading a parameter through it
    # would mean editing its signature and every caller for a banner line. It is now
    # restored afterwards, so the mutation does not outlive the call.
    _saved = optimal.config_deviations
    optimal.config_deviations = deviations_fn
    try:
        optimal.print_banner(args)
    finally:
        optimal.config_deviations = _saved
    print(f"[recon] artifacts ON: recon_mode={args.recon_mode} lambda_mse={args.lambda_mse} "
          f"lambda_sam={args.lambda_sam}; {args.recon_sample_per_class} sample(s)/class at even "
          f"SAM quantiles, cubes capped at {args.recon_max_cubes}, png={args.recon_render_png}",
          flush=True)
    if args.recon_mode == "none":
        print("[recon] WARNING: --recon_mode none - there is no decoder, so no cube can be "
              "written. This entry point becomes train_example_v16_optimal.py with extra "
              "flags; reconstruction/status.json will record the skip.", flush=True)
    elif not args.lambda_mse and not args.lambda_sam:
        print("[recon] WARNING: --lambda_mse and --lambda_sam are both 0, so the decoder gets no "
              "gradient and reconstructs noise. Its metrics will be written and will be "
              "meaningless.", flush=True)


def print_base_banner(args) -> None:
    """The banner `train_example_v18.py` prints: deviations measured against
    `build_base_arg_parser`'s defaults."""
    _print_banner(args, base_config_deviations)


def print_banner(args) -> None:
    _print_banner(args, config_deviations)
    if early_stopping_active(args):
        print(f"[early-stop] ON: the run stops after {args.early_stop_patience} epochs without a "
              f"new best {args.checkpoint_metric} (--early_stopping_enabled off to disable).",
              flush=True)
    else:
        print(f"[early-stop] OFF (--early_stopping_enabled {args.early_stopping_enabled}, "
              f"--early_stop_patience {args.early_stop_patience}): the run goes to "
              f"--epochs {args.epochs} unless another stopping rule fires.", flush=True)


def build_overrides(args) -> "v16.Overrides":
    """`optimal.build_overrides`, with this file's parser, run name and trainer on top.

    v17 S9. This used to be `install_overrides`, and it was the clearest illustration of
    why the rebinding had to go: it called `optimal.install_overrides` (which assigned
    four names into `v16`) and then RE-ASSIGNED three of them, because the base call had
    just set the optimal-only parser and `TrainerG_v12Optimal`. Reading it required
    tracking which assignment won. Here the composition is a single expression and the
    later argument simply wins.

    `optimal.build_overrides` is still called for its side effect on
    `TrainerG_v12Optimal.suppressed_stop_reasons`, which `TrainerG_v13Optimal` inherits.

    `early_stopping_enabled` is set the same way, as a class attribute, because
    `TrainerG_v12Original._check_stopping_rules` reads it off the class and the
    trainer is constructed inside `v16._main`. `train_example_v18.py` calls this
    with its own arguments, which have no `--early_stopping_enabled`, so
    `early_stopping_active` is False there and `TrainerG_v18` keeps the rule off.
    """
    base = optimal.build_overrides(args)
    TrainerG_v13Optimal.early_stopping_enabled = early_stopping_active(args)
    return v16.Overrides(parser_fn=build_arg_parser,
                         run_name_fn=build_run_name,
                         model_fn=base.model_fn,
                         trainer_cls=TrainerG_v13Optimal)


def install_overrides(args) -> None:
    """Deprecated shim (v17 S9), kept for any external caller."""
    ov = build_overrides(args)
    v16.build_arg_parser, v16.build_run_name = ov.parser_fn, ov.run_name_fn
    v16.build_model, v16.TrainerG_v12 = ov.model_fn, ov.trainer_cls
    trainerg_v13.ENTRY_POINT_OVERRIDE = "train_example_v16_optimal_recon.py"


def main() -> None:
    args = build_arg_parser().parse_args()
    print_banner(args)
    trainerg_v13.ENTRY_POINT_OVERRIDE = "train_example_v16_optimal_recon.py"
    v16.main(build_overrides(args))


if __name__ == "__main__":
    main()
