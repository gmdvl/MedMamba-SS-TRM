# -*- coding: utf-8 -*-
"""
test_train_parity.py
=====================
`train.py --profile X` resolves EXACTLY what the entry point it replaced
resolved: every argument, the run directory name, and the two stopping knobs
that used to be class attributes.

The former entry points live in `archive/` (byte-identical, on
`test_frozen_files_untouched.py:FROZEN_GLOBS`), so they stay the reference. The
end-to-end half - both sides actually training on a tiny dataset - is
`test_train_e2e_parity.py`.

Run with: pytest tests/test_train_parity.py -q
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if not (REPO_ROOT / "train_example_v18.py").exists():
    sys.path.append(str(REPO_ROOT / "archive"))     # append: the live tree wins

import train_example_v15 as v15                      # noqa: E402
import train_example_v16 as old_v16                  # noqa: E402
import train_example_v16_optimal as old_optimal      # noqa: E402
import train_example_v16_original as old_original    # noqa: E402
import train_example_v16_recon as old_recon          # noqa: E402
import train_example_v18 as old_v18                  # noqa: E402
from training.train_cli import resolve               # noqa: E402
from training.train_pipeline import build_run_name   # noqa: E402
from training.trainerg import stop_knobs             # noqa: E402

HSI = "data/hsi_v8-80_10_10_importance-new/hsi"
RGB = "data/hsi_v8-80_10_10_importance-new/rgb"
PAD = "data/pad_optimal"
REF = ["--batch_size", "256", "--lambda_sam", "0.1", "--train_subsample_frac", "0.102",
       "--val_subsample_frac", "0.0986"]

# old entry point -> (old parser builder taking argv, old overrides builder). The
# keys name the archived entry points, not profiles: NEW_PROFILE maps them.
OLD = {
    "base": (lambda argv: old_v16.build_arg_parser(), None),
    "recon": (lambda argv: old_recon.build_arg_parser(), None),
    "original": (old_original.build_arg_parser, None),
    "optimal": (old_optimal.build_arg_parser, old_optimal.build_overrides),
    "v18": (old_v18.build_arg_parser, old_v18.build_overrides),
}
# old entry point -> the profile that replaces it
NEW_PROFILE = {"base": "pipeline_defaults", "recon": "pipeline_defaults",
               "original": "medmamba_protocol_norecon", "optimal": "pad_ufes_best_norecon",
               "v18": "paper_recipe"}
# dests renamed when two entry points' flags were merged into one
RENAMED = {"stop_on_val_acc_stall": "stop_on_metric_stall", "early_stopping_enabled": "early_stopping"}

ARGVS = {
    "base": [[], REF, ["--architecture", "split"], ["--recon_mode", "latent", "--lambda_sam", "0.1"]],
    "recon": [[], ["--recon_mode", "latent", "--recon_sample_per_class", "5"],
              ["--recon_render_png", "off", "--recon_max_cubes", "8"]],
    "original": [[], ["--architecture", "split"], ["--early_stopping", "on"], ["--global_stats", "identity"]],
    "optimal": [[], REF, ["--stage", "fit"], ["--stage", "balance"], ["--target_token_grid", "14"],
                ["--architecture", "split"], ["--batch_size", "64"]],
    "v18": [[], REF + ["--epochs", "12", "--run_tag", "abl-base"],
            REF + ["--trm_n_improve", "1", "--run_tag", "depth-n1"],
            ["--batch_size", "256", "--recon_mode", "none", "--run_tag", "recon-off"],
            REF + ["--no_use_wavelengths"], REF + ["--architecture", "split"], ["--stage", "fit"]],
}
CASES = [(prof, data, extra) for prof, argvs in ARGVS.items() for extra in argvs
         for data in (HSI, RGB, PAD)]


def _old_resolve(profile, argv, monkeypatch):
    """What the former entry point resolved: its parser, then `v16._main`'s
    safe-mode and v16 defaults with ITS explicit set."""
    monkeypatch.setattr(sys, "argv", ["old"] + argv)
    build, _ = OLD[profile]
    args = build(argv).parse_args(argv)
    explicit = old_v16.explicitly_passed(argv, parser_fn=lambda: build(argv))
    v15.apply_safe_mode(args, explicit)
    old_v16.apply_v16_defaults(args, explicit)
    return args


@pytest.mark.parametrize("profile,data,extra", CASES,
                         ids=[f"{p}-{Path(d).name}-{'_'.join(e) or 'bare'}" for p, d, e in CASES])
def test_every_argument_resolves_identically(profile, data, extra, monkeypatch):
    argv = ["--data_dir", data] + extra
    old = vars(_old_resolve(profile, argv, monkeypatch))
    new = vars(resolve(argv + ["--profile", NEW_PROFILE[profile]])[0].args)
    diff = {k: (v, new.get(RENAMED.get(k, k))) for k, v in old.items()
            if new.get(RENAMED.get(k, k)) != v}
    assert not diff, f"old != new for {diff}"


@pytest.mark.parametrize("profile,data,extra", CASES,
                         ids=[f"{p}-{Path(d).name}-{'_'.join(e) or 'bare'}" for p, d, e in CASES])
def test_run_name_is_byte_identical(profile, data, extra, monkeypatch):
    argv = ["--data_dir", data] + extra
    ts = "20260928_000000"
    old_args = _old_resolve(profile, argv, monkeypatch)
    old_name = {"base": old_v16.build_run_name, "recon": old_v16.build_run_name,
                "original": old_original.build_run_name_original,
                "optimal": old_optimal.build_run_name_optimal,
                "v18": old_v18.build_run_name}[profile](old_args, ts, modality="x")
    new_args = resolve(argv + ["--profile", NEW_PROFILE[profile]])[0].args
    assert build_run_name(new_args, ts, modality="x") == old_name


@pytest.mark.parametrize("profile,extra", [(p, e) for p, argvs in ARGVS.items() for e in argvs])
def test_stopping_knobs_match_the_old_trainer_class(profile, extra, monkeypatch):
    """The old entry points set these as class attributes at startup; the base
    and original trainers read their class defaults."""
    argv = ["--data_dir", HSI] + extra
    args = _old_resolve(profile, argv, monkeypatch)
    _, overrides = OLD[profile]
    if profile in ("base", "recon"):         # TrainerG_v12 / TrainerG_v13: no patience, stall rule on
        want = {"early_stopping_enabled": False, "suppressed_stop_reasons": frozenset()}
    elif profile == "original":
        old_original.install_overrides(args)
        cls = old_original.TrainerG_v12Original
        want = {"early_stopping_enabled": cls.early_stopping_enabled,
                "suppressed_stop_reasons": cls.suppressed_stop_reasons}
        old_v16.build_arg_parser = old_original._V16_BUILD_ARG_PARSER     # undo the rebinding
        old_v16.build_run_name = old_original._V16_BUILD_RUN_NAME
        old_v16.TrainerG_v12 = old_original._V16_TRAINER
        old_v16.compute_global_channel_stats = old_original._V16_GLOBAL_STATS
        old_v16.build_warmup_cosine_scheduler = old_original._V16_SCHEDULER
    else:
        cls = overrides(args).trainer_cls
        want = {"early_stopping_enabled": cls.early_stopping_enabled,
                "suppressed_stop_reasons": cls.suppressed_stop_reasons}
    got = stop_knobs(resolve(argv + ["--profile", NEW_PROFILE[profile]])[0].args)
    if profile == "original" and args.early_stopping == "on" and not args.early_stop_patience:
        want["early_stopping_enabled"] = False       # patience 0 used to stop at epoch 1
    assert got == want


@pytest.mark.parametrize("profile,schedule,stats", [
    ("pipeline_defaults", "warmup_cosine", "train_fit"),
    ("medmamba_protocol_norecon", "constant", "medmamba_fixed"),
    ("pad_ufes_best_norecon", "warmup_cosine", "train_fit"),
    ("paper_recipe", "warmup_cosine", "train_fit"),
])
def test_schedule_and_normalization_stats_follow_the_profile(profile, schedule, stats):
    """The two functions `train_example_v16_original.py` rebound into v16."""
    args = resolve(["--data_dir", HSI, "--profile", profile])[0].args
    assert (args.lr_schedule, args.global_stats) == (schedule, stats)


def test_safe_mode_now_overrides_profile_defaults():
    """The one deliberate change: the old explicit set counted every profile
    default as typed, so --safe_mode left an optimal-family run compiled, in
    bf16, on the performance loader."""
    args = resolve(["--data_dir", HSI, "--profile", "paper_recipe", "--safe_mode"])[0].args
    assert (args.compile_model, args.amp, args.loader_mode) == ("off", "off", "safe")
    typed = resolve(["--data_dir", HSI, "--profile", "paper_recipe", "--safe_mode", "--amp",
                     "bf16"])[0].args
    assert typed.amp == "bf16"


def test_stage_is_refused_outside_the_optimal_family():
    with pytest.raises(SystemExit):
        resolve(["--data_dir", HSI, "--profile", "pipeline_defaults", "--stage", "fit"])
