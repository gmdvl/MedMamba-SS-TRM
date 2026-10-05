# -*- coding: utf-8 -*-
"""
test_run_naming.py
====================
Unit tests for `training/run_naming.py` - the experiment directory naming
introduced in train_example_v14.py, replacing
`gmedmamba_stageG_{architecture}_run_{timestamp}`.

Pure string/path logic: no GPU, no dataset, no torch. `infer_modality` is the
only function that touches the filesystem, and it is exercised against tmp_path.
"""

from argparse import Namespace

import pytest

from training.run_naming import build_run_name, dataset_slug, infer_modality


def _args(**over):
    base = dict(data_dir="./data/pad_v6/", architecture="recursive", batch_size=256,
                amp="bf16", trm_mixer="mlp", trm_deep_supervision_steps=3,
                fusion_type="se_gate")
    base.update(over)
    return Namespace(**base)


# ----------------------------------------------------------------------
# dataset_slug
# ----------------------------------------------------------------------

@pytest.mark.parametrize("data_dir, expected", [
    ("./data/pad_v6/", "pad_v6"),
    ("./data/pad_v6", "pad_v6"),
    ("data/pad_v6//", "pad_v6"),
    ("/abs/path/to/pad_v6", "pad_v6"),
    # An uninformative leaf borrows its parent, or every HSI run would be "hsi".
    ("./data/hsi_v7/hsi/", "hsi_v7-hsi"),
    ("./data/hsi_v7/rgb", "hsi_v7-rgb"),
    ("./data/hsi_v7/HSI", "hsi_v7-HSI"),      # case-insensitive match, original case kept
    ("./data_pad_v4-v4/", "data_pad_v4-v4"),
])
def test_slug_basic(data_dir, expected):
    assert dataset_slug(data_dir) == expected


def test_slug_uninformative_leaf_with_no_parent():
    """A bare "hsi" has no parent to borrow - it must not crash or return ""."""
    assert dataset_slug("hsi") == "hsi"


def test_slug_empty_and_dot_inputs_fall_back():
    for bad in ("", ".", "./"):
        assert dataset_slug(bad) == "dataset"


def test_slug_sanitizes_unsafe_characters():
    slug = dataset_slug("./data/we ird:name*v2/")
    assert slug == "we-ird-name-v2"
    assert all(c.isalnum() or c in "._-" for c in slug)


def test_slug_truncates_keeping_the_versioned_tail():
    """Dataset dirs here are versioned by SUFFIX, so truncation keeps the tail."""
    slug = dataset_slug("./data_hsi_v6-v4-nb32_mb_40_v2.1/hsi/")
    assert len(slug) <= 24
    assert slug.endswith("-hsi")
    # the distinguishing version survives; the generic "data_hsi_" prefix does not
    assert "v2.1" in slug


def test_slug_respects_explicit_max_len():
    assert len(dataset_slug("./data/abcdefghijklmnopqrstuvwxyz/", max_len=8)) <= 8


# ----------------------------------------------------------------------
# infer_modality - must match train_example_v6.discover_data's rule exactly
# ----------------------------------------------------------------------

def test_modality_rgb_when_no_wavelengths_file(tmp_path):
    assert infer_modality(str(tmp_path)) == "rgb"


def test_modality_hsi_only_for_the_exact_filename(tmp_path):
    (tmp_path / "wavelengths.npy").write_bytes(b"")
    assert infer_modality(str(tmp_path)) == "hsi"


def test_modality_ignores_lookalike_wavelength_files(tmp_path):
    """discover_data checks for `wavelengths.npy` and nothing else - a dataset
    carrying only selected_wavelengths.npy trains silently as RGB, and the run
    name must report that same (surprising) truth rather than a nicer guess."""
    (tmp_path / "selected_wavelengths.npy").write_bytes(b"")
    (tmp_path / "wavelengths_full.npy").write_bytes(b"")
    assert infer_modality(str(tmp_path)) == "rgb"


# ----------------------------------------------------------------------
# build_run_name
# ----------------------------------------------------------------------

def test_recursive_name_carries_mixer_and_supervision_steps():
    name = build_run_name(_args(), timestamp="20260901_140126", modality="rgb")
    assert name == "20260901_140126_pad_v6_recursive_rgb_mlp_sup3_bs256_bf16"


def test_recursive_ss2d_is_visible_in_the_name():
    """The 39x-slower mixer must be obvious from `ls`, not buried in config.json."""
    name = build_run_name(_args(trm_mixer="ss2d"), timestamp="T", modality="rgb")
    assert "ss2d_sup3" in name


def test_attention_mixer_is_abbreviated():
    name = build_run_name(_args(trm_mixer="attention"), timestamp="T", modality="rgb")
    assert "_attn_sup3_" in name


def test_split_omits_the_architecture_key_entirely():
    name = build_run_name(_args(architecture="split"), timestamp="T", modality="hsi")
    assert name == "T_pad_v6_split_hsi_bs256_bf16"
    assert "__" not in name          # no empty field left behind


def test_fullchannel_omits_the_architecture_key():
    name = build_run_name(_args(architecture="fullchannel"), timestamp="T", modality="rgb")
    assert name == "T_pad_v6_fullchannel_rgb_bs256_bf16"


def test_efficient_carries_the_fusion_type_without_underscores():
    name = build_run_name(_args(architecture="efficient"), timestamp="T", modality="rgb")
    assert name == "T_pad_v6_efficient_rgb_segate_bs256_bf16"


def test_amp_off_is_named_fp32_not_off():
    """"off" reads as "no AMP setting recorded"; fp32 states what actually ran."""
    name = build_run_name(_args(amp="off"), timestamp="T", modality="rgb")
    assert name.endswith("_fp32")


@pytest.mark.parametrize("amp", ["bf16", "fp16", "auto"])
def test_other_amp_modes_pass_through(amp):
    assert build_run_name(_args(amp=amp), timestamp="T", modality="rgb").endswith(f"_{amp}")


def test_hsi_dataset_dir_produces_a_readable_name():
    name = build_run_name(_args(data_dir="./data/hsi_v7/hsi/", architecture="split"),
                          timestamp="20260901_151203", modality="hsi")
    assert name == "20260901_151203_hsi_v7-hsi_split_hsi_bs256_bf16"


def test_modality_defaults_to_filesystem_inference(tmp_path):
    (tmp_path / "wavelengths.npy").write_bytes(b"")
    name = build_run_name(_args(data_dir=str(tmp_path), architecture="split"), timestamp="T")
    assert name.split("_")[-3] == "hsi"


def test_timestamp_is_first_so_ls_sorts_chronologically():
    early = build_run_name(_args(), timestamp="20260901_010101", modality="rgb")
    late = build_run_name(_args(data_dir="./data/aaa/"), timestamp="20260901_020202",
                          modality="rgb")
    # later run sorts after the earlier one even though its dataset sorts first
    assert sorted([late, early]) == [early, late]


def test_name_is_filesystem_safe():
    name = build_run_name(_args(data_dir="./data/we ird:name/"), timestamp="T", modality="rgb")
    assert all(c.isalnum() or c in "._-" for c in name)


# ======================================================================
# v15 R6.2 - the settings that changed the result belong in the name
# ======================================================================

from training.run_naming import modality_evidence, modality_warnings   # noqa: E402


def _v15_args(**over):
    """A v15 namespace: v14's fields plus the ones R6.2 added to the
    allowlist. Defaults here are v15's defaults, so a bare call must produce
    the same short name v14 produced."""
    base = dict(data_dir="./data/pad_v6/", architecture="recursive", batch_size=256,
                amp="bf16", trm_mixer="mlp", trm_deep_supervision_steps=3,
                fusion_type="se_gate", loss="ce", sampler="none", normalization=None,
                spectral_token_fusion="concat_mlp",
                train_subsample_frac=1.0, train_subsample_mode="per_epoch",
                val_subsample_frac=1.0, seed=42)
    base.update(over)
    return Namespace(**base)


def test_all_defaults_reproduce_the_v14_short_name():
    name = build_run_name(_v15_args(), timestamp="20260901_140126", modality="rgb")
    assert name == "20260901_140126_pad_v6_recursive_rgb_mlp_sup3_bs256_bf16"


def test_a_v14_namespace_without_the_new_fields_still_works():
    """`build_run_name` must stay a drop-in for `train_example_v14.py`, whose
    namespace has no --normalization / --val_subsample_frac at all."""
    name = build_run_name(_args(), timestamp="20260901_140126", modality="rgb")
    assert name == "20260901_140126_pad_v6_recursive_rgb_mlp_sup3_bs256_bf16"


def test_non_default_loss_and_sampler_appear():
    name = build_run_name(_v15_args(loss="weighted_ce", sampler="moderate_oversample"),
                           timestamp="20260901_140126", modality="rgb")
    assert name.endswith("_bs256_bf16_wce_modover")


def test_subsample_fractions_appear_without_a_decimal_point():
    name = build_run_name(_v15_args(train_subsample_frac=0.05, val_subsample_frac=0.1),
                           timestamp="T", modality="hsi")
    assert "_sub05_" in name and name.endswith("_vsub1")
    assert "." not in name.split("_")[-1]


def test_non_default_seed_appears_and_the_default_does_not():
    assert "seed" not in build_run_name(_v15_args(seed=42), timestamp="T", modality="rgb")
    assert build_run_name(_v15_args(seed=7), timestamp="T", modality="rgb").endswith("_seed7")


def test_normalization_appears_when_set():
    name = build_run_name(_v15_args(normalization="global_zscore"), timestamp="T", modality="hsi")
    assert name.endswith("_zscore")


def test_settings_order_is_stable():
    """Two namespaces with the same settings must name identically regardless
    of attribute insertion order, or `ls` stops being a comparison tool."""
    a = build_run_name(_v15_args(loss="weighted_ce", seed=7, train_subsample_frac=0.05),
                        timestamp="T", modality="hsi")
    b = build_run_name(Namespace(**dict(reversed(list(vars(
        _v15_args(loss="weighted_ce", seed=7, train_subsample_frac=0.05)).items())))),
        timestamp="T", modality="hsi")
    assert a == b


def test_the_two_evidence_runs_would_now_be_distinguishable():
    """The HSI and RGB evidence runs differed in --loss, --sampler and
    --train_subsample_frac, and their v14 names said none of it."""
    hsi = build_run_name(_v15_args(data_dir="./data/hsi_v7-80_10_10/hsi", loss="weighted_ce",
                                    train_subsample_frac=0.05),
                          timestamp="20260902_111050", modality="hsi")
    rgb = build_run_name(_v15_args(sampler="moderate_oversample"),
                          timestamp="20260901_195001", modality="rgb")
    assert "wce" in hsi and "sub05" in hsi
    assert "modover" in rgb
    assert hsi.split("_", 2)[2] != rgb.split("_", 2)[2]


# ======================================================================
# v15 R6.6 - a spectral dataset must not train silently as RGB
# ======================================================================

def test_no_warnings_for_a_well_formed_dataset(tmp_path):
    (tmp_path / "wavelengths.npy").write_bytes(b"")
    (tmp_path / "X_train.npy").write_bytes(b"")
    assert modality_warnings(str(tmp_path)) == []


def test_warns_when_only_a_lookalike_wavelength_file_exists(tmp_path):
    (tmp_path / "selected_wavelengths.npy").write_bytes(b"")
    (tmp_path / "X_train.npy").write_bytes(b"")
    warnings = modality_warnings(str(tmp_path))
    assert any("MODALITY" in w and "selected_wavelengths.npy" in w for w in warnings)
    # ...and the inferred modality is still the honest one.
    assert infer_modality(str(tmp_path)) == "rgb"


def test_warns_when_the_merge_step_never_ran(tmp_path):
    (tmp_path / "hsi_batches").mkdir()
    warnings = modality_warnings(str(tmp_path))
    assert any("finalize/merge step never ran" in w for w in warnings)


def test_warns_about_leftover_shards_next_to_merged_arrays(tmp_path):
    (tmp_path / "hsi_batches").mkdir()
    (tmp_path / "X_train.npy").write_bytes(b"")
    assert any("stale" in w for w in modality_warnings(str(tmp_path)))


def test_modality_evidence_reports_what_it_looked_at(tmp_path):
    (tmp_path / "wavelengths.npy").write_bytes(b"")
    ev = modality_evidence(str(tmp_path))
    assert ev["canonical_wavelengths"] is True
    assert ev["inferred_modality"] == "hsi"
    assert ev["other_wavelength_files"] == []


def test_the_tokenizer_ablation_is_visible_in_the_name():
    """The v15 headline ablation runs three tokenizers and nothing else; their
    directory names must differ."""
    names = {f: build_run_name(_v15_args(spectral_token_fusion=f), timestamp="T", modality="rgb")
             for f in ("add", "scaled", "concat_mlp")}
    assert len(set(names.values())) == 3
    assert names["concat_mlp"].endswith("_bf16"), "the v15 default stays unmarked"
    assert names["add"].endswith("_ADD-legacy"), "the legacy defect must be loud in the name"
    assert names["scaled"].endswith("_scaled")
