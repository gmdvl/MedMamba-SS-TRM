# -*- coding: utf-8 -*-
"""
test_profile_names.py
======================
The `--profile` names (2026-10-01): a name ending in `_norecon` is a promise
that the profile trains without reconstruction, and every other profile does
reconstruct. The old names keep working and resolve to the new ones.

Run with: pytest tests/test_profile_names.py -q
"""

import pytest

from training.train_cli import resolve
from training.train_profiles import DEFAULT_PROFILE, PROFILE_ALIASES, PROFILES

HSI = "data/hsi_v8-80_10_10_importance-new/hsi"


@pytest.mark.parametrize("profile", sorted(PROFILES))
def test_norecon_suffix_matches_what_the_profile_trains(profile):
    args = resolve(["--data_dir", HSI, "--profile", profile])[0].args
    if profile.endswith("_norecon"):
        assert (args.recon_mode, args.lambda_mse, args.lambda_sam) == ("none", 0.0, 0.0)
        assert "NO reconstruction" in PROFILES[profile].summary
    else:
        assert args.recon_mode != "none" and args.lambda_mse > 0
        assert "reconstruction" in PROFILES[profile].summary


def test_every_old_name_maps_to_a_current_profile():
    assert set(PROFILE_ALIASES) == {"base", "original", "optimal", "v18"}
    assert set(PROFILE_ALIASES.values()) == set(PROFILES)
    assert not set(PROFILE_ALIASES) & set(PROFILES)
    assert DEFAULT_PROFILE in PROFILES


@pytest.mark.parametrize("old,new", sorted(PROFILE_ALIASES.items()))
def test_an_old_name_resolves_exactly_like_the_new_one(old, new):
    parsed_old, notes = resolve(["--data_dir", HSI, "--profile", old])
    parsed_new, _ = resolve(["--data_dir", HSI, "--profile", new])
    assert parsed_old.profile == parsed_old.args.profile == new
    assert vars(parsed_old.args) == vars(parsed_new.args)
    assert notes[0].startswith(f"profile={new}") and f"--profile {old}" in notes[0]


def test_paper_recipe_is_the_default():
    assert DEFAULT_PROFILE == "paper_recipe"
    bare = resolve(["--data_dir", HSI])[0]
    typed = resolve(["--data_dir", HSI, "--profile", "paper_recipe"])[0]
    assert bare.profile == "paper_recipe" and vars(bare.args) == vars(typed.args)


def test_an_unknown_name_is_still_refused():
    with pytest.raises(SystemExit):
        resolve(["--data_dir", HSI, "--profile", "no_such_profile"])
