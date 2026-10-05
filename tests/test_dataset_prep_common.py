# -*- coding: utf-8 -*-
"""tests/test_dataset_prep_common.py - MedMamba-SS-TRM data-integrity remediation, Part B Stage 3/4.

Run with: pytest test_dataset_prep_common.py -v
"""
import numpy as np
import pytest

from training.dataset_prep_common import (
    SPLIT_ALGO_VERSION, SkipRegistry, build_balance_selection, grouped_stratified_kfold,
    kfold_split, load_split_assignment, parse_split_spec, patient_split_three,
    resolve_split_fractions, stratified_group_split_three, stratified_patient_split_three,
    write_split_assignment,
)


def _covered(patients, p2c):
    c = set()
    for p in patients:
        c |= set(p2c[p])
    return c


def test_split_algo_version_is_2():
    assert SPLIT_ALGO_VERSION == 2


def test_fill_pass_guarantees_coverage_for_common_class():
    # class 2 is carried only by patients who ALSO carry class 1 (a rarer-looking
    # pairing) - the pre-fill grouping-by-rarest-class step would leave class 2
    # out of validation/test. The fill pass must pull carriers back in.
    p2c = {
        "p1": {0}, "p2": {0}, "p3": {0}, "p4": {0},
        "pA": {1, 2}, "pB": {1, 2}, "pC": {1, 2}, "pD": {1, 2},
        "pR1": {3}, "pR2": {3}, "pR3": {3},
    }
    tr, va, te, warns = stratified_patient_split_three(p2c, "80_10_10", 0)
    for split in (va, te):
        assert {0, 1, 2, 3} <= _covered(split, p2c), f"missing coverage in {sorted(split)}"
    assert set().union(tr, va, te) == set(p2c) and not (tr & va) and not (tr & te) and not (va & te)


def test_genuine_shortage_warns_not_crashes():
    # class 9 has a single patient - coverage is impossible, must be a warning.
    p2c = {f"p{i}": {0} for i in range(8)}
    p2c["lonely"] = {9}
    tr, va, te, warns = stratified_patient_split_three(p2c, "80_10_10", 1)
    assert "lonely" in tr
    assert any("9" in w and "patient" in w for w in warns)


def test_legacy_random_is_deterministic_and_disjoint():
    pats = [f"p{i}" for i in range(20)]
    a = patient_split_three(pats, "80_10_10", 3)
    b = patient_split_three(pats, "80_10_10", 3)
    assert a == b
    tr, va, te, _ = a
    assert not (tr & va) and not (tr & te) and not (va & te)
    assert set().union(tr, va, te) == set(pats)


def test_build_balance_selection_deterministic():
    y = np.array([0] * 12 + [1] * 3 + [2] * 1, dtype=np.int64)
    s1, b1, a1 = build_balance_selection(y, "oversample", 0)
    s2, b2, a2 = build_balance_selection(y, "oversample", 0)
    assert np.array_equal(s1, s2)
    assert len(set(a1.values())) == 1 and b1 == {0: 12, 1: 3, 2: 1}
    su, bu, au = build_balance_selection(y, "undersample", 0)
    assert len(set(au.values())) == 1 and max(au.values()) == 1


def test_skip_registry():
    r = SkipRegistry()
    r.add("img_1", "corrupt jpeg", stage="decode")
    assert "img_1" in r and len(r) == 1
    assert "img_1" in r.summary()
    r2 = SkipRegistry(r.as_dict())
    assert "img_1" in r2


# ---------------------------------------------------------------------------
# Part C - flexible split specs
# ---------------------------------------------------------------------------

def test_parse_split_spec_forms():
    assert parse_split_spec("80_20") == (0.80, 0.00, 0.20)          # preset key
    assert parse_split_spec("75/15/10") == pytest.approx((0.75, 0.15, 0.10))
    assert parse_split_spec("80/20") == pytest.approx((0.80, 0.0, 0.20))
    assert parse_split_spec("0.7/0.15/0.15") == pytest.approx((0.7, 0.15, 0.15))
    # resolve_split_fractions now also accepts a free-form spec
    assert resolve_split_fractions("60/20/20") == pytest.approx((0.6, 0.2, 0.2))
    for bad in ("50/40", "1/2/3/4", "abc", "-10/110"):
        with pytest.raises(ValueError):
            parse_split_spec(bad)


def test_grouped_kfold_disjoint_complete_deterministic():
    p2c = {f"p{i}": {i % 4} for i in range(40)}
    a = grouped_stratified_kfold(p2c, 5, 0)
    b = grouped_stratified_kfold(p2c, 5, 0)
    assert [sorted(f) for f in a] == [sorted(f) for f in b]        # deterministic
    assert set().union(*a) == set(p2c)                             # complete
    for i in range(5):
        for j in range(i + 1, 5):
            assert not (a[i] & a[j])                               # disjoint
    # every class present in every fold (balanced pool)
    for f in a:
        assert {p2c[g].pop() if isinstance(p2c[g], set) else p2c[g] for g in f}  # non-empty


def test_kfold_split_three_way_disjoint():
    p2c = {f"p{i}": {i % 3} for i in range(30)}
    tr, va, te, _w = kfold_split(p2c, 5, 2, seed=0, cv_val_frac=0.15)
    assert tr | va | te == set(p2c)
    assert not (tr & va) and not (tr & te) and not (va & te)
    assert va and te                                               # both non-empty


def test_stratified_group_split_empty_strat_delegates():
    p2c = {f"p{i}": {i % 3} for i in range(24)}
    g2m = {g: (cls, ()) for g, cls in p2c.items()}
    assert stratified_group_split_three(g2m, "80_10_10", 0) == \
        stratified_patient_split_three(p2c, "80_10_10", 0)


def test_stratified_group_split_multikey_covers_classes():
    g2m = {f"p{i}": ({i % 3}, ("A" if i % 2 else "B",)) for i in range(30)}
    tr, va, te, _w = stratified_group_split_three(g2m, "80_10_10", 0)
    assert tr | va | te == set(g2m)
    for split in (va, te):
        assert {list(g2m[g][0])[0] for g in split} == {0, 1, 2}


def test_split_assignment_roundtrip(tmp_path):
    p2c = {f"p{i}": {i % 3} for i in range(21)}
    tr, va, te, _ = stratified_patient_split_three(p2c, "80_10_10", 0)
    groups = {**{g: "train" for g in tr}, **{g: "val" for g in va}, **{g: "test" for g in te}}
    p = str(tmp_path / "split_assignment.json")
    write_split_assignment(p, groups=groups, strategy="stratified", seed=0, spec="80_10_10",
                            fractions=(0.8, 0.1, 0.1), class_names=["a", "b", "c"])
    rtr, rva, rte, meta = load_split_assignment(p)
    assert (rtr, rva, rte) == (tr, va, te)
    assert meta["strategy"] == "stratified" and meta["spec"] == "80_10_10"
