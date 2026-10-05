# -*- coding: utf-8 -*-
"""
test_augmentation_persistent_workers.py
==========================================
MedMamba-SS-TRM v16 plan, Stage 3.4 verification (E-4).

`AugmentedPatchDatasetV16.set_epoch` must actually reach persistent DataLoader
workers - the frozen `AugmentedPatchDataset.set_epoch` (a plain attribute)
does not, which is why `--loader_mode performance` (persistent_workers=True)
has replayed epoch 0's augmentation draw forever in every real run.
"""

import random

import torch
from torch.utils.data import DataLoader, Dataset

from training.augmentation import item_augmentation_seed
from training.augmentation_v16 import AugmentedPatchDatasetV16


class _TinyBase(Dataset):
    """Deterministic base dataset: patch value == index, so any perturbation
    a worker applies is directly attributable to the augmentation draw."""
    def __init__(self, n=8):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return torch.full((1,), float(idx)), idx


def _additive_noise(patch, rng: random.Random, gen: torch.Generator):
    return patch + torch.randn(patch.shape, generator=gen)


def _collect_one_epoch(loader, ds, epoch):
    ds.set_epoch(epoch)
    out = []
    for x, _y in loader:
        out.append(x.clone())
    return torch.cat(out)


def test_set_epoch_changes_output_under_persistent_workers():
    base = _TinyBase(n=8)
    ds = AugmentedPatchDatasetV16(base, _additive_noise, seed=0)
    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=2, persistent_workers=True)

    epoch0 = _collect_one_epoch(loader, ds, 0)
    epoch1 = _collect_one_epoch(loader, ds, 1)

    assert not torch.allclose(epoch0, epoch1), (
        "epoch 0 and epoch 1 produced IDENTICAL augmented output under persistent_workers=True - "
        "set_epoch did not reach the workers (the E-4 bug, reproduced)")
    del loader  # shuts down the persistent workers


def test_shared_epoch_matches_main_process_seed_formula():
    base = _TinyBase(n=4)
    ds = AugmentedPatchDatasetV16(base, lambda p, rng, gen: p, seed=7)
    ds.set_epoch(3)
    assert ds._current_epoch() == 3
    expected = item_augmentation_seed(7, 3, 2, worker_id=0)
    # __getitem__ in the main process (worker_info is None -> worker_id=0)
    # must use exactly this seed formula, matching the frozen module's.
    s = item_augmentation_seed(ds.seed, ds._current_epoch(), 2, worker_id=0)
    assert s == expected


def test_falls_back_gracefully_without_workers():
    base = _TinyBase(n=4)
    ds = AugmentedPatchDatasetV16(base, _additive_noise, seed=0)
    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)
    e0 = _collect_one_epoch(loader, ds, 0)
    e1 = _collect_one_epoch(loader, ds, 1)
    assert not torch.allclose(e0, e1)


if __name__ == "__main__":
    test_set_epoch_changes_output_under_persistent_workers()
    test_shared_epoch_matches_main_process_seed_formula()
    test_falls_back_gracefully_without_workers()
    print("OK")
