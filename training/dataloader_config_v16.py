# -*- coding: utf-8 -*-
"""
training/dataloader_config_v16.py
====================================
MedMamba-SS-TRM v16 plan, Stage 3.4 follow-on (E-4). `training/dataloader_config.py`
is imported at `train_example_v15.py:117` and is therefore FROZEN.

Its own docstring (`dataloader_config.py:114-131`, on `train_val_policies`)
promises validation loaders are "never persistent", but the implementation
keeps `train_policy.persistent_workers` whenever the resolved `val_workers >
0` - so under `--loader_mode performance` (persistent_workers=True on the
train policy), validation gets `val_workers = num_workers - 1 > 0` workers
that ARE persistent, and the run holds those idle worker processes alive for
its entire life for no benefit (a deterministic, single-pass-per-epoch
validation loader has nothing to gain from persistence).

`train_val_policies_v16` calls the frozen `train_val_policies` unchanged and
then forces `persistent_workers=False` on the validation policy - a
four-line wrapper, not a copy.
"""

from __future__ import annotations

from dataclasses import replace

from training.dataloader_config import train_val_policies, DataLoaderPolicy  # frozen


def train_val_policies_v16(loader_mode: str, num_workers, prefetch_factor,
                            persistent_workers, pin_memory):
    train_policy, val_policy = train_val_policies(loader_mode, num_workers, prefetch_factor,
                                                    persistent_workers, pin_memory)
    if val_policy.persistent_workers:
        val_policy = replace(val_policy, persistent_workers=False)
    return train_policy, val_policy
