# -*- coding: utf-8 -*-
"""
test_scheduler_warmup.py
===========================
MedMamba-SS-TRM v16 plan, Stage 3.2 verification (E-2).

`train_example_v16.build_warmup_cosine_scheduler`: LR rises over warmup then
decays; `total_steps` matches `epochs * steps_per_epoch`.
"""

import torch

from training.train_pipeline import build_warmup_cosine_scheduler


def _lrs(optimizer, scheduler, n_steps):
    out = [optimizer.param_groups[0]["lr"]]
    for _ in range(n_steps):
        optimizer.step()
        scheduler.step()
        out.append(optimizer.param_groups[0]["lr"])
    return out


def _make_opt(lr=1e-3):
    p = torch.nn.Parameter(torch.zeros(1))
    return torch.optim.AdamW([p], lr=lr)


def test_lr_rises_over_warmup_then_decays():
    opt = _make_opt(lr=1e-3)
    total_steps = 100
    scheduler = build_warmup_cosine_scheduler(opt, total_steps, warmup_steps=20)
    lrs = _lrs(opt, scheduler, total_steps)

    # Rises (non-decreasing) through warmup...
    warmup_lrs = lrs[:21]
    assert all(b >= a - 1e-12 for a, b in zip(warmup_lrs, warmup_lrs[1:]))
    assert warmup_lrs[-1] > warmup_lrs[0]

    # ...then decays afterward.
    post_warmup = lrs[20:]
    assert post_warmup[-1] < post_warmup[0]
    # Cosine decay reaches (near) zero at the end.
    assert lrs[-1] < 1e-3 * 0.05


def test_default_warmup_is_3pct_floored_at_200():
    opt = _make_opt()
    scheduler = build_warmup_cosine_scheduler(opt, total_steps=100000, warmup_steps=None)
    assert scheduler.warmup_steps == 3000  # 3% of 100000

    opt2 = _make_opt()
    scheduler2 = build_warmup_cosine_scheduler(opt2, total_steps=1000, warmup_steps=None)
    assert scheduler2.warmup_steps == 200  # floor, since 3% of 1000 is 30


def test_total_steps_matches_epochs_times_steps_per_epoch():
    epochs, steps_per_epoch = 5, 37
    total_steps = epochs * steps_per_epoch
    opt = _make_opt()
    scheduler = build_warmup_cosine_scheduler(opt, total_steps, warmup_steps=10)
    assert scheduler.total_steps == total_steps == 185


def test_scheduler_interval_step_wired_into_trainer():
    """E-2's other half: TrainerG_v12 must actually call scheduler.step()
    once per optimizer step when scheduler_interval='step', not once per
    epoch."""
    from training.trainerg_v12 import TrainerG_v12
    import inspect
    src = inspect.getsource(TrainerG_v12._train_one_epoch_deep_supervision)
    assert "self.scheduler_interval == \"step\"" in src
    assert "self.scheduler.step()" in src


if __name__ == "__main__":
    test_lr_rises_over_warmup_then_decays()
    test_default_warmup_is_3pct_floored_at_200()
    test_total_steps_matches_epochs_times_steps_per_epoch()
    test_scheduler_interval_step_wired_into_trainer()
    print("OK")
