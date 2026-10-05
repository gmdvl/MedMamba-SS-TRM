# -*- coding: utf-8 -*-
"""
training/ema.py
======================
MedMamba-SS-TRM v17, Stage 12. `FastEMAHelper(EMAHelper)` - the EMA shadow update as two fused
kernels instead of a Python loop over `named_parameters()`. `medmamba_ss_trm_ema.py` is on the
FROZEN list, so it is subclassed, never edited - the same move `training/npy_data.py`
makes against the frozen `NpyDataset`.

The problem
-----------
`EMAHelper.update` (`medmamba_ss_trm_ema.py:66-71`) is:

    for name, param in module.named_parameters():
        self.shadow[name].data = (1.0 - self.mu) * param.data + self.mu * self.shadow[name].data

Per parameter tensor that is two out-of-place multiplies, one add, and a fresh allocation
that the previous shadow tensor is then dropped for. MedMamba-SS-TRM has **73 parameter
tensors**, so one optimizer step costs ~292 tiny kernel launches and 73 allocations -
every step, on a model whose whole forward is 0.446M parameters.

It is not a synchronisation problem (nothing here reads back to the host), it is a
launch-count and allocator problem, and it is CPU time the de-synced training loop in
`training/trainerg_v12_fast.py` needs in order to run ahead of the GPU at all.
`plan/v17_baseline.md:183` names this call as remaining overhead.

The fix
-------
Two fused in-place kernels over the whole parameter list:

    torch._foreach_mul_(shadow, mu)                      # s *= mu
    torch._foreach_add_(shadow, params, alpha=1 - mu)    # s += (1 - mu) * p

Why NOT `torch._foreach_lerp_(shadow, params, 1 - mu)`, which is also two launches and the
more obvious spelling: `lerp` computes `s + w*(p - s)`, a different expression from the
frozen module's `(1-w)*p + w*s`. They agree in exact arithmetic and round differently, and
at `mu=0.999` the map has a ~1000-step memory, so the difference ACCUMULATES. Measured here
(2000 updates, mu=0.999, 1000 elements, fp32), max absolute deviation from the frozen
expression:

    torch._foreach_lerp_                  3.10e-05
    _foreach_mul_ + _foreach_add_         3.58e-07      <- ~87x closer, same two launches

`best_model.pt` IS the EMA copy (`trainerg_v11.py:588`) and every reported validation number
is computed from it, so the closer variant is worth having for free.

Even so, a fast-EMA run is **not bit-identical** to a frozen-EMA one; it is `allclose`. The
load-bearing tests compare with tolerance (`test_trainerg_v11.py:294-297`,
`test_trm_integration.py:137-160`), so they pass - but a run that needs to be byte-compared
against one already in `experiments/` must use the frozen helper.

What is NOT re-implemented: `ema`, `ema_copy`, `state_dict`, `load_state_dict` and the
public `.shadow` dict are the frozen class's. That is what keeps the `"ema_shadow"`
checkpoint key (`trainerg_v11.py:600`), the per-epoch `deepcopy` validation path
(`trainerg_v10.py:270-273`) and `.shadow`'s `{name: tensor}` contract intact.

`torch._foreach_mul_` / `_foreach_add_` are private APIs and `requirements.txt` pins only
`torch>=2.1`, so both are probed at import and the helper degrades permanently to the frozen
loop on any failure. Same policy as `training/fused_stability_checks.py`: an optimisation must
never be what kills a run.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from medmamba_ss_trm_ema import EMAHelper                  # frozen - subclassed, not edited

_FUSED_OK = hasattr(torch, "_foreach_mul_") and hasattr(torch, "_foreach_add_")
_WARNED = False


def _degrade(detail: str = "") -> None:
    """Print once, then never again. `fused_stability_checks.py:75-82`'s message, verbatim shape."""
    global _FUSED_OK, _WARNED
    _FUSED_OK = False
    if not _WARNED:
        _WARNED = True
        print(f"[v17 S12] fused EMA update unavailable{detail}; falling back to the per-tensor "
              f"path for the rest of this run. Correct, just slower.", flush=True)


class FastEMAHelper(EMAHelper):
    """`EMAHelper` with the per-parameter Python loop replaced by two fused kernels.

    The foreach lists ALIAS the tensors in `self.shadow` - `_foreach_*_` mutates in place,
    so there is no write-back step and `self.shadow[name]` stays correct by construction.
    That aliasing is also the one thing that can go wrong silently, which is why
    `load_state_dict` is overridden below.
    """

    def __init__(self, mu: float = 0.999):
        super().__init__(mu=mu)
        self._names: list = []
        self._shadow_list = None
        self._params = None
        self._src = None

    # ------------------------------------------------------------------
    # cache management
    # ------------------------------------------------------------------

    def _invalidate(self) -> None:
        self._shadow_list = self._params = self._src = None

    def register(self, module: nn.Module) -> None:
        module = module.module if isinstance(module, nn.DataParallel) else module
        super().register(module)                     # same clones, same order, same filter
        self._names = [n for n, p in module.named_parameters() if p.requires_grad]
        self._invalidate()

    def load_state_dict(self, state_dict: dict) -> None:
        """Invalidate the alias cache - the frozen implementation REBINDS `self.shadow`
        wholesale (`medmamba_ss_trm_ema.py:92`), which strands every tensor `_bind` cached.

        Getting this wrong is silent and survives a smoke test: a resumed run would average
        into orphaned tensors while `state_dict()` kept returning the loaded shadow
        unchanged for the rest of the run, so the EMA would simply stop advancing.
        """
        super().load_state_dict(state_dict)
        self._invalidate()

    def _bind(self, module: nn.Module) -> bool:
        """Point `_shadow_list`/`_params` at this module's tensors. False = use the frozen
        path (the two lists could not be built or do not line up)."""
        if self._shadow_list is not None and self._src is module:
            return True
        if not self._names:
            self._names = [n for n, p in module.named_parameters() if p.requires_grad]
        named = dict(module.named_parameters())
        try:
            # Rebuilt BY NAME from the same key list, so the two lists cannot silently
            # misalign - which would corrupt the shadow rather than raise.
            self._params = [named[n] for n in self._names]
            self._shadow_list = [self.shadow[n] for n in self._names]
        except KeyError:
            self._invalidate()
            return False
        # Checked once per bind, never per step. `_foreach_add_` broadcasts, so a shape
        # mismatch would quietly produce the wrong shadow instead of failing.
        if not all(s.shape == p.shape and s.dtype == p.dtype and s.device == p.device
                   for s, p in zip(self._shadow_list, self._params)):
            self._invalidate()
            return False
        self._src = module
        return True

    # ------------------------------------------------------------------
    # the hot path
    # ------------------------------------------------------------------

    @torch.no_grad()
    def update(self, module: nn.Module) -> None:
        module = module.module if isinstance(module, nn.DataParallel) else module
        if not _FUSED_OK or not self._bind(module):
            return EMAHelper.update(self, module)             # frozen path
        try:
            torch._foreach_mul_(self._shadow_list, self.mu)
            torch._foreach_add_(self._shadow_list, self._params, alpha=1.0 - self.mu)
        except Exception as e:                                # pragma: no cover
            _degrade(f" ({type(e).__name__}: {e})")
            self._invalidate()
            EMAHelper.update(self, module)


def upgrade_ema_helper(helper, model, logger=None):
    """Return a `FastEMAHelper` carrying `helper`'s rate and shadow, or `helper` unchanged.

    The frozen `TrainerG_v10.__init__` constructs `EMAHelper` by name
    (`training/trainerg_v10.py:53,66`) and exposes no injection hook, so the object is
    replaced after construction - exactly as `training/trainerg_v12.py:245-266` replaces the
    `GradScaler` that frozen `trainerg_v4.py:133` built, and for the same reason.

    `register` re-clones from the live model, which is what `EMAHelper.register` did moments
    earlier, so the resulting shadow is identical. Safe against `--resume`: `load_checkpoint`
    restores `latest_ema.pt` AFTER `__init__` (`trainerg_v10.py:288-296`), and
    `load_state_dict` above invalidates the cache.
    """
    if helper is None or isinstance(helper, FastEMAHelper):
        return helper
    fast = FastEMAHelper(mu=helper.mu)
    fast.register(model)
    msg = ("[v17 S12] EMA: FastEMAHelper - two fused kernels per step instead of ~4 per "
           "parameter tensor. Not bit-identical to the frozen helper (~4e-7 over a run); "
           "pass --fast_loop off to byte-compare against an existing run.")
    (logger.info if logger is not None else lambda m: print(m, flush=True))(msg)
    return fast
