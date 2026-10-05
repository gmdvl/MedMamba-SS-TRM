# -*- coding: utf-8 -*-
"""
training/trainerg.py
=====================
`TrainerG` - the one trainer every `train.py` profile runs.

It is `train_example_v18.TrainerG_v18` with its composition flattened. That class
was five classes deep, one per entry point, and each profile got a different
subset of them:

    TrainerG_v18          FLOPs recount, latent-space fallback        (paper_recipe)
    TrainerG_v13Optimal   (empty - the composition)                   (train_example_v16_optimal_recon.py)
    TrainerG_v12Optimal   3-tuple training batches, stop message      (pad_ufes_best_norecon)
    TrainerG_v12Original  patience rule, gate-G5 metric alias         (medmamba_protocol_norecon)
    TrainerG_v13          reconstruction artifacts                    (train_example_v16_recon.py)
    TrainerG_v12          everything else                             (pipeline_defaults)

The method bodies below are those classes' bodies, unchanged, in the same MRO
order, so `super()` reaches the same implementation it did. What differed
between profiles was never the methods - every one of them is either inert or
strictly additive outside its own profile - but two CLASS attributes the entry
points mutated at startup. They are now constructor arguments:

    early_stopping_enabled   --early_stopping (and a patience > 0)
    suppressed_stop_reasons  --stop_on_metric_stall off -> {VAL_ACC_STALLED}

`training.trainerg_v12_fast.make_fast(TrainerG)` puts the de-synced epoch in
front, exactly as it did for the old classes.
"""

from __future__ import annotations

import json

from training.trainerg_v4 import StopReason
from training.trainerg_v7 import _StopReasonStr
from training.trainerg_v13 import TrainerG_v13


class _TwoTupleLoader:
    """A DataLoader view yielding `(x, y)` from `(x, y, norm_stats)` batches.

    Any reconstruction mode makes the datasets emit 3-tuples. The recursive
    training loop consumes them; the non-recursive fallback (`TrainerG_v9`'s
    `for x, y in self.train_loader`) cannot, so `--architecture split
    --recon_mode latent` died unpacking the first batch. Wrapped for the
    training epoch only - validation keeps the stats and reports raw-unit
    reconstruction error.
    """

    def __init__(self, loader):
        self._loader = loader

    def __iter__(self):
        for batch in self._loader:
            yield (batch[0], batch[1]) if len(batch) > 2 else batch

    def __len__(self):
        return len(self._loader)

    def __getattr__(self, name):
        return getattr(self._loader, name)


class TrainerG(TrainerG_v13):

    # gate G5 looks the checkpoint metric up in metrics["classification"], where
    # val_accuracy is still called "accuracy".
    _CHECKPOINT_METRIC_ALIASES = {"val_accuracy": "accuracy"}

    _warned_norm_stats = False

    def __init__(self, *args, early_stopping_enabled: bool = False,
                 suppressed_stop_reasons: frozenset = frozenset(), **kwargs):
        super().__init__(*args, **kwargs)
        self.early_stopping_enabled = bool(early_stopping_enabled)
        self.suppressed_stop_reasons = frozenset(suppressed_stop_reasons)
        if not self.early_stopping_enabled:
            # Only the patience rule below and TrainerG_v12's progress line read this. A
            # patience that is not enforced must not print "early-stop 3/40" or cap the ETA,
            # which it did on every optimal-family run. --early_stop_patience stays in cli_args.
            self.early_stop_patience = None

    # ------------------------------------------------------------------
    # was TrainerG_v18 - FLOPs that count, a latent space that survives recon
    # ------------------------------------------------------------------

    def evaluate_test_split(self, test_loader, *a, **kw):
        """The inherited call passes a 3-D shape to `count_flops`, so every
        report has `"flops": {"error": ...}`; recount then, and only then."""
        report = super().evaluate_test_split(test_loader, *a, **kw)
        if report is None:
            return report
        flops = report.get("flops")
        if not (isinstance(flops, dict) and flops.get("error")):
            return report
        try:
            report["flops"] = self._recount_flops(test_loader)
            with open(self.exp_dir / "test_report.json", "w") as f:
                json.dump(report, f, indent=2, default=str)
            self.logger.info(
                f"[flops-v18] recounted at {report['flops']['input_shape']}: "
                f"{report['flops'].get('total_gflops_approx')} GFLOPs, core_applications="
                f"{report['flops'].get('core_applications')} "
                f"(the inherited call passed a 3-D shape; see training/flops_report.py)")
        except Exception as e:
            self.logger.warning(f"[flops-v18] recount failed: {type(e).__name__}: {e}")
        return report

    def _recount_flops(self, test_loader) -> dict:
        from training.flops_report import build_flops_report
        sample_x = next(iter(test_loader))[0]
        _, in_channels, height, width = sample_x.shape
        cli_args = (self.config or {}).get("cli_args", {}) or {}
        return build_flops_report(
            self.model, in_channels=in_channels, patch_hw=(height, width),
            cli_args=cli_args, device=str(self.device),
            wavelengths=self.wavelengths, sensor_range=self.sensor_range,
            backbone_num_params=(self.config or {}).get("backbone_num_params"))

    def _generate_rich_diagnostics(self):
        """The inherited latent-space extractor unpacks 2-tuples, so it writes
        nothing under any reconstruction mode; fill the directory only then."""
        super()._generate_rich_diagnostics()
        out_dir = self.dirs["latent_space"]
        try:
            if any(out_dir.iterdir()):
                return
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

    # ------------------------------------------------------------------
    # was TrainerG_v12Optimal
    # ------------------------------------------------------------------

    def _train_one_epoch(self):
        if self._deep_supervision_base() is not None:
            return super()._train_one_epoch()          # recursive: v12 unpacks 3-tuples itself
        real_loader = self.train_loader
        if not TrainerG._warned_norm_stats:
            TrainerG._warned_norm_stats = True
            self.logger.info(
                "[recon] architecture is not 'recursive', so training runs TrainerG_v9's loop, "
                "which takes (x, y) batches only. The norm_stats element the reconstruction modes "
                "add is dropped for the TRAINING epoch: train MSE/SAM are measured in NORMALIZED "
                "units. Validation is unaffected - it keeps the stats and reports raw-unit "
                "reconstruction error.")
        self.train_loader = _TwoTupleLoader(real_loader)
        try:
            return super()._train_one_epoch()
        finally:
            self.train_loader = real_loader

    def _trigger_stop(self, reason):
        """The single funnel every stopping rule goes through
        (`training/trainerg_v4.py:647`): filtering here neutralises one rule
        without copying `_check_stopping_rules`."""
        if reason in self.suppressed_stop_reasons:
            self.logger.info(
                f"[stopping rule] {getattr(reason, 'value', reason)} suppressed: the checkpoint "
                f"metric repeating EXACTLY for 5 epochs is what a one-class predictor does before "
                f"the class weighting takes hold, not evidence of convergence. Re-enable with "
                f"--stop_on_metric_stall on. Training continues.")
            return
        return super()._trigger_stop(reason)

    # ------------------------------------------------------------------
    # was TrainerG_v12Original
    # ------------------------------------------------------------------

    def _check_stopping_rules(self):
        """v12's rules, plus `TrainerG_v7`'s patience rule when enabled.

        `TrainerG_v12._check_stopping_rules` does not call `super()` and sits
        ahead of `TrainerG_v7` in the MRO, so v7's patience block is unreachable
        for every v12 run. The counter logic mirrors trainerg_v7.py:52-81 rather
        than calling it, because v7's method also runs the v4/v6 heuristics v12
        replaced.
        """
        super()._check_stopping_rules()
        if not self.early_stopping_enabled or self.early_stop_patience is None or self.is_stopped:
            return
        if not self.history:
            return
        last = self.history[-1]
        if not last.get("is_valid_epoch", True):
            return                      # an invalid epoch is not a failed attempt
        candidate = last.get(self.checkpoint_metric)
        if candidate is None:
            return
        # `fit` calls this BEFORE updating best_metric_value: "did THIS epoch beat the prior best".
        if candidate > self.best_metric_value + 1e-9:
            self._epochs_without_improvement = 0
        else:
            self._epochs_without_improvement += 1
        if self._epochs_without_improvement >= self.early_stop_patience:
            self.stop_reason = _StopReasonStr(
                f"EARLY_STOPPED_PATIENCE_{self.checkpoint_metric.upper()}")
            self.is_stopped = True
            self.logger.info(
                f"Early stopping: no improvement in {self.checkpoint_metric!r} for "
                f"{self.early_stop_patience} epochs (best={self.best_metric_value:.4f}).")

    def verify_best_checkpoint_reproduces(self, *a, **kw):
        alias = self._CHECKPOINT_METRIC_ALIASES.get(self.checkpoint_metric)
        if alias is None:
            return super().verify_best_checkpoint_reproduces(*a, **kw)
        reported = self.checkpoint_metric
        self.checkpoint_metric = alias
        try:
            out = super().verify_best_checkpoint_reproduces(*a, **kw)
        finally:
            self.checkpoint_metric = reported
        out["metric"] = reported
        out["metric_looked_up_as"] = alias
        try:
            with open(self.exp_dir / "checkpoint_reproducibility.json", "w") as f:
                json.dump(out, f, indent=2, default=str)
        except OSError:
            pass
        return out


def stop_knobs(args) -> dict:
    """The two constructor arguments above, from the parsed command line."""
    return {
        "early_stopping_enabled": (getattr(args, "early_stopping", "off") == "on"
                                   and bool(getattr(args, "early_stop_patience", None))),
        "suppressed_stop_reasons": (frozenset() if getattr(args, "stop_on_metric_stall", "on") == "on"
                                    else frozenset({StopReason.VAL_ACC_STALLED})),
    }
