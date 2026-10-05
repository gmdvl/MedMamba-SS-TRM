# -*- coding: utf-8 -*-
"""
training/trainerg_v13.py
==========================
`TrainerG_v13(TrainerG_v12)` - reconstruction artifacts that survive the run.

`TrainerG_v12` is NOT on the frozen list, but it produced every v16 run in
`experiments/`, so it is treated the same way: subclassed, never edited. The
subclass adds three things and changes nothing that affects a loss, a
gradient, a checkpoint or a reported classification metric.

  1. `export_reconstruction_samples` - a two-pass exporter that writes the
     reconstructed CUBES to disk (`reconstruction/samples.npz`) plus the
     figures, from an explicitly named checkpoint. v12 kept only spatially
     averaged spectra and rendered them from whatever epoch training
     happened to stop on (`trainerg_v12.py:898`/`:943`), which is not the
     checkpoint that gets shipped.
       Pass 1 ranks every validation sample by SAM (cheap: mean-pooled
       spectra, no cube retained). Pass 2 re-visits only the batches holding
       the selected indices and keeps those cubes. Two forward passes over
       the validation split, no memory proportional to the split.
  2. `reconstruction/metrics_curves.png` - per-epoch SAM / RMSE / PSNR /
     spectral-SSIM, which the run already recorded in `history.json` and
     never plotted.
  3. `reconstruction/status.json` - written on EVERY path, including the
     skip paths. v12's artifact block is a bare `try/except` that logs a
     warning (`trainerg_v12.py:967-974`); four of the eight runs that
     trained a decoder have an empty `reconstruction/` directory and no
     record of why.

Nothing here runs when the model has no decoder (`--recon_mode none`): the
status file says so and the run is otherwise byte-identical to a
`TrainerG_v12` run.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.amp import autocast

from training.trainerg_v12 import TrainerG_v12
from training.normalization import denormalize
from training import recon_artifacts as ra


# Set by `train_example_v16_recon.py` before `_main()` runs, so the run's own
# `config.json` names the script that actually launched it. `_main()` writes
# `"entry_point": "train_example_v16.py"` unconditionally and lives in a file
# this module may not edit.
ENTRY_POINT_OVERRIDE: str | None = None


def _as_bool(value, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "on", "yes")


class TrainerG_v13(TrainerG_v12):

    def __init__(self, *args,
                 recon_save_samples: Optional[bool] = None,
                 recon_sample_per_class: Optional[int] = None,
                 recon_rgb_bands: Optional[str] = None,
                 recon_render_png: Optional[bool] = None,
                 recon_max_cubes: Optional[int] = None,
                 record_config: bool = True,
                 **kwargs):
        super().__init__(*args, **kwargs)
        cli = (self.config or {}).get("cli_args", {}) or {}

        self.recon_save_samples = _as_bool(
            recon_save_samples if recon_save_samples is not None else cli.get("recon_save_samples"), True)
        self.recon_sample_per_class = int(
            recon_sample_per_class if recon_sample_per_class is not None
            else cli.get("recon_sample_per_class", 3) or 3)
        self.recon_rgb_bands = recon_rgb_bands if recon_rgb_bands is not None else cli.get("recon_rgb_bands")
        self.recon_render_png = _as_bool(
            recon_render_png if recon_render_png is not None else cli.get("recon_render_png"), True)
        self.recon_max_cubes = int(
            recon_max_cubes if recon_max_cubes is not None else cli.get("recon_max_cubes", 64) or 64)

        # The run's own record must name the class that actually ran it -
        # unless this trainer is a read-only harness over somebody else's
        # finished run (`record_config=False`, used by
        # `scripts/export_recon_samples.py`), where relabelling a v12
        # run as v13 would falsify its provenance.
        if self.config is not None and record_config:
            # v17 S12 - the ACTUAL class, not the literal. This line used to write
            # "TrainerG_v13" over whatever `train_example_v16._main` had recorded, so a
            # `TrainerG_v13Optimal` run claimed to be a plain v13 one. With `--fast_loop`
            # now able to change the class too, a literal here would hide that as well.
            # Subclasses are exactly what this file is designed to be composed into, so
            # naming the composition is the whole provenance value.
            self.config["trainer"] = type(self).__name__
            if ENTRY_POINT_OVERRIDE:
                self.config["entry_point"] = ENTRY_POINT_OVERRIDE
            self.config["recon_artifacts"] = {
                "save_samples": self.recon_save_samples,
                "sample_per_class": self.recon_sample_per_class,
                "rgb_bands": self.recon_rgb_bands,
                "render_png": self.recon_render_png,
                "max_cubes": self.recon_max_cubes,
            }
            with open(self.exp_dir / "config.json", "w") as f:
                json.dump(self.config, f, indent=2, default=str)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def has_decoder(self) -> bool:
        """True when a reconstruction decoder is attached - i.e. the run was
        launched with `--recon_mode latent` or `raw_input`.

        `latent` puts a `decoder` on the wrapper; `raw_input`
        (`MedMambaSSTRMRawReconWrapper`) does not, and is recognised by the
        recorded `--recon_mode` instead. The attribute check comes first so
        a trainer built without a config still works."""
        if getattr(self.model, "decoder", None) is not None:
            return True
        mode = (self.config or {}).get("cli_args", {}).get("recon_mode")
        return mode is not None and mode != "none"

    def _load_state_for_export(self, checkpoint_path: Path) -> Dict:
        state = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        model_state = state.get("model_state", state)
        self.model.load_state_dict(model_state)
        return {"epoch": state.get("epoch"), "best_epoch": state.get("best_epoch"),
                "best_val_acc": state.get("best_val_acc")}

    def _unpack_batch(self, batch):
        if len(batch) == 3:
            x, y, ns = batch
            return x, y, ns.to(self.device, non_blocking=True)
        return batch[0], batch[1], None

    def _to_device(self, x):
        x = x.to(self.device, non_blocking=True)
        if x.ndim == 4 and self.device_type == "cuda":
            x = x.to(memory_format=torch.channels_last)
        return x

    def _wl_tensor(self, device):
        if self._wl is not None:
            return self._wl
        if self.wavelengths is not None:
            return torch.as_tensor(self.wavelengths, device=device)
        return None

    # ------------------------------------------------------------------
    # the exporter
    # ------------------------------------------------------------------

    def export_reconstruction_samples(self, loader=None, split: str = "validation",
                                       checkpoint_path=None, out_dir=None) -> Dict:
        """Writes the reconstructed cubes and figures. Returns the status
        dict it also writes to `<out_dir>/status.json`."""
        out_dir = Path(out_dir) if out_dir is not None else self.dirs["reconstruction"]
        started = time.time()

        def _fail(reason: str, **extra) -> Dict:
            status = {"written": False, "split": split, "reason": reason,
                      "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), **extra}
            ra.write_status(str(out_dir), **status)
            self.logger.warning(f"[recon-artifacts] not written: {reason}")
            return status

        if not self.recon_save_samples:
            return _fail("disabled by --recon_save_samples off")
        if not self.has_decoder():
            return _fail("no reconstruction decoder attached (--recon_mode none)")

        loader = loader if loader is not None else self.val_loader
        if loader is None:
            return _fail(f"no {split} loader available")

        ckpt = Path(checkpoint_path) if checkpoint_path else (self.exp_dir / "best_model.pt")
        if not ckpt.is_file():
            return _fail(f"checkpoint {ckpt} does not exist")

        saved_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        try:
            ckpt_info = self._load_state_for_export(ckpt)
            self.model.eval()
            amp_enabled, amp_dtype = self._resolve_val_amp()
            wl_np = np.asarray(self.wavelengths, dtype=np.float32) if self.wavelengths is not None else None

            # ---------- pass 1: rank every sample by SAM ----------
            sam_all: List[np.ndarray] = []
            labels_all: List[np.ndarray] = []
            saw_norm_stats = False
            with torch.no_grad():
                for batch in loader:
                    x, y, ns = self._unpack_batch(batch)
                    x = self._to_device(x)
                    with autocast(device_type=self.device_type, dtype=amp_dtype, enabled=amp_enabled):
                        _logits, _q, x_recon = self._forward_with_recon(self.model, x)
                    if x_recon is None:
                        return _fail("forward returned no reconstruction "
                                     "(decoder present but decode path inactive)")
                    if ns is not None:
                        saw_norm_stats = True
                        x_r, recon_r = denormalize(x.float(), ns), denormalize(x_recon.float(), ns)
                        require_nonneg = True
                    else:
                        x_r, recon_r, require_nonneg = x.float(), x_recon.float(), False
                    sam_all.append(ra.batch_sam_deg(x_r, recon_r,
                                                     require_nonnegative=require_nonneg).numpy())
                    labels_all.append(y.numpy() if hasattr(y, "numpy") else np.asarray(y))

            if not sam_all:
                return _fail(f"{split} loader yielded no batches")
            sam = np.concatenate(sam_all)
            labels = np.concatenate(labels_all)

            if not saw_norm_stats:
                self.logger.warning(
                    "[recon-artifacts] the loader returned no norm_stats, so the cubes are saved in "
                    "NORMALIZED units, not reflectance. Build the dataset with "
                    "return_norm_stats=True to fix this.")

            per_class = self.recon_sample_per_class
            num_classes = int(max(len(self.class_names), int(labels.max()) + 1))
            wanted, tags = ra.select_quantile_indices(sam, labels, per_class, num_classes)
            if wanted.size == 0:
                return _fail("no samples selected (per_class < 1 or empty split)")
            if wanted.size > self.recon_max_cubes:
                wanted, tags = wanted[:self.recon_max_cubes], tags[:self.recon_max_cubes]
            wanted_set = set(int(i) for i in wanted)

            # ---------- pass 2: keep only the selected cubes ----------
            keep_true: Dict[int, np.ndarray] = {}
            keep_recon: Dict[int, np.ndarray] = {}
            keep_label: Dict[int, int] = {}
            cursor = 0
            with torch.no_grad():
                for batch in loader:
                    x, y, ns = self._unpack_batch(batch)
                    bs = int(x.shape[0])
                    local = [i for i in range(bs) if (cursor + i) in wanted_set]
                    if not local:
                        cursor += bs
                        continue
                    x = self._to_device(x)
                    with autocast(device_type=self.device_type, dtype=amp_dtype, enabled=amp_enabled):
                        _logits, _q, x_recon = self._forward_with_recon(self.model, x)
                    if ns is not None:
                        x_r, recon_r = denormalize(x.float(), ns), denormalize(x_recon.float(), ns)
                    else:
                        x_r, recon_r = x.float(), x_recon.float()
                    for i in local:
                        g = cursor + i
                        keep_true[g] = x_r[i].detach().float().cpu().numpy()
                        keep_recon[g] = recon_r[i].detach().float().cpu().numpy()
                        keep_label[g] = int(y[i])
                    cursor += bs
                    if len(keep_true) == len(wanted_set):
                        break

            missing = sorted(wanted_set - set(keep_true))
            order = [int(i) for i in wanted if int(i) in keep_true]
            tags_kept = [t for t, i in zip(tags, wanted) if int(i) in keep_true]
            if not order:
                return _fail("second pass retained no cubes", missing_indices=missing[:20])

            x_true = np.stack([keep_true[i] for i in order])
            x_recon_arr = np.stack([keep_recon[i] for i in order])
            lab = np.asarray([keep_label[i] for i in order], dtype=np.int64)

            metrics = ra.per_sample_full_cube_metrics(
                torch.from_numpy(x_true), torch.from_numpy(x_recon_arr),
                wavelengths=(torch.from_numpy(wl_np) if wl_np is not None else None),
                require_nonnegative=saw_norm_stats)

            provenance = {
                "run_dir": str(self.exp_dir),
                "split": split,
                "checkpoint": str(ckpt),
                "checkpoint_epoch": ckpt_info.get("epoch"),
                "checkpoint_best_epoch": ckpt_info.get("best_epoch"),
                "checkpoint_metric": getattr(self, "checkpoint_metric", None),
                "selection_rule": (f"{per_class} samples per class at evenly spaced SAM quantiles "
                                   f"(q=0 best ... q=1 worst), ranked on spatially mean-pooled "
                                   f"spectra in reflectance units"),
                "n_split_samples": int(sam.size),
                "denormalized": bool(saw_norm_stats),
                "normalization": (self.config or {}).get("cli_args", {}).get("normalization"),
                "recon_mode": (self.config or {}).get("cli_args", {}).get("recon_mode"),
                "amp": (self.config or {}).get("cli_args", {}).get("amp"),
                "split_sam_deg_mean": float(np.nanmean(sam)),
                "split_sam_deg_median": float(np.nanmedian(sam)),
                "missing_indices": missing,
                "trainer": "TrainerG_v13",
            }

            result = ra.save_reconstruction_samples(
                str(out_dir), x_true, x_recon_arr, lab, np.asarray(order, dtype=np.int64),
                tags_kept, metrics, wl_np, list(self.class_names), provenance,
                band_spec=self.recon_rgb_bands, render_png=self.recon_render_png)

            status = {
                "written": True, "split": split, "n_samples": int(x_true.shape[0]),
                "checkpoint": str(ckpt), "denormalized": bool(saw_norm_stats),
                "npz": result["npz"], "figures": len(result["figures"]),
                "rgb_bands": result["bands"], "rgb_band_note": result["band_note"],
                "elapsed_s": round(time.time() - started, 2),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            ra.write_status(str(out_dir), **status)
            self.logger.info(f"[recon-artifacts] wrote {status['n_samples']} cubes + "
                              f"{status['figures']} figures to {out_dir}")
            return status
        except Exception as e:
            import traceback
            return _fail(f"{type(e).__name__}: {e}", traceback=traceback.format_exc())
        finally:
            try:
                self.model.load_state_dict(saved_state)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # hooks
    # ------------------------------------------------------------------

    def _generate_rich_diagnostics(self):
        """v12's diagnostics, then the per-epoch reconstruction curves and
        the cube export. Deliberately AFTER `super()` so a failure here can
        never cost the run its confusion matrices or latent-space plots."""
        super()._generate_rich_diagnostics()

        try:
            wrote = ra.plot_recon_metric_curves(
                self.history, str(self.dirs["reconstruction"] / "metrics_curves.png"))
            if wrote:
                self.logger.info("[recon-artifacts] wrote reconstruction/metrics_curves.png")
        except Exception as e:
            self.logger.warning(f"[recon-artifacts] metric curves failed: {type(e).__name__}: {e}")

        self.export_reconstruction_samples(split="validation")
