# -*- coding: utf-8 -*-
"""
training/trainerg_v6.py
========================
Stage B ("Classification") on top of `training/trainerg_v5.py`
(TrainerG_v5): makes the classification loss pluggable
(`training.losses.build_criterion` - CE / weighted CE / focal / focal+weighted)
instead of `F.cross_entropy` being hardcoded in both the train and
validation loops. Everything else (macro-F1 checkpoint selection, class-
collapse monitoring, gradient-health tracking) is inherited unchanged from
TrainerG_v5.

`self.criterion` defaults to plain `nn.CrossEntropyLoss()` if not given, so
this is a drop-in replacement for TrainerG_v5 when no experiment is running.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast

from training.trainerg_v5 import TrainerG_v5
from training.eval_utils import compute_classification_metrics
from training.gan import discriminator_loss, generator_adversarial_loss
from training.spectral_recon_metrics import compute_all_spectral_metrics, compute_per_sample_spectral_metrics


class TrainerG_v6(TrainerG_v5):
    def __init__(self, *args, criterion: Optional[nn.Module] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.criterion = criterion if criterion is not None else nn.CrossEntropyLoss()
        if hasattr(self.criterion, "to"):
            self.criterion = self.criterion.to(self.device)

    # ------------------------------------------------------------------
    def _train_one_epoch(self):
        self.grad_health.start_epoch()
        self.model.train()
        total_loss = total_cls = total_mse = total_sam = total_gan = 0.0
        total_correct = total_n = 0
        total_grad_norm = 0.0
        n_batches = 0

        use_bf16 = self.device_type == "cuda" and torch.cuda.is_bf16_supported()
        amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
        amp_enabled = self.device_type == "cuda"

        for x, y in self.train_loader:
            x = x.to(self.device, non_blocking=True)
            if x.ndim == 4 and self.device_type == "cuda":
                x = x.to(memory_format=torch.channels_last)
            y = y.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=self.device_type, dtype=amp_dtype, enabled=amp_enabled):
                out = self.model(x)
                if isinstance(out, tuple):
                    logits, x_recon = out
                else:
                    logits, x_recon = out, None

                cls_loss = self.criterion(logits, y)
                if x_recon is not None:
                    mse_loss = F.mse_loss(x_recon, x)
                    sam_loss = self.sam_loss_fn(x, x_recon)
                else:
                    mse_loss = torch.tensor(0.0, device=self.device)
                    sam_loss = torch.tensor(0.0, device=self.device)

                gan_loss = torch.tensor(0.0, device=self.device)
                if self.use_gan and x_recon is not None:
                    gan_loss = generator_adversarial_loss(self.discriminator, x_recon)

                loss = (cls_loss + self.lambda_mse * mse_loss + self.lambda_sam * sam_loss
                        + self.lambda_gan * gan_loss)

            loss_finite = bool(torch.isfinite(loss))
            if not loss_finite:
                self.grad_health.record_batch(loss_is_finite=False, grad_is_finite=False, optimizer_stepped=False)
                self.optimizer.zero_grad(set_to_none=True)
                continue

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.grad_clip_norm if self.grad_clip_norm else float("inf"))
            grad_norm = float(grad_norm)
            grad_finite = bool(np.isfinite(grad_norm))

            if not grad_finite:
                self.grad_health.record_batch(loss_is_finite=True, grad_is_finite=False, optimizer_stepped=False)
                self.logger.warning("Non-finite gradient norm - skipping this batch's optimizer step.")
                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.update()
                continue

            total_grad_norm += grad_norm
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.grad_health.record_batch(loss_is_finite=True, grad_is_finite=True, optimizer_stepped=True)

            if self.use_gan and x_recon is not None:
                self.disc_optimizer.zero_grad(set_to_none=True)
                with autocast(device_type=self.device_type, dtype=amp_dtype, enabled=amp_enabled):
                    d_loss = discriminator_loss(self.discriminator, x, x_recon.detach())
                if not (torch.isnan(d_loss) or torch.isinf(d_loss)):
                    d_loss.backward()
                    self.disc_optimizer.step()

            total_loss += loss.item() * x.size(0)
            total_cls += cls_loss.item() * x.size(0)
            total_mse += mse_loss.item() * x.size(0)
            total_sam += sam_loss.item() * x.size(0)
            total_gan += gan_loss.item() * x.size(0)
            total_correct += (logits.argmax(dim=-1) == y).sum().item()
            total_n += x.size(0)
            n_batches += 1

        epoch_health = self.grad_health.finish_epoch()
        denom = max(1, total_n)
        return {
            "loss": total_loss / denom, "cls_loss": total_cls / denom,
            "mse_loss": total_mse / denom, "sam_loss": total_sam / denom,
            "gan_loss": total_gan / denom, "acc": total_correct / denom,
            "grad_norm": total_grad_norm / max(1, n_batches),
            "gradient_health": epoch_health,
        }

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _validate_one_epoch(self):
        self.model.eval()
        total_loss = total_cls = 0.0
        total_n = 0
        spectral_accum = {}
        spectral_batches = 0
        all_preds, all_labels, all_probs = [], [], []
        per_sample_metrics_accum = {}
        recon_true_means, recon_pred_means = [], []
        recon_collected = 0

        use_bf16 = self.device_type == "cuda" and torch.cuda.is_bf16_supported()
        amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
        amp_enabled = self.device_type == "cuda"

        for x, y in self.val_loader:
            x = x.to(self.device, non_blocking=True)
            if x.ndim == 4 and self.device_type == "cuda":
                x = x.to(memory_format=torch.channels_last)
            y = y.to(self.device, non_blocking=True)

            with autocast(device_type=self.device_type, dtype=amp_dtype, enabled=amp_enabled):
                out = self.model(x)
                if isinstance(out, tuple):
                    logits, x_recon = out
                else:
                    logits, x_recon = out, None
                cls_loss = self.criterion(logits, y)
                if x_recon is not None:
                    mse_loss = F.mse_loss(x_recon, x)
                else:
                    mse_loss = torch.tensor(0.0, device=self.device)
                loss = cls_loss + self.lambda_mse * mse_loss

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            total_loss += loss.item() * x.size(0)
            total_cls += cls_loss.item() * x.size(0)
            total_n += x.size(0)

            probs = F.softmax(logits.float(), dim=-1)
            all_probs.append(probs.cpu().numpy())
            all_preds.append(probs.argmax(dim=-1).cpu().numpy())
            all_labels.append(y.cpu().numpy())

            if x_recon is not None:
                wl_tensor = torch.as_tensor(self.wavelengths, device=x.device) if self.wavelengths is not None else None
                batch_spectral = compute_all_spectral_metrics(x, x_recon, wavelengths=wl_tensor)
                for k, v in batch_spectral.items():
                    spectral_accum[k] = spectral_accum.get(k, 0.0) + v * x.size(0)
                spectral_batches += x.size(0)

                if recon_collected < self.recon_sample_cap:
                    per_sample = compute_per_sample_spectral_metrics(x, x_recon, wavelengths=wl_tensor)
                    for k, v in per_sample.items():
                        per_sample_metrics_accum.setdefault(k, []).extend(v)
                    x_mean = x.float().mean(dim=(2, 3)).cpu().numpy()
                    xr_mean = x_recon.float().mean(dim=(2, 3)).cpu().numpy()
                    remaining = self.recon_sample_cap - recon_collected
                    recon_true_means.append(x_mean[:remaining])
                    recon_pred_means.append(xr_mean[:remaining])
                    recon_collected += min(x.size(0), remaining)

        preds = np.concatenate(all_preds) if all_preds else np.array([], dtype=np.int64)
        labels = np.concatenate(all_labels) if all_labels else np.array([], dtype=np.int64)
        probs = np.concatenate(all_probs) if all_probs else np.zeros((0, 0))

        # v15 R0.4f - pass the full label set explicitly. Without `labels=`,
        # sklearn infers the classes from the union of y_true and y_pred, so a
        # class with zero validation support (or one that is both absent from
        # the labels and never predicted) silently drops OUT of the macro
        # average, making macro-F1 an average over fewer classes than the model
        # has. Latent on these two datasets, which have every class in every
        # split - but it is exactly the kind of thing that turns a collapse
        # into a flattering number. `probs.shape[1]` is the model's own output
        # width, so this needs no extra plumbing through the trainer.
        label_set = list(range(probs.shape[1])) if probs.size else None
        classification = (compute_classification_metrics(labels, preds, labels=label_set)
                           if len(labels) > 0
                           else {"accuracy": 0.0, "balanced_accuracy": 0.0, "precision_macro": 0.0,
                                 "recall_macro": 0.0, "f1_macro": 0.0, "precision_weighted": 0.0,
                                 "recall_weighted": 0.0, "f1_weighted": 0.0})

        spectral = ({k: v / max(1, spectral_batches) for k, v in spectral_accum.items()}
                    if spectral_batches > 0 else {})

        denom = max(1, total_n)
        return {
            "loss": total_loss / denom, "cls_loss": total_cls / denom,
            "classification": classification, "spectral": spectral,
            "preds": preds, "labels": labels, "probs": probs,
            "recon_per_sample_metrics": per_sample_metrics_accum,
            "recon_true_means": np.concatenate(recon_true_means, axis=0) if recon_true_means else None,
            "recon_pred_means": np.concatenate(recon_pred_means, axis=0) if recon_pred_means else None,
        }
