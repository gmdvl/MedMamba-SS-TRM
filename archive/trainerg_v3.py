# -*- coding: utf-8 -*-
"""
training/trainer.py
===================
TrainerG_v1 engine with integrated Spectral Reconstruction evaluation,
Spectral Angle Mapper (SAM) calculation, and joint multi-task loss tracking.
"""

import os
import sys
import time
import json
import csv
import logging
from enum import Enum
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.amp import autocast, GradScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, balanced_accuracy_score
)


class StopReason(str, Enum):
    COMPLETED = "TRAINING_COMPLETED"
    VAL_DIVERGENCE = "VAL_DIVERGENCE"
    TRAIN_DIVERGENCE = "TRAIN_DIVERGENCE"
    LOSS_NAN = "LOSS_NAN"
    LOSS_INF = "LOSS_INF"
    GRAD_EXPLOSION = "GRADIENT_EXPLOSION"
    VAL_ACC_STALLED = "VAL_ACCURACY_STALLED"
    VAL_LOSS_STALLED = "VAL_LOSS_STALLED"
    TRAIN_ACC_STALLED = "TRAIN_ACCURACY_STALLED"
    TRAIN_LOSS_STALLED = "TRAIN_LOSS_STALLED"
    USER_INTERRUPTED = "USER_INTERRUPTED"


def compute_sam(x_true: torch.Tensor, x_pred: torch.Tensor, eps: float = 1e-8) -> float:
    """Computes Spectral Angle Mapper (SAM) in degrees across spectral channels."""
    x_true = x_true.float()
    x_pred = x_pred.float()
    b, c, h, w = x_true.shape
    
    true_vec = x_true.permute(0, 2, 3, 1).reshape(-1, c)
    pred_vec = x_pred.permute(0, 2, 3, 1).reshape(-1, c)

    dot_product = torch.sum(true_vec * pred_vec, dim=-1)
    norm_true = torch.norm(true_vec, p=2, dim=-1)
    norm_pred = torch.norm(pred_vec, p=2, dim=-1)

    cos_sim = dot_product / (norm_true * norm_pred + eps)
    cos_sim = torch.clamp(cos_sim, -1.0 + eps, 1.0 - eps)

    sam_rad = torch.acos(cos_sim)
    sam_deg = torch.rad2deg(sam_rad)
    return float(sam_deg.mean().item())


def compute_spectral_rmse(x_true: torch.Tensor, x_pred: torch.Tensor) -> float:
    """Computes Root Mean Square Error across reconstructed spectral bands."""
    mse = F.mse_loss(x_pred.float(), x_true.float())
    return float(torch.sqrt(mse).item())


class TrainerG_v3:
    def __init__(self, model, train_loader, val_loader, optimizer, scheduler, device, exp_dir, recon_weight=1.0, config=None):
        self.device = torch.device(device)
        self.device_type = "cuda" if "cuda" in str(self.device) else "cpu"
        self.recon_weight = recon_weight

        self.model = model.to(self.device)
        if self.device_type == "cuda":
            self.model = self.model.to(memory_format=torch.channels_last)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.exp_dir = Path(exp_dir)
        self.config = config or {}
        
        # Setup stdout logger
        self.logger = logging.getLogger("TrainerG_v1")
        self.logger.setLevel(logging.INFO)
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
            
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(stdout_handler)
        self.logger.propagate = False

        # Precision & state tracking
        self.scaler = GradScaler(device=self.device_type)
        self.current_epoch = 0
        self.best_epoch = 0
        self.best_val_acc = -1.0
        self.history = []
        self.stall_counters = {}
        self.stop_reason = None
        self.is_stopped = False
        
        # Directories
        self.dirs = {
            "checkpoints": self.exp_dir / "checkpoints",
            "reports": self.exp_dir / "reports"
        }
        for d in self.dirs.values():
            d.mkdir(parents=True, exist_ok=True)

    def _validate_metric(self, val, name):
        if val is None or np.isnan(val) or np.isinf(val):
            raise ValueError(f"Invalid metric value encountered for {name}: {val}")
        return float(val)

    def fit(self, max_epochs, start_epoch=1):
        self.current_epoch = start_epoch - 1
        self.logger.info(f"Starting execution loop from epoch {start_epoch} to {max_epochs}...")
        
        try:
            for epoch in range(start_epoch, max_epochs + 1):
                if self.is_stopped:
                    break
                    
                self.current_epoch = epoch
                epoch_start_time = time.time()
                
                # 1. Training Phase
                train_metrics = self._train_one_epoch()
                if self.is_stopped:
                    break
                    
                # 2. Validation Phase
                val_metrics = self._validate_one_epoch()
                
                epoch_time = time.time() - epoch_start_time
                current_lr = self.optimizer.param_groups[0]['lr']
                self.scheduler.step()

                # Consolidate Metrics (including Spectral Reconstruction)
                metrics = {
                    "epoch": epoch,
                    "train_loss": train_metrics["loss"],
                    "train_cls_loss": train_metrics["cls_loss"],
                    "train_recon_loss": train_metrics["recon_loss"],
                    "train_accuracy": train_metrics["acc"],
                    "val_loss": val_metrics["loss"],
                    "val_cls_loss": val_metrics["cls_loss"],
                    "val_accuracy": val_metrics["acc"],
                    "spectral_recon_loss": val_metrics["spectral_recon_loss"],
                    "spectral_sam_deg": val_metrics["spectral_sam_deg"],
                    "spectral_rmse": val_metrics["spectral_rmse"],
                    "precision": val_metrics["precision"],
                    "recall": val_metrics["recall"],
                    "f1": val_metrics["f1"],
                    "macro_f1": val_metrics["macro_f1"],
                    "balanced_accuracy": val_metrics["balanced_acc"],
                    "learning_rate": current_lr,
                    "epoch_time": epoch_time
                }
                
                self.history.append(metrics)
                self._check_stopping_rules()
                
                is_best = val_metrics["acc"] > self.best_val_acc
                if is_best:
                    self.best_val_acc = val_metrics["acc"]
                    self.best_epoch = epoch
                    
                self._save_checkpoint(is_best)
                self._save_epoch_report(metrics, is_best)
                self._update_history_csv(metrics)
                self._log_epoch(metrics, is_best)
                
                if epoch == max_epochs and not self.is_stopped:
                    self.stop_reason = StopReason.COMPLETED
                    
        except KeyboardInterrupt:
            self.stop_reason = StopReason.USER_INTERRUPTED
            self.logger.warning("Training process interrupted manually by user.")
        finally:
            if self.stop_reason is None:
                self.stop_reason = StopReason.COMPLETED
            self._generate_final_report()

    def _train_one_epoch(self):
        self.model.train()
        total_loss, total_cls_loss, total_recon_loss = 0.0, 0.0, 0.0
        total_n, correct = 0, 0
        
        use_bf16 = self.device_type == "cuda" and torch.cuda.is_bf16_supported()
        amp_dtype = torch.bfloat16 if use_bf16 else torch.float16

        for x, y in self.train_loader:
            x = x.to(self.device, non_blocking=True)
            if x.ndim == 4 and self.device_type == "cuda":
                x = x.to(memory_format=torch.channels_last)
                
            y = y.to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)
            
            with autocast(device_type=self.device_type, dtype=amp_dtype):
                out = self.model(x)
                if isinstance(out, tuple):
                    logits, x_recon = out
                    cls_loss = F.cross_entropy(logits, y)
                    recon_loss = F.mse_loss(x_recon, x)
                    loss = cls_loss + self.recon_weight * recon_loss
                else:
                    logits = out
                    cls_loss = F.cross_entropy(logits, y)
                    recon_loss = torch.tensor(0.0, device=self.device)
                    loss = cls_loss
                
            if torch.isnan(loss) or torch.isinf(loss):
                self._trigger_stop(StopReason.LOSS_NAN if torch.isnan(loss) else StopReason.LOSS_INF)
                return {}

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                self._trigger_stop(StopReason.GRAD_EXPLOSION)
                return {}

            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item() * x.size(0)
            total_cls_loss += cls_loss.item() * x.size(0)
            total_recon_loss += recon_loss.item() * x.size(0)
            correct += (logits.argmax(dim=-1) == y).sum().item()
            total_n += x.size(0)
            
        return {
            "loss": self._validate_metric(total_loss / max(1, total_n), "train_loss"),
            "cls_loss": self._validate_metric(total_cls_loss / max(1, total_n), "train_cls_loss"),
            "recon_loss": self._validate_metric(total_recon_loss / max(1, total_n), "train_recon_loss"),
            "acc": self._validate_metric(correct / max(1, total_n), "train_accuracy")
        }

    @torch.no_grad()
    def _validate_one_epoch(self):
        self.model.eval()
        total_loss, total_cls_loss, total_recon_loss = 0.0, 0.0, 0.0
        total_sam_deg, total_rmse = 0.0, 0.0
        total_n = 0
        all_preds, all_labels = [], []
        
        use_bf16 = self.device_type == "cuda" and torch.cuda.is_bf16_supported()
        amp_dtype = torch.bfloat16 if use_bf16 else torch.float16

        for x, y in self.val_loader:
            x = x.to(self.device, non_blocking=True)
            if x.ndim == 4 and self.device_type == "cuda":
                x = x.to(memory_format=torch.channels_last)
                
            y = y.to(self.device, non_blocking=True)
            
            with autocast(device_type=self.device_type, dtype=amp_dtype):
                out = self.model(x)
                if isinstance(out, tuple):
                    logits, x_recon = out
                    cls_loss = F.cross_entropy(logits, y)
                    recon_loss = F.mse_loss(x_recon, x)
                    loss = cls_loss + self.recon_weight * recon_loss
                    sam_deg = compute_sam(x, x_recon)
                    rmse_val = compute_spectral_rmse(x, x_recon)
                else:
                    logits = out
                    x_recon = x
                    cls_loss = F.cross_entropy(logits, y)
                    recon_loss = torch.tensor(0.0, device=self.device)
                    loss = cls_loss
                    sam_deg = 0.0
                    rmse_val = 0.0
                
            if torch.isnan(loss) or torch.isinf(loss):
                continue
                
            total_loss += loss.item() * x.size(0)
            total_cls_loss += cls_loss.item() * x.size(0)
            total_recon_loss += recon_loss.item() * x.size(0)
            total_sam_deg += sam_deg * x.size(0)
            total_rmse += rmse_val * x.size(0)
            
            total_n += x.size(0)
            all_preds.append(logits.argmax(dim=-1).cpu().numpy())
            all_labels.append(y.cpu().numpy())
            
        preds = np.concatenate(all_preds) if all_preds else np.array([])
        labels = np.concatenate(all_labels) if all_labels else np.array([])
        
        return {
            "loss": self._validate_metric(total_loss / max(1, total_n), "val_loss"),
            "cls_loss": self._validate_metric(total_cls_loss / max(1, total_n), "val_cls_loss"),
            "spectral_recon_loss": self._validate_metric(total_recon_loss / max(1, total_n), "spectral_recon_loss"),
            "spectral_sam_deg": self._validate_metric(total_sam_deg / max(1, total_n), "spectral_sam_deg"),
            "spectral_rmse": self._validate_metric(total_rmse / max(1, total_n), "spectral_rmse"),
            "acc": self._validate_metric(accuracy_score(labels, preds) if len(labels) > 0 else 0.0, "val_accuracy"),
            "precision": self._validate_metric(precision_score(labels, preds, average="weighted", zero_division=0), "precision"),
            "recall": self._validate_metric(recall_score(labels, preds, average="weighted", zero_division=0), "recall"),
            "f1": self._validate_metric(f1_score(labels, preds, average="weighted", zero_division=0), "f1"),
            "macro_f1": self._validate_metric(f1_score(labels, preds, average="macro", zero_division=0), "macro_f1"),
            "balanced_acc": self._validate_metric(balanced_accuracy_score(labels, preds) if len(labels) > 0 else 0.0, "balanced_acc")
        }

    def _check_stopping_rules(self):
        if len(self.history) < 2:
            return
            
        curr, prev = self.history[-1], self.history[-2]
        
        if curr["val_loss"] > prev["val_loss"] and curr["val_accuracy"] < prev["val_accuracy"]:
            self._increment_stall("val_diverge")
        else:
            self.stall_counters["val_diverge"] = 0
            
        if curr["train_loss"] > prev["train_loss"] and curr["train_accuracy"] < prev["train_accuracy"]:
            self._increment_stall("train_diverge")
        else:
            self.stall_counters["train_diverge"] = 0

        stalls = [
            ("val_acc_stall", curr["val_accuracy"] == prev["val_accuracy"], StopReason.VAL_ACC_STALLED),
            ("val_loss_stall", curr["val_loss"] == prev["val_loss"], StopReason.VAL_LOSS_STALLED),
            ("train_acc_stall", curr["train_accuracy"] == prev["train_accuracy"], StopReason.TRAIN_ACC_STALLED),
            ("train_loss_stall", curr["train_loss"] == prev["train_loss"], StopReason.TRAIN_LOSS_STALLED),
        ]
        
        for key, is_stalled, reason in stalls:
            if is_stalled:
                self._increment_stall(key, reason)
            else:
                self.stall_counters[key] = 0

        if self.stall_counters.get("val_diverge", 0) >= 2:
            self._trigger_stop(StopReason.VAL_DIVERGENCE)
        if self.stall_counters.get("train_diverge", 0) >= 2:
            self._trigger_stop(StopReason.TRAIN_DIVERGENCE)

    def _increment_stall(self, key, reason=None):
        self.stall_counters[key] = self.stall_counters.get(key, 0) + 1
        if self.stall_counters[key] >= 2 and reason:
            self._trigger_stop(reason)

    def _trigger_stop(self, reason: StopReason):
        self.stop_reason = reason
        self.is_stopped = True

    def _save_checkpoint(self, is_best):
        state = {
            "epoch": self.current_epoch,
            "best_epoch": self.best_epoch,
            "best_val_acc": self.best_val_acc,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "scaler_state": self.scaler.state_dict(),
            "history": self.history,
            "stall_counters": self.stall_counters
        }
        
        torch.save(state, self.dirs["checkpoints"] / f"epoch_{self.current_epoch:04d}.pt")
        torch.save(state, self.dirs["checkpoints"] / "latest.pt")
        if is_best:
            torch.save(state, self.dirs["checkpoints"] / "best.pt")

    def _save_epoch_report(self, metrics, is_best):
        report = {"epoch": metrics["epoch"], "metrics": metrics, "status": {"checkpoint_saved": True, "best_checkpoint": is_best}}
        with open(self.dirs["reports"] / f"epoch_{self.current_epoch:04d}.json", "w") as f:
            json.dump(report, f, indent=4)

    def _update_history_csv(self, metrics):
        path = self.exp_dir / "history.csv"
        file_exists = path.exists()
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(metrics)

    def _log_epoch(self, m, is_best):
        msg = (
            f"\n--- Epoch {m['epoch']:04d} ---\n"
            f"Train -> Loss: {m['train_loss']:.4f} (Cls: {m['train_cls_loss']:.4f}, Recon: {m['train_recon_loss']:.4f}) | Acc: {m['train_accuracy']:.4f}\n"
            f"Val   -> Loss: {m['val_loss']:.4f} | Acc: {m['val_accuracy']:.4f} | "
            f"Prec: {m['precision']:.4f} | Rec: {m['recall']:.4f} | F1: {m['macro_f1']:.4f}\n"
            f"Spectral Recon -> Loss: {m['spectral_recon_loss']:.4f} | SAM: {m['spectral_sam_deg']:.2f}° | RMSE: {m['spectral_rmse']:.4f}\n"
            f"Stats -> Time: {m['epoch_time']:.1f}s | LR: {m['learning_rate']:.2e} | "
            f"Saved: epoch_{m['epoch']:04d}.pt ({'BEST' if is_best else 'LATEST'})"
        )
        print(msg, flush=True)

    def _generate_final_report(self):
        report = {
            "completion_status": {
                "stop_reason": self.stop_reason.value if self.stop_reason else "UNKNOWN",
                "total_epochs_run": self.current_epoch,
                "best_epoch": self.best_epoch,
                "best_val_accuracy": self.best_val_acc
            },
            "history": self.history
        }
        with open(self.exp_dir / "experiment_report.json", "w") as f:
            json.dump(report, f, indent=4)
            
        print("\n====================================", flush=True)
        print("TRAINING PROCESS ENDED", flush=True)
        print("====================================", flush=True)
        print(f"Reason: {self.stop_reason.value if self.stop_reason else 'UNKNOWN'}", flush=True)
        print(f"Total Epochs Completed: {self.current_epoch}", flush=True)
        print(f"Best Epoch: {self.best_epoch} (Val Acc: {self.best_val_acc:.4f})", flush=True)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state"])
        self.scaler.load_state_dict(checkpoint["scaler_state"])
        
        self.current_epoch = checkpoint["epoch"]
        self.best_epoch = checkpoint.get("best_epoch", 0)
        self.best_val_acc = checkpoint.get("best_val_acc", -1.0)
        self.history = checkpoint.get("history", [])
        self.stall_counters = checkpoint.get("stall_counters", {})
        return self.current_epoch