# -*- coding: utf-8 -*-
"""
sweeps/smoke.py - the GPU check the CPU parity tests cannot do.

    python run_experiments.py sweeps/smoke.py all

Two stages, each a few minutes on the RTX 5060 Ti:

  profiles    one 1-epoch run per profile, on the dataset the profile was built for:
              exercises what only a GPU run reaches - torch.compile, bf16 autocast on
              CUDA, the fused fast loop, pinned-memory workers, spectral checkpointing.
  gpu-parity  the archived train_example_v18.py and `train.py --profile paper_recipe` on the
              same HSI subsample, both --deterministic, then a comparison of their
              histories (this file, run as a script).

Runs are tagged `smoke-*`; delete them from experiments/ afterwards. HSI runs skip
the test split (349k patches, ~15 min) and the deep dataset read.

    python sweeps/smoke.py           # the gpu-parity comparison on its own
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_experiments import PY, cmd, find_runs, train  # noqa: E402

HSI = "data/hsi_v8-80_10_10_importance-new/hsi"
PAD = "data/pad_optimal"
ONE = dict(epochs=1, seed=42, skip_dataset_validation=True)
HSI_SMALL = dict(batch_size=256, lambda_sam=0.1, train_subsample_frac=0.005,
                 val_subsample_frac=0.01, eval_test="off")
PARITY = ["--data_dir", HSI, "--epochs", "1", "--seed", "42", "--skip_dataset_validation",
          "--deterministic", "--batch_size", "256", "--lambda_sam", "0.1",
          "--train_subsample_frac", "0.005", "--val_subsample_frac", "0.01", "--eval_test", "off"]
ENV = ["env", "PYTHONPATH=.:archive", "CUBLAS_WORKSPACE_CONFIG=:4096:8"]
TIMING = {"GPU_memory_MB", "artifact_time_s", "data_wait_s", "epoch_time", "s_per_step",
          "train_time_s", "val_time_s"}
CORE = ("train_loss", "val_loss", "val_accuracy", "balanced_accuracy", "f1_macro")


def smoke(profile, data, **flags):
    tag = f"smoke-{profile.replace('_', '-')}"        # '_' separates fields in run names
    job = train(tag, data, name=tag, profile=profile, **ONE, **flags)
    job.done = lambda: bool(find_runs(tag, data, need="history.json"))
    return job


def _history_done(tag):
    return lambda: bool(find_runs(tag, HSI, need="history.json"))


STAGES = {
    "profiles": [
        smoke("pipeline_defaults", HSI, **HSI_SMALL),
        smoke("medmamba_protocol_norecon", PAD),
        smoke("pad_ufes_best_norecon", PAD),
        smoke("paper_recipe", HSI, **HSI_SMALL),
    ],
    "gpu-parity": [
        cmd("parity-old-v18", ENV + [PY, "archive/train_example_v18.py", *PARITY,
                                     "--run_tag", "smoke-old-v18"],
            done_if=_history_done("smoke-old-v18")),
        cmd("parity-new-v18", ENV + [PY, "train.py", "--profile", "paper_recipe", *PARITY,
                                     "--run_tag", "smoke-new-v18"],
            done_if=_history_done("smoke-new-v18")),
        cmd("parity-compare", [PY, "sweeps/smoke.py"]),
    ],
}
GROUPS = {"all": ["profiles", "gpu-parity"]}


def compare() -> int:
    """Old vs new v18 on the GPU. Bit-identical is the expectation under
    --deterministic; ops without a deterministic CUDA kernel run in warn-only
    mode, so a difference at the 1e-6 level is noise, not the refactor."""
    runs = {side: find_runs(f"smoke-{side}-v18", HSI, need="history.json") for side in ("old", "new")}
    if not all(runs.values()):
        print(f"[parity] missing run(s): {[s for s, r in runs.items() if not r]}")
        return 1
    old, new = runs["old"][-1], runs["new"][-1]
    h_old, h_new = (json.loads((d / "history.json").read_text()) for d in (old, new))
    worst = {}
    for a, b in zip(h_old, h_new):
        for k, v in a.items():
            if k in TIMING or not isinstance(v, (int, float)) or isinstance(v, bool):
                continue
            w = b.get(k)
            rel = 0.0 if v == w else abs(v - w) / max(abs(v), abs(w), 1e-12) if isinstance(w, (int, float)) else 1.0
            worst[k] = max(worst.get(k, 0.0), rel)
    a_old, a_new = (json.loads((d / "config.json").read_text())["cli_args"] for d in (old, new))
    renamed = {"stop_on_val_acc_stall": "stop_on_metric_stall", "early_stopping_enabled": "early_stopping"}
    arg_diff = {k: (v, a_new.get(renamed.get(k, k))) for k, v in a_old.items()
                if k != "run_tag" and a_new.get(renamed.get(k, k)) != v}
    core = max(worst.get(k, 0.0) for k in CORE)
    for k in CORE:
        print(f"[parity] {k:18s} old {h_old[-1].get(k)!r:>24}  new {h_new[-1].get(k)!r:>24}  "
              f"rel diff {worst.get(k, 0.0):.2e}")
    top = sorted(worst.items(), key=lambda kv: -kv[1])[:5]
    print("[parity] largest relative differences: " + ", ".join(f"{k}={v:.1e}" for k, v in top))
    print(f"[parity] cli_args differing: {arg_diff or 'none'}")
    verdict = ("IDENTICAL" if not arg_diff and max(worst.values(), default=0.0) == 0.0 else
               "SAME within GPU nondeterminism" if not arg_diff and core <= 1e-4 else "DIFFERENT")
    print(f"[parity] {verdict}  ({old.name} vs {new.name})")
    return 0 if verdict != "DIFFERENT" else 1


if __name__ == "__main__":
    sys.exit(compare())
