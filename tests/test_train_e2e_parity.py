# -*- coding: utf-8 -*-
"""
test_train_e2e_parity.py
=========================
Both sides actually TRAIN. For every profile, the former entry point
(`archive/`) and `train.py --profile X` run two epochs on the same tiny
synthetic HSI dataset on the CPU, and the results must be identical:

    history.json          every per-epoch number except wall-clock timings
    test_predictions.npz  the held-out predictions, bitwise
    gates.json            G1/G2/G9, G5, G6, G7, G8
    config.json           every cli_arg the old run recorded; the step schedule

CPU runs are deterministic run-to-run at this size (checked when this test was
written), so any difference is the refactor. ~10 s per run, 10 runs.

Run with: pytest tests/test_train_e2e_parity.py -q
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
OLD_DIR = REPO_ROOT if (REPO_ROOT / "train_example_v18.py").exists() else REPO_ROOT / "archive"
OLD_SCRIPT = {
    "base": "train_example_v16.py", "original": "train_example_v16_original.py",
    "optimal": "train_example_v16_optimal.py", "v18": "train_example_v18.py",
    "recon": "train_example_v16_recon.py",
}
# old entry point -> its profile; _recon has none of its own: the artifact flags exist everywhere now
NEW_PROFILE = {"base": "pipeline_defaults", "recon": "pipeline_defaults",
               "original": "medmamba_protocol_norecon", "optimal": "pad_ufes_best_norecon",
               "v18": "paper_recipe"}
TINY = ["--epochs", "2", "--batch_size", "16", "--trm_dim", "16", "--trm_n_latent", "1",
        "--trm_n_improve", "1", "--trm_deep_supervision_steps", "1", "--compile", "off",
        "--loader_mode", "safe", "--drift_check_patches", "64",
        # a 0.05 M-parameter model on random-ish inputs sits just under gate G2's
        # 0.01 logit-std floor at init; the gate still runs, on both sides
        "--min_logit_std", "1e-4"]
TIMING = {"GPU_memory_MB", "artifact_time_s", "data_wait_s", "epoch_time", "s_per_step",
          "train_time_s", "val_time_s"}
RENAMED = {"stop_on_val_acc_stall": "stop_on_metric_stall", "early_stopping_enabled": "early_stopping"}


@pytest.fixture(scope="module")
def tiny_hsi(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("tiny") / "hsi"
    out.mkdir()
    rng = np.random.default_rng(0)
    for split, n in (("train", 96), ("val", 48), ("test", 48)):
        y = np.repeat(np.arange(3), n // 3).astype(np.int64)
        x = rng.random((n, 8, 8, 8), dtype=np.float32) * 0.5 + (y[:, None, None, None] * 0.15)
        np.save(out / f"X_{split}.npy", x.astype(np.float32))
        np.save(out / f"y_{split}.npy", y)
    np.save(out / "wavelengths.npy", np.linspace(450, 900, 8).astype(np.float32))
    (out / "class_names.json").write_text(json.dumps(["a", "b", "c"]))
    return out


def _train(script: Path, argv, cwd: Path) -> Path:
    cwd.mkdir(parents=True)
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(REPO_ROOT), str(REPO_ROOT / "archive")]),
               CUDA_VISIBLE_DEVICES="")
    proc = subprocess.run([sys.executable, str(script)] + argv, cwd=cwd, env=env,
                          capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, f"{script.name} failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"
    (run_dir,) = list((cwd / "experiments").iterdir())
    return run_dir


def _history(run_dir: Path):
    return [{k: v for k, v in row.items() if k not in TIMING}
            for row in json.loads((run_dir / "history.json").read_text())]


def _gates(run_dir: Path) -> dict:
    """gates.json minus G5's checkpoint path, which contains the run directory."""
    gates = json.loads((run_dir / "gates.json").read_text())
    if isinstance(gates.get("G5"), dict):
        gates["G5"].pop("checkpoint", None)
    return gates


@pytest.mark.parametrize("profile", sorted(OLD_SCRIPT))
def test_old_entry_point_and_train_py_produce_the_same_run(profile, tiny_hsi, tmp_path):
    argv = ["--data_dir", str(tiny_hsi)] + TINY
    old = _train(OLD_DIR / OLD_SCRIPT[profile], argv, tmp_path / "old")
    new = _train(REPO_ROOT / "train.py", argv + ["--profile", NEW_PROFILE[profile]],
                 tmp_path / "new")

    assert old.name[16:] == new.name[16:], "run directory names differ beyond the timestamp"

    h_old, h_new = _history(old), _history(new)
    assert len(h_old) == len(h_new)
    for epoch, (a, b) in enumerate(zip(h_old, h_new), 1):
        diff = {k: (a[k], b.get(k)) for k in a if a[k] != b.get(k)}
        assert not diff, f"epoch {epoch}: {diff}"

    p_old, p_new = np.load(old / "test_predictions.npz"), np.load(new / "test_predictions.npz")
    assert sorted(p_old.files) == sorted(p_new.files)
    for key in p_old.files:
        np.testing.assert_array_equal(p_old[key], p_new[key], err_msg=key)

    assert _gates(old) == _gates(new)

    c_old, c_new = (json.loads((d / "config.json").read_text()) for d in (old, new))
    a_old, a_new = c_old["cli_args"], c_new["cli_args"]
    diff = {k: (v, a_new.get(RENAMED.get(k, k))) for k, v in a_old.items()
            if k != "data_dir" and a_new.get(RENAMED.get(k, k)) != v}
    assert not diff, f"cli_args differ: {diff}"
    assert c_old["scheduler"]["total_steps"] == c_new["scheduler"]["total_steps"]
    assert c_old["scheduler"]["warmup_steps"] == c_new["scheduler"]["warmup_steps"]
