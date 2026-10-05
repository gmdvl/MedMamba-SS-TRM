# -*- coding: utf-8 -*-
"""
sweeps/paper_runs.py - the v18 experiment set: every run behind the paper's numbers
(`sweeps/v18.py` until 2026-10-02; the run tags are unchanged).

    python run_experiments.py sweeps/paper_runs.py --status
    python run_experiments.py sweeps/paper_runs.py depth --dry_run

Transcribed from documentations/15_v18_commands.md and scripts/short_runs_v18*.sh
(archived). All of these runs EXIST in experiments/, so a normal invocation
skips them; `tests/test_sweeps.py` resolves every train job here and checks it
against the `cli_args` its run recorded - i.e. that `train.py --profile paper_recipe`
with these flags is the command that produced the paper.

Every train job is the reference recipe with the stated flags changed and
nothing else. 12-epoch arms are read against `abl-base`, never against the
20-epoch reference: `--epochs` sets the cosine horizon.
"""

from run_experiments import cmd, train

HSI = "data/hsi_v8-80_10_10_importance-new/hsi"
RGB = "data/hsi_v8-80_10_10_importance-new/rgb"
REF = dict(profile="paper_recipe", batch_size=256, lambda_sam=0.1,
           train_subsample_frac=0.102, val_subsample_frac=0.0986)
NOSAM = {k: v for k, v in REF.items() if k != "lambda_sam"}     # recon off: no SAM term exists
HEAD42 = "experiments/20260915_031356_10_10_importance-new-hsi_recursive_hsi_mlp_sup3_bs256_bf16_focalw_zscore_sub0102_vsub0986_optimal_recon_ref20"
SEEDS = (1, 7, 13, 23)


def baseline(arch, seed):
    return train(f"s{seed}", HSI, name=f"{arch}-s{seed}", script="scripts/train_hsi_baseline.py",
                 match_arch=arch, arch=arch, seed=seed)


STAGES = {
    "reference": [train("ref20", HSI, **REF, epochs=20, seed=42)],
    "abl-base": [train("abl-base", HSI, **REF, epochs=12, seed=42)],
    "rq0": [train("rq0-rgb", RGB, **REF, epochs=20, seed=42)]
           + [train(f"rq0-s{s}", d, name=f"rq0-s{s}-{m}", **REF, epochs=20, seed=s)
              for s in SEEDS for m, d in (("hsi", HSI), ("rgb", RGB))],
    "arch": [train(f"arch-{a}", HSI, **REF, epochs=12, seed=42, architecture=a)
             for a in ("split", "fullchannel")],
    "depth": [train(f"depth-n{t}", HSI, **REF, epochs=12, seed=42, trm_n_improve=t) for t in (1, 2, 4)],
    "k2": [train("k2-control", HSI, **REF, epochs=12, seed=42,
                 trm_n_latent=1, trm_n_improve=1, trm_deep_supervision_steps=1)],
    "recon-off": [train("recon-off", HSI, **NOSAM, epochs=12, seed=42, recon_mode="none")],
    "enc-index": [train("enc-index", HSI, **REF, epochs=12, seed=42, no_use_wavelengths=True)],
    "bands": [job for n in (16, 8) for job in (
        cmd(f"bands{n}-build", ["python", "scripts/make_band_subset_build.py", "--src", HSI,
                                "--keep", str(n), "--out", f"data/hsi_v8-bands{n}"],
            done_if=f"data/hsi_v8-bands{n}/y_test.npy"),
        train(f"bands{n}", f"data/hsi_v8-bands{n}", **REF, epochs=12, seed=42))],
    "probes": [train(f"probe-split-lr{lr}", HSI, **REF, epochs=3, seed=42, architecture="split", lr=lr)
               for lr in ("1e-4", "3e-5")]
              + [train("probe-split-norecon", HSI, **NOSAM, epochs=3, seed=42,
                       architecture="split", recon_mode="none"),
                 train("probe-split-clip5", HSI, **REF, epochs=3, seed=42,
                       architecture="split", max_gradient_norm=5.0)],
    "baselines": [baseline(a, s) for a in ("spectralformer", "hybridsn") for s in SEEDS + (42,)],
    "zeroshot": [
        cmd("zeroshot", ["python", "scripts/eval_band_decimation.py", "--run_dir", HEAD42,
                         "--keep_bands", "32,16,8,4,2"],
            done_if=f"{HEAD42}/band_decimation/summary.json"),
        cmd("zeroshot-fixed-eta", ["python", "scripts/eval_band_decimation_fixed_eta.py",
                                   "--run_dir", HEAD42, "--keep_bands", "32,16,8"],
            done_if=f"{HEAD42}/band_decimation_fixed_eta/summary.json"),
    ],
}

GROUPS = {"all": list(STAGES)}
