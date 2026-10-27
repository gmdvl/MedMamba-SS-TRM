"""
sweeps/minority_augmentation.py - does minority-class augmentation help DCIS?
(training/minority_augmentation.py, branch feature/minority-class-augmentation)

    python run_experiments.py sweeps/minority_augmentation.py --status
    python run_experiments.py sweeps/minority_augmentation.py screen --dry_run
    python run_experiments.py sweeps/minority_augmentation.py screen

Data: the trainsel build. The v8 build's band selection saw held-out labels;
trainsel's saw training captures only, on the same split with the same counts.
DCIS is 5.0% of the training patches (122,850 of 2,452,086) from 5 patients.

`screen`: 12-epoch, seed-42 arms, read against `v20-abl-base-s42`. One knob per
arm against `minaug-default`:
    minaug-default   factor 2, 'stats' mixing at p 0.5, strength 1.0
    minaug-nomix     no mixing: oversampled copies differ only by augmentation draw
    minaug-pixel     intra-class mixup instead of statistics mixing
    minaug-f1        no oversampling: mixing only
    minaug-original  class weights from y_train.npy, i.e. compensated twice

`seeds`: the default arm at 20 epochs on the paper's 5 seeds, read against
`v20-head-s{1,7,13,23,42}`. Change SEED_ARM to the screen's winner first.

Both references are also jobs of sweeps/paper_evidence_upgrade.py (`controls`,
`headline`), and plan/HSI_augmentation_pipeline.plan.md (Phase 10) reuses
`minaug-nomix` and `minaug-default` as its copy arms A0os and A1. A job is done
once a run with its tag exists on its data, so whichever sweep runs a shared
tag first, the others skip it - which is only sound while every sweep builds
it with byte-identical flags (tests/test_minority_sweep.py). Hence no
`--minority_classes DCIS` (`auto` resolves to DCIS) and no `--trm_ema_rate` pin
(the profile resolves 0.9995 at both fracs: the upper clamp).

Oversampling lengthens the epoch, so every factor-2 arm sets
`--train_subsample_frac` to keep the reference's 250,113 rows per epoch: the
same optimizer steps, with DCIS taking a larger share of them. Without that,
an arm would also get ~5% more steps than its reference.
"""

from run_experiments import train

HSI = "data/hsi_v9-trainsel/hsi"
REF = dict(profile="paper_recipe", batch_size=256, lambda_sam=0.1,
           train_subsample_frac=0.102, val_subsample_frac=0.0986)
SEEDS = (1, 7, 13, 23, 42)

N_TRAIN, N_DCIS = 2452086, 122850                 # y_train.npy of the trainsel build
KEPT = round(N_TRAIN * REF["train_subsample_frac"])  # 250,113 rows per epoch


def matched(factor: float, **flags) -> dict:
    """REF + `--minority_augment on` at `factor`, with the rows per epoch held at KEPT."""
    frac = (REF["train_subsample_frac"] if factor == 1.0
            else round(KEPT / (N_TRAIN + round(N_DCIS * (factor - 1))), 7))
    return {**REF, "train_subsample_frac": frac, "minority_augment": "on",
            "minority_oversample_factor": factor, **flags}


ARMS = {
    "minaug-default": matched(2.0),
    "minaug-nomix": matched(2.0, minority_mix_prob=0.0),
    "minaug-pixel": matched(2.0, minority_mix_mode="pixel"),
    "minaug-f1": matched(1.0),
    "minaug-original": matched(2.0, minority_class_weights="original"),
}
SEED_ARM = "minaug-default"

STAGES = {
    "screen": [train("v20-abl-base-s42", HSI, **REF, epochs=12, seed=42)]
              + [train(tag, HSI, **flags, epochs=12, seed=42) for tag, flags in ARMS.items()],
    "seeds": [train(f"v20-head-s{s}", HSI, **REF, epochs=20, seed=s) for s in SEEDS]
             + [train(f"{SEED_ARM}-s{s}", HSI, **ARMS[SEED_ARM], epochs=20, seed=s) for s in SEEDS],
}
