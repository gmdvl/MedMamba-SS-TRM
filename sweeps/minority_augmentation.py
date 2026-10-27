"""
sweeps/minority_augmentation.py - does minority-class augmentation help DCIS?
(training/minority_augmentation.py, branch feature/minority-class-augmentation)

    python run_experiments.py sweeps/minority_augmentation.py --status
    python run_experiments.py sweeps/minority_augmentation.py screen --dry_run
    python run_experiments.py sweeps/minority_augmentation.py screen

DCIS is 5.0% of the training patches (122,850 of 2,452,086) from 5 patients.

`screen`: 12-epoch, seed-42 arms, read against `abl-base` (the paper's 12-epoch
reference, already in experiments/ - a normal invocation skips it). One knob per
arm against `minaug-default`:
    minaug-default   factor 2, 'stats' mixing at p 0.5, strength 1.0
    minaug-nomix     no mixing: oversampled copies differ only by augmentation draw
    minaug-pixel     intra-class mixup instead of statistics mixing
    minaug-f1        no oversampling: mixing only
    minaug-original  class weights from y_train.npy, i.e. compensated twice

`seeds`: the default arm at 20 epochs on the paper's 5 seeds, read against
`ref20` + `rq0-s{1,7,13,23}-hsi`. Change SEED_ARM to the screen's winner first.

Oversampling lengthens the epoch, so every factor-2 arm sets
`--train_subsample_frac` to keep the reference's 250,113 rows per epoch: the
same optimizer steps, with DCIS taking a larger share of them. Without that,
an arm would also get ~5% more steps than its reference.
"""

from run_experiments import train

HSI = "data/hsi_v8-80_10_10_importance-new/hsi"
REF = dict(profile="paper_recipe", batch_size=256, lambda_sam=0.1,
           train_subsample_frac=0.102, val_subsample_frac=0.0986)
SEEDS = (1, 7, 13, 23)

N_TRAIN, N_DCIS = 2452086, 122850                 # y_train.npy of the -new build
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
    "screen": [train("abl-base", HSI, **REF, epochs=12, seed=42)]
              + [train(tag, HSI, **flags, epochs=12, seed=42) for tag, flags in ARMS.items()],
    "seeds": [train(f"{SEED_ARM}-s{s}", HSI, **ARMS[SEED_ARM], epochs=20, seed=s)
              for s in (42,) + SEEDS],
}
