# -*- coding: utf-8 -*-
"""
augment_pipeline.py
===================
The HSI augmentation pipeline (plan/HSI_augmentation_pipeline.plan.md).

    python augment_pipeline.py --help
    python augment_pipeline.py validate preprocess --data_dir data/hsi_v9-trainsel/hsi \\
                                                    --out data/hsi_v9-trainsel-aug
    python augment_pipeline.py all --data_dir data/hsi_v9-trainsel/hsi --out data/hsi_v9-trainsel-aug
    python augment_pipeline.py --list --data_dir data/hsi_v9-trainsel/hsi --out data/hsi_v9-trainsel-aug

Each stage writes `<out>/<stage dir>/` and finishes with `done.json`; a done
stage is skipped unless its config, the seed, the input build or an upstream
stage changed (or `--force`). The planned commands are `validate | preprocess |
pool | fit-cvae | fit-ae | fit-ldm | generate | qc | export | lopo-utility |
bench | all`; `--help` shows which exist yet. The command line is
`hsi_augment/cli.py`; the stage contract is `hsi_augment/stage.py`; a new stage
is a file in `hsi_augment/stages/` (see `hsi_augment/registry.py`).
"""

import sys

from hsi_augment.cli import main

if __name__ == "__main__":
    sys.exit(main())
