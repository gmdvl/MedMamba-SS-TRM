# MedMamba-SS and MedMamba-SS-TRM

**Band-count-agnostic spectral–spatial medical image classification with a weight-shared recursive backbone.**

This repository holds the code for a master's thesis at Lakehead University and the companion paper
*"MedMamba-SS and MedMamba-SS-TRM: Band-Count-Agnostic Spectral–Spatial Medical Image Classification
with a Weight-Shared Recursive Backbone"* (Gabriel M. Del Valle L., Saad B. Ahmed).

It extends two published works, and it is evaluated on two public datasets. All four are third-party
material. **If you use this repository, you must also cite them.** See [Citing](#citing) and
[Third-party material and licences](#third-party-material-and-licences).

| | what it is | source |
|---|---|---|
| **MedMamba** | the Vision-Mamba medical image classifier this work extends | [YubiaoYue/MedMamba](https://github.com/YubiaoYue/MedMamba) · [arXiv:2403.03849](https://arxiv.org/abs/2403.03849) |
| **Tiny Recursive Model (TRM)** | the weight-shared recursive reasoning scheme used by MedMamba-SS-TRM | [SamsungSAILMontreal/TinyRecursiveModels](https://github.com/SamsungSAILMontreal/TinyRecursiveModels) · [arXiv:2510.04871](https://arxiv.org/abs/2510.04871) |
| **HistologyHSI-BC-Recurrence** | breast histology hyperspectral images (primary dataset) | [TCIA collection](https://www.cancerimagingarchive.net/collection/histologyhsi-bc-recurrence/) · [doi:10.7937/6KPY-YT49](https://doi.org/10.7937/6KPY-YT49) |
| **PAD-UFES-20** | smartphone skin-lesion photographs (RGB dataset) | [Mendeley Data](https://data.mendeley.com/datasets/zr7vgbcyr2/1) · [doi:10.17632/zr7vgbcyr2.1](https://doi.org/10.17632/zr7vgbcyr2.1) |

---

## Overview

Hyperspectral images carry many narrow bands. MedMamba embeds its input with a convolution whose
shape depends on the channel count, and it does not know which wavelength each channel is. This
repository extends it twice:

- **MedMamba-SS** replaces the patch embedding with a *spectral pathway*. The pathway has a
  wavelength-encoded tokenizer shared across bands, a bidirectional spectral selective scan, and band
  pooling. None of its parameters depends on the number of bands. The pathway conditions every stage of
  MedMamba's hierarchical spatial backbone.
- **MedMamba-SS-TRM** replaces that hierarchy with a single two-layer core that is applied recursively,
  following the Tiny Recursive Model. It refines a latent state `z` and an answer state `y`, and it
  uses deep supervision and an optional halting head.

Main results from the paper, on breast histology (45 patients, patient-grouped splits):

- MedMamba-SS-TRM has **446,409 parameters** at both 32 bands and 3 bands.
- On the five test patients, it reaches **92.8 ± 1.9 % balanced accuracy** over five seeds.
- The recursive model is **6.2× smaller** than the hierarchy it replaces, but it needs **22.3× more
  arithmetic**.

> **Caveat:** the 32 bands were chosen using labels that included held-out patients, so the
> hyperspectral numbers are **provisional**. The paper discloses this. The benefit of 32 bands over
> 3 bands also depends on the patient. See the paper for the full results, the baselines and the
> limitations.

---

## Repository layout

| path | contents |
|---|---|
| [`medmamba_ss_trm.py`](medmamba_ss_trm.py) | the models: `MedMambaSS`, `MedMambaSSTRM`, their backbones, and the shared `MedMambaSSTRMConfig` |
| [`medmamba_ss_trm_ema.py`](medmamba_ss_trm_ema.py) | weight EMA, ported from the TRM reference implementation |
| [`medmamba_ss_fullchannel.py`](medmamba_ss_fullchannel.py), [`medmamba_ss_efficient.py`](medmamba_ss_efficient.py) | model variants used in ablations |
| [`train.py`](train.py) | **one training run**, configured by `--profile` |
| [`run_experiments.py`](run_experiments.py) + [`sweeps/`](sweeps/) | **many runs**: the paper's experiment sets, with resume and status reporting |
| [`prepare_histologyhsi_bc.py`](prepare_histologyhsi_bc.py) | HistologyHSI-BC-Recurrence: ENVI cubes → patch `.npy` (HSI and RGB with a shared split) |
| [`prepare_pad_ufes_20_optimal.py`](prepare_pad_ufes_20_optimal.py), [`prepare_pad_ufes_20_original.py`](prepare_pad_ufes_20_original.py) | PAD-UFES-20: images → `.npy` (the tuned protocol and MedMamba's protocol) |
| [`training/`](training/) | trainer, data loading, losses, metrics, HSI baselines (HybridSN, SpectralFormer), plotting |
| [`scripts/`](scripts/) | evaluation, FLOPs, latent-space analysis, band decimation, baselines, paper tables and figures |
| [`tests/`](tests/) | the CPU test suite, including parity tests against the archived entry points |
| [`documentations/`](documentations/) | detailed reference documentation; [start with its README](documentations/README.md) |
| [`archive/`](archive/) | superseded entry points, kept unchanged as the reference for the parity tests ([README](archive/README.md)) |
| [`paper/`](paper/) | manuscript sources and figures |

> **Older names.** Code, logs and documents written before 2026-10-02 call the project *GMedMamba*.
> `gmedmamba.py` is now `medmamba_ss_trm.py`, `GMedMamba` is now `MedMambaSS`, and
> `GMedMambaRecursive` is now `MedMambaSSTRM`. The old import names still work through
> `archive/gmedmamba.py`.

---

## Installation

The project was developed with Python 3.11 and PyTorch 2.11 built for CUDA 12.8, on an RTX 5060 Ti
(Blackwell, `sm_120`). Blackwell GPUs need a **cu128** PyTorch build. Older builds such as cu124 have
no kernels for them.

```bash
mamba create --name medmamba-ss-trm --file mamba-spec.txt
mamba activate medmamba-ss-trm
pip install -r requirements.txt

# check that the build matches the GPU
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_arch_list())"
```

`spectral` (ENVI reader) is required for any histology preparation. `openpyxl` is only needed to read
the clinical workbook. The selective scan is written in pure PyTorch, so `mamba_ssm` and custom CUDA
kernels are not needed.

---

## Data

**This repository does not include any data.** Download each dataset from its official source and
accept its terms of use (see [below](#third-party-material-and-licences)).

### HistologyHSI-BC-Recurrence (hyperspectral + RGB)

Download the collection from [TCIA](https://www.cancerimagingarchive.net/collection/histologyhsi-bc-recurrence/)
(≈1.2 TB). A single `--modality both` pass writes `hsi/` and `rgb/` with the **same samples and the
same patient-grouped split**, which is what makes the HSI-vs-RGB comparison controlled:

```bash
python prepare_histologyhsi_bc.py \
    --root /path/to/HistologyHSI-BC-Recurrence/ \
    --out_dir ./data/hsi_v8-80_10_10_importance \
    --modality both --label_source tissue \
    --patch_size 11 --stride 11 --roi_min_frac 0.8 \
    --band_selection importance --num_bands 32 \
    --band_min_gap 8 --band_max_corr 0.95 --band_min_coverage 0.30 \
    --split 80_10_10 --split_strategy stratified \
    --capture_gain median_ratio --hsi_value_scale auto \
    --rgb_source synthetic \
    --seed 42 --num_workers 8 --verify_level deep
```

### PAD-UFES-20 (RGB)

Download from [Mendeley Data](https://data.mendeley.com/datasets/zr7vgbcyr2/1). Splits are grouped
by `patient_id`, so no patient appears in two splits:

```bash
python prepare_pad_ufes_20_optimal.py \
    --root /path/to/PAD-UFES-20 \
    --out_dir ./data/pad_optimal \
    --tiling whole --img_size 224 \
    --split 70/15/15 --split_strategy stratified --balance_classes none \
    --seed 42 --num_workers 8
```

[`documentations/12_commands_datasets.md`](documentations/12_commands_datasets.md) explains every flag
and the checks to run after preparation.

---

## Training

```bash
# the paper's configuration (default profile: paper_recipe)
python train.py --profile paper_recipe --data_dir data/hsi_v8-80_10_10_importance/hsi \
    --batch_size 256 --epochs 12 --lambda_sam 0.1 \
    --train_subsample_frac 0.102 --val_subsample_frac 0.0986 --seed 42 --run_tag my-run

# best configuration for PAD-UFES-20
python train.py --profile pad_ufes_best_norecon --data_dir ./data/pad_optimal

python train.py --help    # every flag
```

| profile | use it for | reconstruction |
|---|---|---|
| `paper_recipe` *(default)* | reproducing or extending the paper's numbers | latent, `lambda_mse 0.1` |
| `pad_ufes_best_norecon` | best metrics on PAD-UFES-20 | none |
| `medmamba_protocol_norecon` | comparing against MedMamba's own training recipe | none |
| `pipeline_defaults` | the bare pipeline, with no tuning on top | latent, `lambda_mse 1.0` |

`--lambda_sam 0.1`, `--batch_size`, `--epochs` and the subsample fractions are **not** profile
defaults. The paper runs pass them explicitly. The exact values for each run are in
[`sweeps/paper_runs.py`](sweeps/paper_runs.py).

To run a whole experiment set:

```bash
python run_experiments.py sweeps/paper_evidence_upgrade.py --list      # stages and jobs
python run_experiments.py sweeps/paper_evidence_upgrade.py --status    # done / waiting / to run
python run_experiments.py sweeps/paper_evidence_upgrade.py headline    # run one stage
```

Each run writes its own directory under `experiments/`, containing `history.json`, the best
checkpoint, plots and the test metrics. See [`documentations/17_train_and_sweeps.md`](documentations/17_train_and_sweeps.md)
and [`documentations/18_top_runs.md`](documentations/18_top_runs.md).

### Baselines

- **HybridSN and SpectralFormer** are re-implemented for 11 × 11 × C patches in
  [`training/hsi_baselines.py`](training/hsi_baselines.py). Train them with
  `scripts/train_hsi_baseline.py --arch {hybridsn,spectralformer}`.
- **Shallow baselines** (linear probes, etc.): `scripts/shallow_baseline_v16.py`.
- **Original MedMamba** results come from the [official MedMamba code](https://github.com/YubiaoYue/MedMamba),
  run on the same `.npy` splits.

---

## Tests

```bash
pytest            # CPU only; data/, experiments/ and archive/ are excluded by pytest.ini
```

The suite covers the model, data preparation, numerical stability and run naming. It also checks
that `train.py` resolves every argument, and trains to identical results, compared with the archived
entry points that produced the paper's numbers.

---

## Citing

If you use this code, please cite this work **and** the works it builds on, listed below.

```bibtex
@misc{delvalle2026medmambasstrm,
  title  = {{MedMamba-SS} and {MedMamba-SS-TRM}: Band-Count-Agnostic Spectral--Spatial Medical
            Image Classification with a Weight-Shared Recursive Backbone},
  author = {Del Valle L., Gabriel M. and Ahmed, Saad B.},
  year   = {2026},
  note   = {Manuscript under review}
}
```

### Methods: required citations

**MedMamba.** This is the citation requested by the MedMamba repository:

```bibtex
@article{yue2024medmamba,
  title   = {MedMamba: Vision Mamba for Medical Image Classification},
  author  = {Yue, Yubiao and Li, Zhenzhang},
  journal = {arXiv preprint arXiv:2403.03849},
  year    = {2024}
}
```

**Tiny Recursive Model.** The TRM repository asks for this citation, and also asks you to cite the
Hierarchical Reasoning Model (HRM) that TRM builds on:

```bibtex
@misc{jolicoeurmartineau2025morerecursivereasoningtiny,
  title         = {Less is More: Recursive Reasoning with Tiny Networks},
  author        = {Alexia Jolicoeur-Martineau},
  year          = {2025},
  eprint        = {2510.04871},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2510.04871}
}

@misc{wang2025hierarchicalreasoningmodel,
  title         = {Hierarchical Reasoning Model},
  author        = {Guan Wang and Jin Li and Yuhao Sun and Xing Chen and Changling Liu and Yue Wu
                   and Meng Lu and Sen Song and Yasin Abbasi Yadkori},
  year          = {2025},
  eprint        = {2506.21734},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  url           = {https://arxiv.org/abs/2506.21734}
}
```

### Datasets: required citations

**HistologyHSI-BC-Recurrence.** This dataset is distributed by The Cancer Imaging Archive (TCIA)
under CC BY 4.0. The TCIA Data Usage Policy requires you to cite the dataset **with its DOI** in any
publication or presentation that uses the data, and to acknowledge TCIA as the repository. TCIA also
asks you to cite its own publication (Clark et al., 2013). Cite the data descriptor paper as well:

> Quintana Quintana, L., Sauras-Colón, E., Lopez Pablo, C., Santana Núñez, J., Fiorin, A., Ortega
> Sarmiento, S., Fabelo, H., Gallardo Borràs, N., Fischer-Carles, A., Adalid Llansa, L., Mata Cano, D.,
> Bosch Príncep, R., & Callico, G. (2025). *Recurrent Breast Cancer: Histopathological and Hyperspectral
> Images Database (HistologyHSI-BC-Recurrence)* (Version 1) [Data set]. The Cancer Imaging Archive.
> https://doi.org/10.7937/6KPY-YT49

```bibtex
@misc{quintana2025histologyhsi_data,
  title        = {Recurrent Breast Cancer: Histopathological and Hyperspectral Images Database
                  ({HistologyHSI-BC-Recurrence}) (Version 1) [Data set]},
  author       = {Quintana Quintana, L. and Sauras-Col{\'o}n, E. and Lopez Pablo, C. and
                  Santana N{\'u}{\~n}ez, J. and Fiorin, A. and Ortega Sarmiento, S. and Fabelo, H.
                  and Gallardo Borr{\`a}s, N. and Fischer-Carles, A. and Adalid Llansa, L. and
                  Mata Cano, D. and Bosch Pr{\'i}ncep, R. and Callico, G.},
  year         = {2025},
  publisher    = {The Cancer Imaging Archive},
  doi          = {10.7937/6KPY-YT49}
}

@article{quintana2025histologyhsi_paper,
  title   = {Histological Hyperspectral Breast Cancer Recurrence Database
             ({HistologyHSI-BC} Recurrence)},
  author  = {Quintana-Quintana, Laura and Sauras-Col{\'o}n, Esther and Fiorin, Alessio and
             Santana-Nunez, Javier and Ortega, Samuel and Gallardo-Borr{\`a}s, No{\`e}lia and
             Fischer-Carles, Alba and S{\'a}nchez-Alc{\'a}ntara, T{\'a}bata and Fabelo, Himar and
             Adalid-Llansa, Laia and Mata-Cano, Daniel and Bosch-Pr{\'i}ncep, Ramon and
             Lejeune, Maryl{\`e}ne and Callico, Gustavo M. and L{\'o}pez-Pablo, Carlos},
  journal = {Scientific Data},
  volume  = {12},
  pages   = {1886},
  year    = {2025},
  doi     = {10.1038/s41597-025-06157-4}
}

@article{clark2013tcia,
  title   = {The Cancer Imaging Archive ({TCIA}): Maintaining and Operating a Public Information
             Repository},
  author  = {Clark, Kenneth and Vendt, Bruce and Smith, Kirk and Freymann, John and Kirby, Justin and
             Koppel, Paul and Moore, Stephen and Phillips, Stanley and Maffitt, David and
             Pringle, Michael and Tarbox, Lawrence and Prior, Fred},
  journal = {Journal of Digital Imaging},
  volume  = {26},
  number  = {6},
  pages   = {1045--1057},
  year    = {2013},
  doi     = {10.1007/s10278-013-9622-7}
}
```

**PAD-UFES-20.** This dataset is distributed on Mendeley Data under CC BY 4.0. Cite the dataset by
its DOI, together with its data article. The Mendeley record also lists the related study in
*Computers in Biology and Medicine*:

```bibtex
@misc{pacheco2020padufes20_data,
  title     = {{PAD-UFES-20}: a skin lesion dataset composed of patient data and clinical images
               collected from smartphones},
  author    = {Pacheco, Andre G. C. and Lima, Gustavo R. and Salom{\~a}o, Amanda S. and
               Krohling, Breno and Biral, Igor P. and de Angelo, Gabriel G. and Alves Jr, F{\'a}bio C. R.
               and Esgario, Jos{\'e} G. M. and Simora, Alana C. and Castro, Pedro B. C. and
               Rodrigues, Felipe B. and Frasson, Patricia H. L. and Krohling, Renato A. and
               Knidel, Helder and Santos, Maria C. S. and Esp{\'i}rito Santo, Rachel B. and
               Macedo, Telma L. S. G. and Canuto, Tania R. P. and de Barros, Lu{\'i}z F. S.},
  year      = {2020},
  publisher = {Mendeley Data},
  version   = {1},
  doi       = {10.17632/zr7vgbcyr2.1}
}

@article{pacheco2020padufes20,
  title   = {{PAD-UFES-20}: A skin lesion dataset composed of patient data and clinical images
             collected from smartphones},
  author  = {Pacheco, Andre G. C. and Lima, Gustavo R. and Salom{\~a}o, Amanda S. and
             Krohling, Breno and Biral, Igor P. and de Angelo, Gabriel G. and Alves Jr, F{\'a}bio C. R.
             and Esgario, Jos{\'e} G. M. and Simora, Alana C. and Castro, Pedro B. C. and
             Rodrigues, Felipe B. and Frasson, Patricia H. L. and Krohling, Renato A. and
             Knidel, Helder and Santos, Maria C. S. and do Esp{\'i}rito Santo, Rachel B. and
             Macedo, Telma L. S. G. and Canuto, Tania R. P. and de Barros, Lu{\'i}z F. S.},
  journal = {Data in Brief},
  volume  = {32},
  pages   = {106221},
  year    = {2020},
  doi     = {10.1016/j.dib.2020.106221}
}

@article{pacheco2020impact,
  title   = {The impact of patient clinical information on automated skin cancer detection},
  author  = {Pacheco, Andre G. C. and Krohling, Renato A.},
  journal = {Computers in Biology and Medicine},
  volume  = {116},
  pages   = {103545},
  year    = {2020},
  doi     = {10.1016/j.compbiomed.2019.103545}
}
```

### Other methods used in the code

These are not distributed here, but the code implements or compares against them. Cite them if you
use the corresponding parts:

- **Mamba**: Gu, A., Dao, T. *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.* COLM 2024.
- **VMamba** (the SS2D design MedMamba inherits): Liu, Y. et al. *VMamba: Visual State Space Model.* NeurIPS 2024.
- **HybridSN**: Roy, S. K. et al. IEEE GRSL 17(2):277–281, 2020. [doi:10.1109/LGRS.2019.2918719](https://doi.org/10.1109/LGRS.2019.2918719)
- **SpectralFormer**: Hong, D. et al. IEEE TGRS 60:5518615, 2022. [doi:10.1109/TGRS.2021.3130716](https://doi.org/10.1109/TGRS.2021.3130716)

---

## Third-party material and licences

| material | how it is used here | licence / terms | what you must do |
|---|---|---|---|
| [MedMamba](https://github.com/YubiaoYue/MedMamba) | design basis for `MedMambaSS` (SS2D selective scan, SS-Conv-SSM blocks, hierarchical stages), re-implemented in pure PyTorch; the original code is the MedMamba baseline | the repository provides no licence file (checked 2026-10-04) | cite `yue2024medmamba`; for reuse of the original code beyond citation, contact its authors |
| [TinyRecursiveModels](https://github.com/SamsungSAILMontreal/TinyRecursiveModels) | recursive scheme of `MedMambaSSTRM`; `medmamba_ss_trm_ema.py` is ported from its `models/ema.py` | MIT (Copyright (c) 2025 Samsung Electronics Co., Ltd.) | cite TRM **and** HRM; the MIT notice is reproduced at the top of `medmamba_ss_trm_ema.py` and must stay with that code |
| [HistologyHSI-BC-Recurrence](https://www.cancerimagingarchive.net/collection/histologyhsi-bc-recurrence/) | primary dataset (not redistributed) | CC BY 4.0 + [TCIA Data Usage Policy](https://www.cancerimagingarchive.net/data-usage-policies-and-restrictions/) | cite the dataset DOI (10.7937/6KPY-YT49), acknowledge TCIA, and cite Clark et al. (2013) and the data descriptor |
| [PAD-UFES-20](https://data.mendeley.com/datasets/zr7vgbcyr2/1) | RGB dataset (not redistributed) | CC BY 4.0 | cite the dataset DOI (10.17632/zr7vgbcyr2.1) and its data article |

The datasets contain de-identified patient data. Use them only as their terms allow, and do not try to
re-identify individuals.

## License

No licence has been chosen yet for the original code in this repository. Until one is added, all
rights are reserved by the authors. Third-party components keep their own licences, listed above.

## Acknowledgements

We thank the authors of MedMamba, the Tiny Recursive Model and the Hierarchical Reasoning Model for
releasing their code. We also thank the creators of HistologyHSI-BC-Recurrence, The Cancer Imaging
Archive, and the PAD-UFES-20 team at the Federal University of Espírito Santo for making their data
publicly available.
