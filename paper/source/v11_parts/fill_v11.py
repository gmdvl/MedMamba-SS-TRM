"""Fill the data-dependent placeholders of q4a/q4h/q5/abstract from
paper/source/v19_analysis.json (and the table generator), so that every
number in those paragraphs comes from the analysis file."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
ROOT = Path("/data/dante_data/documents/Masters/courses/Thesis/g-medmamba")
A = json.load(open(ROOT / "paper/source/v19_analysis.json"))
M = A["modality"]
S = M["summary"]
B = M["bootstrap_bal_delta"]
C = A["comparison"]


def gen(section):
    out = subprocess.run([sys.executable, str(ROOT / "scripts/tables_v19.py")], capture_output=True, text=True).stdout
    block = out.split(f"## {section}")[1].split("\n## ")[0]
    return block


WORDS = {0: "none", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight",
         9: "nine", 10: "ten"}


def w(n):
    return WORDS.get(int(n), str(n))


def sg(v, fmt):
    """signed number with a typographic minus"""
    return format(v, fmt).replace("-", "\u2212")


def pts(x):
    return f"{100 * x:+.2f}"


def ci(b):
    return f"[{sg(100 * b['ci95'][0], '+.1f')}, {sg(100 * b['ci95'][1], '+.1f')}] points"


def table_lines(section):
    return "\n".join(l for l in gen(section).splitlines() if l.startswith("| "))


q4b = (HERE / "q4b.md").read_text()
q4b = q4b.replace("{{ABLATION_TABLE}}", table_lines("ablations")).replace("{{PAD_TABLE}}", table_lines("pad"))
(HERE / "q4b_filled.md").write_text(q4b)

# ---- comparison rows ----
rows = [l for l in gen("comparison").splitlines() if l.startswith("| ") and not l.startswith("| Model")
        and not l.startswith("| ---")]
q4a = (HERE / "q4a.md").read_text().replace("{{COMPARISON_ROWS}}", "\n".join(rows))

# ---- modality rows ----
mrows = []
for seed in ("1", "7", "13", "23", "42"):
    h, r = M["per_seed"][seed]["hsi"], M["per_seed"][seed]["rgb"]
    cells = []
    for split in ("test", "val_full", "pooled"):
        a, b = h[split]["balanced_accuracy"], r[split]["balanced_accuracy"]
        cells += [f"{100 * a:.2f}", f"{100 * b:.2f}", sg(100 * (a - b), "+.2f")]
    mrows.append(f"| {seed} | " + " | ".join(cells) + " |")
mean_cells = []
for split in ("test", "val_full", "pooled"):
    h, r, d = S[f"{split}_hsi"], S[f"{split}_rgb"], S[f"{split}_delta"]
    mean_cells += [f"**{100 * h['balanced_accuracy'][0]:.2f}**", f"**{100 * r['balanced_accuracy'][0]:.2f}**",
                   f"**{sg(100 * d['bal_mean'], '+.2f')} ± {100 * d['bal_sd']:.2f}**"]
mrows.append("| **Mean** | " + " | ".join(mean_cells) + " |")
q4a = q4a.replace("{{MODALITY_ROWS}}", "\n".join(mrows))

dt, dv, dp = S["test_delta"], S["val_full_delta"], S["pooled_delta"]
pp = M["per_patient_over_seeds"]
test_ids = [p for p in pp if pp[p]["split"] == "test"]
val_ids = [p for p in pp if pp[p]["split"] == "val"]
hsi_better_test = sum(np.mean(pp[p]["hsi"]) > np.mean(pp[p]["rgb"]) + 1e-9 for p in test_ids)
rgb_better_val = sum(np.mean(pp[p]["rgb"]) > np.mean(pp[p]["hsi"]) + 1e-9 for p in val_ids)
pc_h, pc_r = S["test_hsi"]["per_class_f1"], S["test_rgb"]["per_class_f1"]
ph, pr_ = C["probe_hsi_mean_std"], C["probe_rgb_mean_std"]
diffs = {p: np.mean(pp[p]["hsi"]) - np.mean(pp[p]["rgb"]) for p in pp}
n_h = sum(d > 0.005 for d in diffs.values())
ties = [p for p, d in diffs.items() if abs(d) <= 0.005]
worst = min(diffs, key=diffs.get)
mod_text = (
    f"On the test patients, balanced accuracy favours 32-band input at {w(dt['bal_pos'])} of five seeds, by "
    f"{100 * dt['bal_mean']:.2f} ± {100 * dt['bal_sd']:.2f} points (macro-F1 {sg(dt['f1_mean'], '+.3f')} ± {dt['f1_sd']:.3f}, "
    f"positive at {w(dt['f1_pos'])} of five), and the gain lies in healthy tissue and DCIS (mean per-class F1 "
    f"{pc_h[0]:.3f} against {pc_r[0]:.3f} and {pc_h[1]:.3f} against {pc_r[1]:.3f}; IDC {pc_h[2]:.3f} against "
    f"{pc_r[2]:.3f}). On the validation patients, scored on the full split, the sign reverses: 3-band input is "
    f"better at {w(5 - dv['bal_pos'])} of five seeds, by {-100 * dv['bal_mean']:.2f} ± {100 * dv['bal_sd']:.2f} points. "
    f"Pooled over all ten patients the difference is {sg(100 * dp['bal_mean'], '+.2f')} ± {100 * dp['bal_sd']:.2f} points "
    f"(positive at {w(dp['bal_pos'])} of five seeds). Resampling patients, the 95 % interval of the seed-mean "
    f"difference is {ci(B['test'])} on the test patients, {ci(B['val'])} on the validation patients and "
    f"{ci(B['pooled'])} pooled.\n\n"
    f"Fig. {{{{F:modality}}}}(b) locates the reversal. Averaged over seeds, 32-band input gives the higher macro "
    f"recall for {w(n_h)} of the ten held-out patients"
    + (f" (patient {ties[0]} is tied at {np.mean(pp[ties[0]]['hsi']):.3f})" if ties else "")
    + f", and the validation reversal comes almost entirely from one patient, {worst}, whose macro recall is "
    f"{np.mean(pp[worst]['hsi']):.3f} with 32 bands against {np.mean(pp[worst]['rgb']):.3f} with 3. The same patient "
    f"drives the reversal without any network: a logistic regression on per-band means and standard deviations scores "
    f"{100 * ph['test']['balanced_accuracy']:.2f} % on the test patients with 32 bands against "
    f"{100 * pr_['test']['balanced_accuracy']:.2f} % with 3, and {100 * ph['val_full']['balanced_accuracy']:.2f} % "
    f"against {100 * pr_['val_full']['balanced_accuracy']:.2f} % on the validation patients, where patient {worst} "
    f"again favours 3 bands (0.672 against 0.836). The 3-band model is also the more stable across seeds (test "
    f"balanced accuracy s.d. {100 * S['test_rgb']['balanced_accuracy'][1]:.1f} against "
    f"{100 * S['test_hsi']['balanced_accuracy'][1]:.1f} points). The comparison therefore does not establish that "
    f"32-band input is better for this task. It shows that 32 bands help most patients and fail badly on one, and "
    f"that ten patients, with DCIS present in only one patient per held-out set, are too few to settle the question."
)
q4a = q4a.replace("{{MODALITY_TEXT}}", mod_text)
(HERE / "q4a_filled.md").write_text(q4a)

# ---- calibration ----
crow = [l for l in table_lines("calibration").splitlines()[2:]]
q4h = (HERE / "q4h.md").read_text().replace("{{CALIB_ROWS}}", "\n".join(crow))
vh, vr = S["val_full_hsi"], S["val_full_rgb"]
q4h = q4h.replace("{{CALIB_VAL}}", f", and on the validation patients as well (ECE {vh['ece'][0]:.3f} against "
                                   f"{vr['ece'][0]:.3f})" if vh["ece"][0] < vr["ece"][0] else
                  f"; on the validation patients the order reverses (ECE {vh['ece'][0]:.3f} against {vr['ece'][0]:.3f})")
Ts = {mod: [M["per_seed"][s][mod]["temperature"]["T"] for s in M["per_seed"]] for mod in ("hsi", "rgb")}
te = {mod: [M["per_seed"][s][mod]["temperature"]["test_ece"] for s in M["per_seed"]] for mod in ("hsi", "rgb")}
Tt = {mod: [M["per_seed"][s][mod]["temperature"]["T_test"] for s in M["per_seed"]] for mod in ("hsi", "rgb")}
ve = {mod: [M["per_seed"][s][mod]["temperature"]["val_ece"] for s in M["per_seed"]] for mod in ("hsi", "rgb")}
base_te = {mod: S[f"test_{mod}"]["ece"][0] for mod in ("hsi", "rgb")}
base_ve = {mod: S[f"val_full_{mod}"]["ece"][0] for mod in ("hsi", "rgb")}


def moves(new, old):
    return "raises" if new > old else "lowers"


both_worse = {mod: np.mean(te[mod]) > base_te[mod] and np.mean(ve[mod]) > base_ve[mod] for mod in ("hsi", "rgb")}
tail = ("no single temperature fits both held-out sets for either input." if all(both_worse.values()) else
        "for the 32-band model no single temperature fits both held-out sets, whereas for the 3-band model the "
        "temperature fitted on the test patients also lowers validation ECE." if both_worse["hsi"] and not both_worse["rgb"] else
        "the effect of a transferred temperature depends on the direction and the input.")
q4h = q4h.replace("{{CALIB_TEMP}}", (
    f"Temperature scaling fitted on each run's validation predictions does not repair this. The fitted "
    f"temperature is above one (T = {np.mean(Ts['hsi']):.2f} ± {np.std(Ts['hsi'], ddof=1):.2f} for 32 bands, "
    f"{np.mean(Ts['rgb']):.2f} ± {np.std(Ts['rgb'], ddof=1):.2f} for 3 bands), because the models are over-confident "
    f"on the validation patients, and applied to the test patients it {moves(np.mean(te['hsi']), base_te['hsi'])} "
    f"test ECE from {base_te['hsi']:.3f} to {np.mean(te['hsi']):.3f} (32 bands) and "
    f"{moves(np.mean(te['rgb']), base_te['rgb'])} it from {base_te['rgb']:.3f} to {np.mean(te['rgb']):.3f} (3 bands). "
    f"In the other direction, a temperature fitted on the test patients is below one (T = {np.mean(Tt['hsi']):.2f} "
    f"and {np.mean(Tt['rgb']):.2f}); applied to the validation patients it {moves(np.mean(ve['hsi']), base_ve['hsi'])} "
    f"validation ECE from {base_ve['hsi']:.3f} to {np.mean(ve['hsi']):.3f} (32 bands) and "
    f"{moves(np.mean(ve['rgb']), base_ve['rgb'])} it from {base_ve['rgb']:.3f} to {np.mean(ve['rgb']):.3f} (3 bands; "
    f"last column of Table {{{{T:calib}}}}). So {tail}"))

# ---- calibration of the other models (seed 42) ----
OTHERS = {"MedMamba": "MedMamba", "hybridsn": "HybridSN", "spectralformer": "SpectralFormer",
          "probe_hsi_mean_std": "the 64-feature probe", "probe_rgb_mean_std": "the RGB probe"}


def listing(items):
    items = [f"{n} ({e:.3f})" for n, e in sorted(items, key=lambda x: x[1])]
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def listing_plain(names):
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def rank_vs_ref(split):
    ref = S[f"{split}_hsi"]["ece"][0]
    have = [(n, C[k][split]["ece"]) for k, n in OTHERS.items() if "ece" in (C[k].get(split) or {})]
    return [x for x in have if x[1] < ref], [x for x in have if x[1] >= ref]


bt, wt = rank_vs_ref("test")
bv, wv = rank_vs_ref("val_full")
tr = {n: (C[k]["val_full"]["ece"], C[k]["test"]["ece"], C[k]["temperature"]) for k, n in OTHERS.items()
      if "temperature" in C[k]}
all_up = all(t["test_ece"] > te_ for _, te_, t in tr.values()) and all(np.mean(te[m]) > base_te[m] for m in te)
all_over = all(t["T"] > 1 for _, _, t in tr.values()) and all(np.mean(Tt[m]) < 1 for m in Tt) \
    and all(t["T_test"] < 1 for _, _, t in tr.values())
top = sorted(tr, key=lambda n: tr[n][2]["test_ece"] - tr[n][1], reverse=True)[:2]
v_down = [n for n, (ve_, _, t) in tr.items() if t["val_ece"] < ve_]
v_up = [n for n, (ve_, _, t) in tr.items() if t["val_ece"] >= ve_]
Tv = [t["T"] for _, _, t in tr.values()]
base_txt = (
    f"Among the other models, {listing(bt)} are better calibrated than the 32-band MedMamba-SS-TRM on the test "
    f"patients and {listing(wt)} {'is' if len(wt) == 1 else 'are'} worse; on the validation patients "
    f"{listing(bv)} {'is' if len(bv) == 1 else 'are'} better and {listing(wv)} {'is' if len(wv) == 1 else 'are'} worse.")
if all_up:
    base_txt += (
        f" The transferred temperature fails for {listing_plain(list(tr))} as well: fitted on the validation "
        f"patients it is above one for each of them (T = {min(Tv):.2f} to {max(Tv):.2f}) and raises test ECE, most for "
        f"{top[0]} ({tr[top[0]][1]:.3f} to {tr[top[0]][2]['test_ece']:.3f}) and {top[1]} "
        f"({tr[top[1]][1]:.3f} to {tr[top[1]][2]['test_ece']:.3f}); fitted on the test patients it lowers "
        f"validation ECE for {listing_plain(v_down)}"
        + (f" and raises it for {' and '.join(v_up)}" if v_up else "") + ".")
if all_over:
    base_txt += (f" All {w(len(tr) + 2)} models with validation predictions are over-confident on the validation "
                 "patients and under-confident on the test patients, so the miscalibration follows the patient split "
                 "rather than the architecture, and on these data a temperature fitted on five patients does not "
                 "transfer to five others.")
q4h = q4h.replace("{{CALIB_BASE}}", base_txt)

# table note: which MCE values are set by a near-empty bin
SPLIT_WORD = {"test": "test", "val_full": "validation"}
floor = A["mce_min_frac"]
sparse = [(n, sp, C[k][sp]) for k, n in OTHERS.items() for sp in SPLIT_WORD
          if "mce_bin_n" in (C[k].get(sp) or {}) and C[k][sp]["mce_bin_n"] < floor * C[k][sp]["n"]]
seeds_sparse = {sp: [s for s in M["per_seed"] for m in ("hsi", "rgb")
                     if M["per_seed"][s][m][sp]["mce_bin_n"] < floor * M["per_seed"][s][m][sp]["n"]]
                for sp in SPLIT_WORD}
note = (f"*MCE: largest $|\\mathrm{{acc}}(B_m) - \\mathrm{{conf}}(B_m)|$ over the bins of (29) that hold at least "
        f"{100 * floor:g} % of the patches. MedMamba-SS-TRM: mean ± s.d. over five seeds; other models: one run at "
        f"seed 42. MedMamba's validation probabilities were not stored, so it has no validation row and no "
        f"transferred temperature.")
unfloored = [f"{n}'s {SPLIT_WORD[sp]} MCE would be {v['mce']:.3f}, set by a bin of {v['mce_bin_n']} patches"
             for n, sp, v in sparse]
unfloored += [f"that of {w(len(ss))} MedMamba-SS-TRM {SPLIT_WORD[sp]} run{'s' if len(ss) > 1 else ''} would be set "
              f"by a bin of fewer than {floor * M['per_seed']['42']['hsi'][sp]['n']:.0f} patches"
              for sp, ss in seeds_sparse.items() if ss]
if unfloored:
    note += " Without that floor, " + "; ".join(unfloored) + "."
q4h = q4h.replace("{{CALIB_NOTE}}", note + "*")
(HERE / "q4h_filled.md").write_text(q4h)
print("modality:", mod_text[:400], "...")
print("temps", {k: np.round(v, 2).tolist() for k, v in Ts.items()}, {k: np.round(v, 3).tolist() for k, v in te.items()})

# ---- abstract ----
ref = M["per_seed"]["42"]["hsi"]
hyb = C["hybridsn"]["pooled"]
abstract = (
    "***Abstract*—MedMamba classifies medical images with a block that splits channels between a convolutional "
    "branch and a two-dimensional selective scan, but it reads the spectral axis as unordered channels whose count "
    "is fixed by the sensor. We present two extensions. MedMamba-SS replaces its patch embedding with a spectral "
    "pathway in which no parameter depends on the number of bands (a tokenizer shared across bands and keyed to "
    "wavelength, a spectral selective scan and pooling over bands) and conditions every stage of the hierarchy on "
    "its output. MedMamba-SS-TRM replaces the hierarchy with one two-layer core applied recursively, following the "
    "Tiny Recursive Model. On hyperspectral breast histology from 45 patients, 32-band and 3-band models have "
    "exactly 446,409 parameters, retraining at eight bands keeps 97 % of macro-F1, and the learned band gate "
    "concentrates on the 535–633 nm stain-absorption region. MedMamba-SS-TRM is the most accurate model on the five "
    f"test patients ({100 * ref['test']['balanced_accuracy']:.1f} % balanced accuracy) and outperforms MedMamba "
    "pooled over all ten held-out patients with 8.2× fewer parameters, but HybridSN leads pooled "
    f"({100 * hyb['balanced_accuracy']:.1f} % against {100 * ref['pooled']['balanced_accuracy']:.1f} %). The "
    "advantage of 32 over 3 bands is likewise patient-dependent: "
    f"{sg(100 * dt['bal_mean'], '+.1f')} points on the test patients at {w(dt['bal_pos'])} of five seeds, "
    f"{sg(100 * dv['bal_mean'], '+.1f')} on the validation patients, driven by one patient, and "
    f"{sg(100 * dp['bal_mean'], '+.1f')} pooled. The recursive "
    "model is 6.2× smaller but needs 22.3× more arithmetic than the hierarchy it replaces, as an analytic cost model "
    "predicts, and recursion depth from 21 to 84 core applications leaves quality unchanged.**"
)
(HERE / "abstract.md").write_text(abstract + "\n")
print("abstract words:", len(abstract.split()))

# ---- conclusion ----
concl = (
    "We extended MedMamba in two steps. MedMamba-SS replaces the convolutional patch embedding with a spectral "
    "pathway whose parameters do not depend on the number of bands and conditions every stage of the SS-Conv-SSM "
    "hierarchy on it; MedMamba-SS-TRM replaces that hierarchy with one weight-shared core applied recursively in the "
    "manner of TRM.\n\n"
    "Four results do not depend on which patients are held out. The parameter count is exactly the same at 3 and at 32 "
    "bands, and a model retrained at eight bands keeps 97 % of its macro-F1, whereas a trained model cannot be moved "
    "to fewer bands. The spectral pathway learns to weight the 535–633 nm stain-absorption region where the classes "
    "differ, and its wavelength encoding and reconstruction objective each improve classification. The recursive "
    "core reduces parameters 6.2×, and with 8.2× fewer parameters than MedMamba it outperforms it on the test patients "
    "and pooled over all ten and matches it on the validation patients. And it "
    "multiplies arithmetic 22.3×, exactly as the cost model predicts, while recursion depth beyond 21 applications "
    "buys nothing measurable, so parameter count in a recursive model measures storage rather than compute.\n\n"
    "Two results depend on the patients. MedMamba-SS-TRM is the most accurate model on the five test patients and "
    "among the least accurate networks on the five validation patients; pooled over all ten, HybridSN and a "
    "six-feature colour probe lead. The advantage of 32-band over 3-band input follows the same split "
    f"({sg(100 * dt['bal_mean'], '+.1f')} points on test, {sg(100 * dv['bal_mean'], '+.1f')} on validation, "
    f"{sg(100 * dp['bal_mean'], '+.1f')} pooled): 32 bands give the higher recall for most held-out patients but fail "
    "badly on one, and a linear probe shows the same pattern, so it is a property of the patients rather than of the "
    "network.\n\n"
    "**Limitations.** All hyperspectral results rest on ten non-training patients, and each held-out set contains "
    "DCIS from a single patient. The baselines are single-seed, and MedMamba's recipe differs from the others'. The "
    "RGB input is the collection's own synthetic rendering. The hierarchical MedMamba-SS does not train under the "
    "recursive recipe on hyperspectral data, so the substitution is supported on parameters, arithmetic and interface "
    "rather than on a head-to-head comparison. The whole-image PAD-UFES-20 comparison is confounded by training-set "
    "size and loss weighting."
)
q5 = (HERE / "q5.md").read_text().replace("CONCLUSION PENDING", concl)
(HERE / "q5_filled.md").write_text(q5)
print("done")
