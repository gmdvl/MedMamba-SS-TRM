# `training/prep/` — Generic Dataset-Preparation Core (+ v7 / v6 adapters)

> **New in this revision.** `prepare_histologyhsi_bc_v6.py` and
> `prepare_pad_ufes_20_v5.py` stay **frozen** for reproducing existing datasets.
> `prepare_histologyhsi_bc_v7.py` and `prepare_pad_ufes_20_v6.py` are thin
> **adapters** over `training/prep/` — same on-disk contract, faster, more
> flexible splits.

---

## 1. Why

The two v5/v6 prep scripts each copy-pasted ~500 lines of pipeline shell
(`ShardWriter`, `_progress.json` lifecycle, `unify_all`, `main`) and ran
**single-threaded**, `save_manifest()`-per-item (fsync storm),
one-shard-per-image, `gc.collect()` per flush, and re-read the whole dataset
~3× at finalize. `training/prep/` is the shared, parallel, dataset-agnostic
pipeline; a `DatasetAdapter` supplies only the format-specific bits.

## 2. Package layout

| module | responsibility |
|---|---|
| `training/prep/adapter.py` | `PrepItem`, `ModalitySpec`, `DatasetAdapter` ABC |
| `training/prep/core.py` | `run_prep(adapter, argv)` — orchestration + shared CLI |
| `training/prep/parallel.py` | `ProcessPoolExecutor` driver, worker, `--writer direct` |
| `training/prep/shard_writer.py` | `ShardWriter` (spans items; `do_fsync=False`; no per-flush GC) |
| `training/prep/manifest.py` | `_progress.json` v2 lifecycle, `cleanup_orphan_shards` |
| `training/prep/splits.py` | resolve `--split*` → group sets; `split_assignment.json` |
| `training/prep/dryrun.py` | `--dry_run` report |

## 3. The adapter contract

```python
class DatasetAdapter(ABC):
    name: str
    flat_output: bool = False          # True -> arrays in <out>/ ; False -> <out>/<modality>/

    class_names -> list[str]
    modalities() -> list[str]                       # default ["image"]
    add_cli_args(parser)                            # register adapter flags
    preprocess_args(args)                           # normalise flags (e.g. --test_size -> --split)
    compat_keys(args) -> dict                       # settings locked into _progress.json

    discover(args) -> Iterable[PrepItem]            # cheap; no pixel data
    pass1(items, args) -> dict | None               # optional global pre-pass (band selection)
    write_pass1_outputs(state, out_dir) -> [names]  # e.g. selected_band_indices.npy
    finalize_modality(mod, mod_dir, state, manifest) -> [names]   # e.g. wavelengths.npy

    modality_spec(mod, args, state) -> ModalitySpec # store_dtype / value_clip / sample_shape
    load_item(item, mod, args, state) -> Iterator[(HWC_float32, label)]     # runs in a worker
    load_item_multi(item, mods, args, state) -> {mod: iterator} | None      # share per-item work
    expected_sample_count(item, mod, args, state) -> int | None             # enables --writer direct
```

`PrepItem(key, group_id, label, payload, strat_keys=())` — `key` is the
resume + deterministic-order id; `group_id` is the split grouping key
(patient); `payload` is small & picklable (paths, not arrays).

## 4. Pipeline (`run_prep`)

```
discover → compute split (splits.compute_split) → write split_assignment.json
 → pre-extraction class-coverage gate → adapter.pass1 → new _progress.json
 → cleanup_orphan_shards
 → ProcessPoolExecutor: per item → adapter.load_item[_multi] → ShardWriter
      (parent checkpoints _progress.json every --manifest_flush_every items / seconds,
       and on exit/signal; one bad item → SkipRegistry, run continues)
 → deterministic-order shard sort (by item key)
 → realized class-coverage re-check
 → per (modality, split): unify_split(sha_out=…) → finalize_dataset(precomputed_sha256=…, verify_level)
 → class_names.json, dataset_statistics.json
```

## 5. Shared CLI (all adapters)

| flag | default | notes |
|---|---|---|
| `--split` | `80_20` | preset **or** `75/15/10` / `80/20` / `0.7/0.15/0.15` |
| `--split_strategy` | `stratified` | or `legacy_random` |
| `--split_file PATH` | — | reuse a frozen `split_assignment.json` (across modalities/datasets) |
| `--kfold K --fold I` | — | grouped, class-stratified CV; `--cv_val_frac`, `--cv_repeat` |
| `--seed` | `42` | (v7 changes HSI's old default of 0) |
| `--num_workers` | `min(cpu, 8)` | `0`/`1` = inline |
| `--batch_size` | `256` | samples per shard (not locked on resume) |
| `--store_dtype` | `auto` | `auto` = adapter's `modality_spec`; else `float32`/`float16`/`uint8` |
| `--balance_classes` | `none` | `undersample`/`oversample`, TRAIN split only, at unify |
| `--writer` | `shards` | `direct` (experimental): write final arrays, no shards; aborts on any skip |
| `--deterministic_order` | on | shard order = item-key order, independent of worker completion |
| `--verify_level` | `auto` | `auto` = deep if dataset < `--deep_verify_max_gb` (4) else `probe`; also `structural` |
| `--manifest_flush_every` / `--manifest_flush_seconds` | 64 / 30 | checkpoint cadence |
| `--dry_run`, `--limit N`, `--fresh`, `--finalize_only`, `--keep_batches` | | |

## 6. `_progress.json` v2

Superset of v5/v6: `train_patients` / `validation_patients` / `test_patients`
stay at the top level (external leakage-check + the old `training/data.py` readers, deleted 2026-10-01).
Added: `schema:2`, `adapter`, `split:{spec,strategy,split_algo_version,kfold_algo_version,…}`,
`pass1_state`, `completed_items` (`{key:{split, modalities:{mod:[[x,y,n]…]}, counts, extra}}`),
`skipped_items`. `check_manifest_compatible` locks the `args` block **except**
`batch_size` / worker / verify knobs (advisory only).

## 7. Shared-path speedups (`training/dataset_prep_unify.py`, `npy_integrity.py`)

- **`unify_split` identity fast path** — no `row_order`/balancing → stream each
  shard's whole payload into consecutive rows via `copy_npy_payload`
  (chunked `readinto`, SIGBUS-safe; no per-row seek, no O(out_n·n_shards) scan).
- **Folded sha256** — `atomic_array_writer(sha256=True)` hashes while writing;
  `unify_split(sha_out=…)` returns the digest; `finalize_dataset(precomputed_sha256=…)`
  skips the separate hash read. Finalize goes from ~3 full passes → ~1.
- **`verify_level="probe"`** — structural + the 64-row all-zero/truncation
  probe, no end-to-end read. `deep` still available.
- These are additive; v5/v6 keep their old defaults and byte-for-byte output.

## 8. Split options (`training/dataset_prep_common.py`)

`parse_split_spec`, `resolve_split_fractions` (now also accepts specs),
`grouped_stratified_kfold` + `kfold_split` (`KFOLD_ALGO_VERSION = 1`),
`stratified_group_split_three` (multi-key strat; delegates to the plain
3-way split when no extra keys), `write_split_assignment` /
`load_split_assignment`. `SPLIT_ALGO_VERSION` and the plain stratified 3-way
split are unchanged (byte-identical per seed).

## 9. Adapter specifics

**`prepare_pad_ufes_20_v6.py`** — `flat_output=True`, one `image` modality.
`load_item` adds `ImageOps.exif_transpose` (orientation) and JPEG
`Image.draft` (fast down-scale decode). `--store_dtype uint8` keeps raw 0-255
(loader min-max norm handles it); default float32 `/255`. `--test_size` kept
as a deprecated `--split` alias.

**`prepare_histologyhsi_bc_v7.py`** — `flat_output=False`, `hsi` / `rgb`
modalities; **delegates every HSI operation to the frozen
`prepare_histologyhsi_bc_v6` helpers** (`discover_captures`, `open_envi_lazy`,
`calibrate_patch`, `get_calibration_means`, ROI, `patch_coords`,
`compute_band_statistics`, `select_bands*`) — no logic fork. `pass1` probes
sampled cubes for wavelengths + a calibration/value-range check; HSI
`store_dtype` defaults to **float16** only when the sampled cubes are
calibrated and in range (else float32), with `ShardWriter`'s
`Float16OverflowError` as a hard backstop. RGB keeps v6's raw-0-255 float32
(opt-in `--rgb_normalize`). `wavelengths.npy` / `wavelengths_full.npy` are
written per modality by `finalize_modality`.

**The two adapters you actually run** are one generation further on and are documented in
[`11_v16_optimal_and_reconstruction.md`](11_v16_optimal_and_reconstruction.md) §2–§3, not
here: `prepare_histologyhsi_bc_v8.py` (subclasses v7 — adds `--capture_gain`,
`--hsi_value_scale`, `--emit_group_sidecars`, `--rgb_source`) and
`prepare_pad_ufes_20_optimal.py` (subclasses v6 — `--tiling`, a three-way `70/15/15`
default). Everything in §2–§8 above is the core they both inherit and is unchanged.

## 10. Tests

`test_dataset_prep_unify.py` / `test_dataset_prep_common.py` (extended),
`test_prep_core.py` (fake adapter: contract, resume-after-partial, skip
fail-soft, worker-count invariance, dry-run, `--writer direct` parity),
`test_prep_pad_fixture.py` (synthetic PNGs), `test_prep_hsi_fixture.py`
(synthetic ENVI cubes; skipped without `spectral`).

## 11. Verification

```bash
# PAD: speed + size, then load-check
time python prepare_pad_ufes_20.py --root <pad> --out_dir /tmp/pad_v6 \
     --split 80/10/10 --num_workers 8 --patch_size 11
python -c "from training.npy_integrity import validate_dataset as v; \
  v({'X_train':'/tmp/pad_v6/X_train.npy','y_train':'/tmp/pad_v6/y_train.npy'}, level='deep')"

# HSI: float16 halves on-disk size vs the v6 dataset
time python prepare_histologyhsi_bc_v7.py --root <hsi> --out_dir /tmp/hsi_v7 \
     --band_selection importance --num_bands 32 --num_workers 8
du -sh /tmp/hsi_v7/hsi

# (v13 is the entry point this section was written against; it now lives in archive/.
#  The current equivalent is:)
python train_example_v16.py --data_dir /tmp/hsi_v7/hsi --manifest_check strict \
     --dataset_validation_level deep --epochs 1
```


---

## v17 — progress, ETA, and a parallel pass 1

`training/prep/` is FROZEN in its entirety (`test_frozen_files_untouched.py:
FROZEN_DIR_PREFIXES`), so neither change below edits it.

**Rate and ETA on extraction.** `core.py:395-402` printed `412/644 done (410 ok, 2
skipped)` on the manifest-flush cadence, and the line immediately above it computes

```python
rate = state_box["n"] / max(1e-6, now - state_box.get("t0", now))
```

which it then never uses — the one number that says whether a prep has five minutes or two
hours left was calculated and thrown away. `training/prep_progress.py` rebinds
`training.prep.core.run_extraction` to a wrapper that decorates the caller's `on_result`
with a `ProgressReporter`. This is the same module-attribute rebinding
`prepare_histologyhsi_bc_v8.py:400` already uses for `_v6._find_rgb_image`, and for the
same reason: the behaviour has to change without the frozen file changing. The frozen
callback is still called first and unchanged — it owns the manifest, the skip registry and
the `FatalItemError` `SystemExit` — and the wrapper only counts. It is idempotent, and a
no-op if the frozen signature ever moves.

Installed from the `main()` of `prepare_histologyhsi_bc_v8.py` and
`prepare_pad_ufes_20_optimal.py`.

**Pass 1 runs in parallel.** `prepare_histologyhsi_bc_v8.pass1` walked all 644 captures
serially in the parent process while pass 2 had used a process pool since v6
(`parallel.py:104`), and `_sample_capture_median` reads pixels one at a time in a Python
loop — ~400 random seeks per capture, ~257,600 single-pixel ENVI reads for the corpus.

It is now dispatched over `--num_workers`, and this is **exactly value-preserving**, which
is the only reason it was safe to do: `_sample_capture_median` is called with
`seed=args.seed` — the *same* seed for every capture — so each capture's RNG stream is
already independent of the others and of iteration order. The gain loop additionally
iterates `sorted(medians.items())`, because the pool fills the dict in completion order and
`capture_gain_suspects` is written to `capture_gain_report.json`; the values were already
order-independent, and this makes the report byte-identical too.

Verify after any change here:

```bash
python prepare_histologyhsi_bc.py --root <...> --out_dir /tmp/chk1 --limit 20 --num_workers 1 --allow_missing_classes
python prepare_histologyhsi_bc.py --root <...> --out_dir /tmp/chk8 --limit 20 --num_workers 8 --allow_missing_classes
diff <(python -c "import json;print(json.load(open('/tmp/chk1/hsi/capture_gain_report.json'))['per_capture'])") \
     <(python -c "import json;print(json.load(open('/tmp/chk8/hsi/capture_gain_report.json'))['per_capture'])")
```
