# -*- coding: utf-8 -*-
"""
scripts/gpu_tune.py
=========================
Measure what a MedMamba-SS-TRM configuration actually costs on THIS GPU, without
training a run, so `--batch_size`, `--target_token_grid` and `--compile` are
chosen from numbers instead of from a linear extrapolation.

Why
---
Every run in `experiments/` uses 1-9% of the 16 GB card
(`GPU_memory_MB` in `history.json`: 874 MB at the 11x11 HSI grid and batch
256; 1,432 MB at 28x28 and batch 32). The two cost models in the repository
disagree with each other and with the measurements:
`train_example_v16_original.print_trm_cost` extrapolates from a single fp32
OOM and reads ~2x LOW at the 28x28 grid, while
`train_example_v16_optimal.print_gpu_budget` is fitted to one measured point
and is only as good as that fit away from it. Both are guesses. This script
measures.

It is also the only way to answer the `--compile` question. `--compile`
defaults to `off`, and this workload is the shape `torch.compile` is for:
the recursive core is applied ~63 times per forward over small fixed-shape
tensors, so the step is dominated by kernel-launch overhead rather than by
arithmetic - exactly what Inductor fuses away. Nobody has benchmarked it
here. One run of this script settles it.

What it does
------------
For each cell of the sweep it builds the REAL model through
`training.config_presets.build_model` (the same call `train_example_v16.py`
makes, with the same `trm_kwargs`), runs `--warmup` discarded training steps
and then `--iters` timed ones on synthetic data of the requested shape, and
reports median ms/step and `torch.cuda.max_memory_allocated`. An OOM is
caught and reported as a row, not as a crash, so a sweep that walks off the
card still prints everything below the cliff.

Nothing is written to `experiments/` and no checkpoint is touched: this
allocates a fresh model per cell and throws it away.

Usage
-----
    # the two knobs with real headroom, on the PAD whole-image input
    python scripts/gpu_tune.py --input_side 224 \
        --batch_size 16,32,64 --token_grid 28,56

    # does torch.compile pay for itself?
    python scripts/gpu_tune.py --input_side 11 --batch_size 256 \
        --token_grid 11 --compile off,on

    # is gradient checkpointing still worth it with 90% of the card idle?
    python scripts/gpu_tune.py --input_side 224 --batch_size 32 \
        --token_grid 28 --checkpoint_core on,off

    # v17 S12 - BOTH checkpointing schemes at once, on the HSI prep. This is the
    # table that decides --no_trm_checkpoint_core and --spectral_checkpointing off:
    # every HSI run uses ~836 MB of a 16,706 MB card while paying for two separate
    # memory-saving schemes. Expect the spectral_ckpt=off cells to be the ones that
    # can OOM (measured at ~10,243 MB at batch 256, before the recon decoder).
    python scripts/gpu_tune.py --input_side 11 --in_channels 32 --num_classes 3 \
        --token_grid 11 --batch_size 128,256 \
        --checkpoint_core on,off --spectral_checkpointing on,off --compile on

Read the result as a rate, not an absolute: the numbers include the
synthetic-data path, not your DataLoader, so a real epoch is this plus I/O.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from training.config_presets import build_model, v15_config_overrides
from training.torch_compile import enable_torch_compile
from training.spectral_checkpoint import enable_spectral_gradient_checkpointing


_COMPILE_SUMMARY_SEEN = set()


def _csv(text, cast=str):
    return [cast(v.strip()) for v in str(text).split(",") if v.strip()]


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Measure ms/step and peak VRAM for MedMamba-SS-TRM configurations on this GPU.")
    p.add_argument("--input_side", type=int, default=224,
                   help="H=W of one sample as stored on disk: 224 for PAD whole images, 112 for "
                        "--tiling mil9 tiles, 11 for the published patch grid.")
    p.add_argument("--in_channels", type=int, default=3, help="3 for RGB, 32 for the HSI prep.")
    p.add_argument("--num_classes", type=int, default=6)
    p.add_argument("--batch_size", default="32", help="comma-separated, e.g. 16,32,64")
    p.add_argument("--token_grid", default="28",
                   help="comma-separated target tokens per side. The stem stride is "
                        "max(1, input_side // token_grid), the same rule "
                        "train_example_v16_optimal.resolve_patch_size uses.")
    p.add_argument("--compile", dest="compile_model", default="off", help="off / on / off,on")
    p.add_argument("--spectral_checkpointing", default="auto",  # comma-separated, e.g. off,on
                    help="v17 S3: on / off / auto, mirroring train_example_v16.py:573-576 "
                         "(auto = on when in_channels >= 16). Before this flag existed the "
                         "script NEVER enabled it, while the real training path does on HSI - "
                         "which is why the 11x11x32 rows read 10,157 MB against the 874 MB the "
                         "production run actually recorded, and why batch 512 'OOM'd here.")
    p.add_argument("--checkpoint_core", default="on",
                   help="on / off / on,off - gradient-checkpoint every RecursiveCore.f call "
                        "(cfg.trm_checkpoint_core). 'off' trades VRAM for speed, which is the "
                        "trade this card has headroom for.")
    p.add_argument("--amp", default="bf16", choices=["bf16", "fp16", "off"])
    p.add_argument("--architecture", default="recursive")
    p.add_argument("--modality", default=None, help="default: rgb when in_channels==3 else hsi")
    p.add_argument("--trm_dim", type=int, default=128)
    p.add_argument("--trm_mixer", default="mlp")
    p.add_argument("--trm_deep_supervision_steps", type=int, default=3)
    p.add_argument("--trm_n_latent", type=int, default=6)
    p.add_argument("--trm_n_improve", type=int, default=3)
    p.add_argument("--warmup", type=int, default=3, help="discarded steps (compile/autotune land here)")
    p.add_argument("--iters", type=int, default=10, help="timed steps; the median is reported")
    return p


def _resolve_spectral_ckpt(args, value: str = None) -> bool:
    """`--spectral_checkpointing auto` resolves exactly as train_example_v16.py:573-576.

    v17 S12 - `value` lets this axis be SWEPT ("off,on") rather than fixed for the whole
    table. It is the one remaining knob with real headroom: the flag's own help text
    (`train_example_v15.py:485`) measures it at +15% step time for 10,243 MB -> 1,007 MB at
    batch 256, and every run in `experiments/` uses 1-9% of a 16 GB card - so whether the
    memory it buys is worth the time is a question this script should be able to answer in
    one table instead of two invocations.
    """
    value = args.spectral_checkpointing if value is None else value
    if value == "on":
        return True
    if value == "off":
        return False
    return args.in_channels >= 16


def _make_model(args, patch_size: int, checkpoint_core: bool, device, spectral_ckpt: bool):
    modality = args.modality or ("rgb" if args.in_channels == 3 else "hsi")
    cfg_overrides = dict(v15_config_overrides(args.architecture))
    cfg_overrides["patch_size"] = patch_size
    trm_kwargs = None
    if args.architecture == "recursive":
        trm_kwargs = dict(
            trm_dim=args.trm_dim, trm_mixer=args.trm_mixer,
            trm_deep_supervision_steps=args.trm_deep_supervision_steps,
            trm_n_latent=args.trm_n_latent, trm_n_improve=args.trm_n_improve,
            trm_checkpoint_core=checkpoint_core,
        )
    model = build_model(args.architecture, modality, args.num_classes,
                        trm_kwargs=trm_kwargs, cfg_overrides=cfg_overrides).to(device)
    # v17 S3 - the real path enables this on HSI and this script never did, so every
    # HSI row it printed described a configuration no run uses. Applied BEFORE
    # enable_torch_compile, matching train_example_v16.py's order.
    if spectral_ckpt:
        enable_spectral_gradient_checkpointing(model, strict=True)
    return model


def _time_cell(args, batch, grid, compile_on, checkpoint_core, device, spectral_ckpt):
    """One sweep cell. Returns (ms_per_step, peak_mb) or raises the OOM."""
    patch_size = max(1, args.input_side // grid)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model = _make_model(args, patch_size, checkpoint_core, device, spectral_ckpt)
    if compile_on:
        # The repo's own compile path, not a bare `torch.compile(model)`.
        # `enable_torch_compile` targets `RecursiveCore._f_forward` and the
        # selective scan - the two hot loops - and rebinds them IN PLACE, so
        # the compiled code is still reached when the step below calls
        # `forward_deep_supervision` directly. Wrapping the module instead
        # would compile `forward`, which that call bypasses, and `--compile on`
        # would measure exactly nothing. This is also what
        # `train_example_v16.py:578` does, so the number transfers to a real run.
        # It returns a SUMMARY STRING and swallows failures ("FAILED (...) -
        # continuing uncompiled"). Unreported, a fallback would show up as a
        # `compile=on` row that was never compiled - i.e. a measurement saying
        # compile does nothing, when what happened is that it did not run.
        summary = enable_torch_compile(model, mode=None)
        if not _COMPILE_SUMMARY_SEEN or summary not in _COMPILE_SUMMARY_SEEN:
            _COMPILE_SUMMARY_SEEN.add(summary)
            print(f"  [compile] {summary}", flush=True)
        if "unavailable" in summary or "FAILED" in summary:
            raise RuntimeError(f"torch.compile did not engage: {summary}")
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

    # NCHW, matching what NpyDataset hands the trainer after its permute.
    x = torch.randn(batch, args.in_channels, args.input_side, args.input_side, device=device)
    y = torch.randint(0, args.num_classes, (batch,), device=device)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "off": None}[args.amp]
    lossf = torch.nn.CrossEntropyLoss()

    # Match what the trainer actually executes, or the timing is a fiction.
    # `MedMambaSSTRM.forward` runs every deep-supervision segment but
    # returns only `logits_list[-1]`, and the core detaches `(y, z)` between
    # segments - so a loss on that one tensor backprops through the LAST
    # segment only, and would under-measure the step by roughly
    # `trm_deep_supervision_steps`. `TrainerG_v10._train_one_epoch` (v10:116)
    # instead uses `sum(criterion(l, y) for l in logits_list) / len(logits_list)`,
    # which gives every segment a backward pass. Reproduced here.
    base = getattr(model, "_orig_mod", model)
    deep = hasattr(base, "forward_deep_supervision")

    def step():
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=dtype, enabled=dtype is not None):
            if deep:
                logits_list, _q = base.forward_deep_supervision(x)
                loss = sum(lossf(l.float(), y) for l in logits_list) / len(logits_list)
            else:
                out = model(x)
                while isinstance(out, (tuple, list)):
                    out = out[-1] if not torch.is_tensor(out[0]) else out[0]
                loss = lossf(out.float(), y)
        loss.backward()
        opt.step()

    for _ in range(args.warmup):
        step()
    torch.cuda.synchronize()

    times = []
    for _ in range(args.iters):
        t0 = time.perf_counter()
        step()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)

    peak = torch.cuda.max_memory_allocated() / 1024 ** 2
    del model, opt, x, y
    torch.cuda.empty_cache()
    return statistics.median(times), peak


def main() -> int:
    args = build_arg_parser().parse_args()
    if not torch.cuda.is_available():
        print("No CUDA device visible - this script measures a GPU and has nothing to measure.\n"
              "If you expected one: a cu124 torch has no sm_120 kernels and cannot drive a "
              "Blackwell card. Check torch.cuda.get_arch_list().", file=sys.stderr)
        return 2

    device = "cuda"
    total_mb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 2
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    print(f"device      : {torch.cuda.get_device_name(0)}  ({total_mb:,.0f} MB)")
    print(f"torch       : {torch.__version__}  (cuda {torch.version.cuda}, "
          f"arch {torch.cuda.get_arch_list()})")
    print(f"input       : {args.input_side}x{args.input_side}x{args.in_channels}, "
          f"amp={args.amp}, arch={args.architecture}, mixer={args.trm_mixer}")
    print(f"spectral ckpt: {args.spectral_checkpointing} -> "
          f"{', '.join(('ON' if _resolve_spectral_ckpt(args, v) else 'off') for v in _csv(args.spectral_checkpointing))}"
          f"  (the real path turns this on at in_channels >= 16)")
    print(f"timing      : {args.warmup} warmup + {args.iters} timed steps, median reported")
    print(f"NOT timed   : gradient health checks, GradScaler, EMA, DataLoader, augmentation,\n"
          f"              validation, artifact writing. On the PAD baseline those are 0.415 s\n"
          f"              of a 0.852 s real step - see plan/v17_baseline.md and the\n"
          f"              train_time_s / data_wait_s fields in a run's history.json.\n")

    header = (f"{'batch':>6} {'grid':>6} {'stride':>7} {'tokens':>7} {'ckpt':>5} {'spec':>5} "
              f"{'compile':>8} {'ms/step':>9} {'peak MB':>9} {'% card':>7} {'samp/s':>8}")
    print(header)
    print("-" * len(header))

    rows = []
    for spec_raw in _csv(args.spectral_checkpointing):
        spec = _resolve_spectral_ckpt(args, spec_raw)
        for compile_on in [v == "on" for v in _csv(args.compile_model)]:
            for ckpt in [v == "on" for v in _csv(args.checkpoint_core)]:
                for grid in _csv(args.token_grid, int):
                    for batch in _csv(args.batch_size, int):
                        stride = max(1, args.input_side // grid)
                        real_grid = args.input_side // stride
                        tokens = real_grid * real_grid
                        cell = (f"{batch:>6} {real_grid:>6} {stride:>7} {tokens:>7} "
                                f"{'on' if ckpt else 'off':>5} {'on' if spec else 'off':>5} "
                                f"{'on' if compile_on else 'off':>8}")
                        try:
                            ms, peak = _time_cell(args, batch, grid, compile_on, ckpt, device, spec)
                        except torch.OutOfMemoryError:
                            torch.cuda.empty_cache()
                            print(f"{cell} {'OOM':>9} {'-':>9} {'-':>7} {'-':>8}")
                            continue
                        except Exception as e:                  # compile failures, mostly
                            torch.cuda.empty_cache()
                            print(f"{cell}   FAILED: {type(e).__name__}: {str(e)[:60]}")
                            continue
                        print(f"{cell} {ms:>9.1f} {peak:>9.0f} "
                              f"{100 * peak / total_mb:>6.1f}% {1000.0 * batch / ms:>8.1f}")
                        rows.append((batch, tokens, ckpt, compile_on, ms, peak, spec))

    if rows:
        best = max(rows, key=lambda r: 1000.0 * r[0] / r[4])
        print(f"\nhighest throughput: batch {best[0]}, {best[1]} tokens, "
              f"ckpt={'on' if best[2] else 'off'}, spectral_ckpt={'on' if best[6] else 'off'}, "
              f"compile={'on' if best[3] else 'off'} "
              f"-> {1000.0 * best[0] / best[4]:.1f} samples/s at {best[5]:,.0f} MB "
              f"({100 * best[5] / total_mb:.1f}% of the card)")
        print("Throughput is not the objective - accuracy is. More tokens costs time and BUYS "
              "spatial detail; a bigger batch costs nothing here but changes the gradient-noise "
              "scale the LR was tuned against. Use this to know the price, then decide.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
