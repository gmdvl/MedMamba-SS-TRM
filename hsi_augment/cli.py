# -*- coding: utf-8 -*-
"""
hsi_augment/cli.py
==================
The `augment_pipeline.py` command line (the root script only calls `main`).

    python augment_pipeline.py --help                      # commands, planned ones included
    python augment_pipeline.py --list --data_dir D --out O # each stage: done / stale / missing / ...
    python augment_pipeline.py validate preprocess --data_dir data/hsi_v9-trainsel/hsi \\
                                                    --out data/hsi_v9-trainsel-aug
    python augment_pipeline.py all --data_dir ... --out ... [--force] [--resume auto]
    python augment_pipeline.py demo --data_dir ... --out ... --chunk_rows 5000

Commands are stage names, group names (a group runs all its stages) or `all`.
Several may be given, as the plan's `validate preprocess`; they run in
dependency order whatever order they are typed in. Commands come first, flags
after.

Common flags: --data_dir, --out, --seed, --config, --force, --resume.
A single STAGE also takes one flag per field of its config dataclass
(`augment_pipeline.py <stage> --help` lists them). When several stages run,
their settings come from `--config file.json` only, because two stages may
have a field of the same name.

Precedence: config dataclass defaults < --config file < flags typed.

Exit status: 0 ok, 1 a stage failed (its done.json is absent), 2 usage error,
130 interrupted.
"""

from __future__ import annotations

import argparse
import sys
from typing import Dict, List, Optional, Sequence, Tuple

from hsi_augment.config import (ConfigError, add_config_flags, build_config, explicit_values,
                                load_config_file)
from hsi_augment.registry import ALL, Registry, RegistryError, load_builtin_stages
from hsi_augment.stage import (DEFAULT_SEED, RESUME_MODES, RunOptions, StageError, run_stage,
                               stage_status)

PROG = "augment_pipeline.py"


def _split_targets(argv: Sequence[str]) -> Tuple[List[str], List[str]]:
    """Leading words are commands; everything from the first flag on is flags.
    Splitting by position (not by parsing) keeps `--seed 7 demo` from being
    read as the commands ['7', 'demo']."""
    targets: List[str] = []
    i = 0
    while i < len(argv) and not str(argv[i]).startswith("-"):
        targets.append(str(argv[i]))
        i += 1
    return targets, list(argv[i:])


def _add_common_flags(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("common")
    # SUPPRESS: absent from the namespace unless typed, so --config can supply them.
    g.add_argument("--data_dir", default=argparse.SUPPRESS,
                   help="input build: its array dir (e.g. data/hsi_v9-trainsel/hsi) or its root")
    g.add_argument("--out", default=argparse.SUPPRESS,
                   help="output root (e.g. data/hsi_v9-trainsel-aug); one subdirectory per stage")
    g.add_argument("--seed", type=int, default=argparse.SUPPRESS,
                   help=f"master seed; each stage derives its own (default: {DEFAULT_SEED})")
    g.add_argument("--config", default=None,
                   help='JSON defaults: {"data_dir", "out", "seed", "stages": {name: {field: value}}}; '
                        "typed flags win")
    g.add_argument("--force", action="store_true", help="re-run stages that are already done")
    g.add_argument("--resume", choices=RESUME_MODES, default="off",
                   help="auto: an unfinished stage with the same fingerprint continues from its "
                        "checkpoints (default: off)")


def _commands_epilog(registry: Registry) -> str:
    lines = ["commands:"]
    for s in registry.specs():
        grp = f" [group {s.group}]" if s.group else ""
        extra = "" if s.in_all else " (not in 'all')"
        lines.append(f"  {s.name:<16}{s.help}{grp}{extra}")
    for g, members in registry.groups().items():
        lines.append(f"  {g:<16}group: {', '.join(m.name for m in members)}")
    lines.append(f"  {ALL:<16}every registered stage marked for 'all', in dependency order")
    planned = registry.planned_only()
    if planned:
        lines.append("planned (not implemented yet):")
        for name, where in planned.items():
            lines.append(f"  {name:<16}{where}")
    return "\n".join(lines)


def build_top_parser(registry: Registry) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=PROG, formatter_class=argparse.RawDescriptionHelpFormatter,
        description="HSI augmentation pipeline (plan/HSI_augmentation_pipeline.plan.md). "
                    "Commands first, then flags.",
        epilog=_commands_epilog(registry))
    p.add_argument("commands", nargs="*", metavar="COMMAND", help="stage, group, or 'all'")
    p.add_argument("--list", action="store_true",
                   help="list stages; with --data_dir and --out, also each stage's status")
    _add_common_flags(p)
    return p


def build_stage_parser(registry: Registry, targets: Sequence[str]) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog=f"{PROG} {' '.join(targets)}",
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                description=_targets_description(registry, targets))
    _add_common_flags(p)
    if len(targets) == 1 and targets[0] in registry:
        spec = registry.get(targets[0])
        add_config_flags(p, spec.config_cls, title=f"{spec.name} settings ({spec.config_cls.__name__})")
    return p


def _targets_description(registry: Registry, targets: Sequence[str]) -> str:
    specs = registry.resolve(targets)
    lines = ["runs, in order:"]
    for s in specs:
        req = f"  requires {', '.join(s.requires)}" if s.requires else ""
        lines.append(f"  {s.name:<16}-> <out>/{s.dirname}{req}")
    if not (len(targets) == 1 and targets[0] in registry):
        lines.append("stage settings: --config file.json only (several stages run)")
    return "\n".join(lines)


def _options(ns: argparse.Namespace, file_cfg: Dict, parser: argparse.ArgumentParser) -> RunOptions:
    data_dir = getattr(ns, "data_dir", None) or file_cfg.get("data_dir")
    out = getattr(ns, "out", None) or file_cfg.get("out")
    if not data_dir or not out:
        parser.error("--data_dir and --out are required (as flags or in --config)")
    seed = getattr(ns, "seed", None)
    if seed is None:
        seed = file_cfg.get("seed", DEFAULT_SEED)
    try:
        return RunOptions(data_dir=str(data_dir), out=str(out), seed=seed, force=bool(ns.force),
                          resume=ns.resume)
    except ValueError as e:
        parser.error(str(e))
        raise  # unreachable; parser.error exits


def _stage_configs(registry: Registry, specs, file_cfg: Dict, ns: argparse.Namespace,
                   flags_for: Optional[str]) -> Dict[str, object]:
    sections = file_cfg.get("stages", {})
    unknown = sorted(n for n in sections if n not in registry)
    if unknown:
        raise ConfigError(f"--config has settings for unregistered stage(s) {unknown}")
    cfgs = {}
    for s in specs:
        explicit = explicit_values(ns, s.config_cls) if s.name == flags_for else {}
        cfgs[s.name] = build_config(s.config_cls, sections.get(s.name), explicit)
    return cfgs


def _list(registry: Registry, ns: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    status: Dict[str, Tuple[str, str]] = {}
    have_io = getattr(ns, "data_dir", None) or getattr(ns, "out", None) or ns.config
    if have_io:
        file_cfg = load_config_file(ns.config)
        opts = _options(ns, file_cfg, parser)
        cfgs = _stage_configs(registry, registry.specs(), file_cfg, ns, None)
        for s in registry.specs():
            status[s.name] = stage_status(s, cfgs[s.name], opts, registry)
    print(f"{'stage':<16}{'group':<12}{'dir':<22}{'requires':<20}{'purposes':<24}status")
    for s in registry.specs():
        st, why = status.get(s.name, ("-", ""))
        print(f"{s.name:<16}{s.group or '-':<12}{s.dirname:<22}{','.join(s.requires) or '-':<20}"
              f"{','.join(s.purposes):<24}{st}{(' - ' + why) if why else ''}")
    planned = registry.planned_only()
    if planned:
        print("planned (not implemented yet): " + ", ".join(f"{k} {v}" for k, v in planned.items()))
    return 0


def main(argv: Optional[Sequence[str]] = None, registry: Optional[Registry] = None) -> int:
    registry = registry if registry is not None else load_builtin_stages()
    argv = list(sys.argv[1:] if argv is None else argv)
    targets, rest = _split_targets(argv)

    if not targets:
        top = build_top_parser(registry)
        ns = top.parse_args(rest)
        try:
            if ns.list:
                return _list(registry, ns, top)
        except (ConfigError, StageError) as e:
            print(f"[augment_pipeline] {e}", file=sys.stderr)
            return 2
        top.print_help()
        return 2

    try:
        specs = registry.resolve(targets)
        parser = build_stage_parser(registry, targets)
    except RegistryError as e:
        print(f"[augment_pipeline] {e}", file=sys.stderr)
        return 2
    ns = parser.parse_args(rest)
    try:
        file_cfg = load_config_file(ns.config)
        opts = _options(ns, file_cfg, parser)
        flags_for = targets[0] if len(targets) == 1 and targets[0] in registry else None
        cfgs = _stage_configs(registry, specs, file_cfg, ns, flags_for)
    except ConfigError as e:
        print(f"[augment_pipeline] {e}", file=sys.stderr)
        return 2
    if not specs:
        print("[augment_pipeline] no registered stage to run", flush=True)
        return 0

    print(f"[augment_pipeline] {len(specs)} stage(s): {' -> '.join(s.name for s in specs)}  "
          f"data_dir={opts.data_dir} out={opts.out} seed={opts.seed}"
          f"{' force' if opts.force else ''}{' resume=auto' if opts.resume == 'auto' else ''}", flush=True)
    results = []
    for s in specs:
        try:
            results.append(run_stage(s, cfgs[s.name], opts, registry))
        except StageError as e:
            print(f"[{s.name}] FAILED: {e}", file=sys.stderr, flush=True)
            return 1
        except KeyboardInterrupt:
            print(f"[{s.name}] interrupted; no done.json was written", file=sys.stderr, flush=True)
            return 130
    for r in results:
        wall = r.done.get("timings", {}).get("wall_s")
        took = f" in {wall:.1f}s" if r.status == "ran" and wall is not None else ""
        print(f"[augment_pipeline] {r.name}: {r.status}{took} -> {r.stage_dir}", flush=True)
    return 0
