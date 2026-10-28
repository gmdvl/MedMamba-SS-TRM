# -*- coding: utf-8 -*-
"""
hsi_augment/registry.py
=======================
How a stage becomes an `augment_pipeline.py` subcommand.

Why a registry, and why auto-discovery
--------------------------------------
About fifteen issues add stages in parallel branches. A central list that each
one appends to is a merge conflict per branch. Instead, a stage is ONE NEW FILE
in `hsi_augment/stages/`; the CLI imports every module there
(`load_builtin_stages`), and the module registers itself with a decorator:

    # hsi_augment/stages/validate.py
    from dataclasses import dataclass
    from hsi_augment.config import StageConfig, param
    from hsi_augment.registry import register_stage

    @dataclass(frozen=True)
    class ValidateConfig(StageConfig):
        chunk_rows: int = param(20_000, help="rows per memmap chunk")

    @register_stage("validate", config=ValidateConfig, dirname="prep/validate", order=10,
                    purposes=("structural",), help="P1: dataset discovery and validation")
    def run(ctx, cfg):
        from hsi_augment import discovery        # heavy imports inside, --help stays fast
        return discovery.run(ctx, cfg)           # optional summary dict -> done.json

Nothing else is edited: the subcommand, its flags (generated from the config
fields), its directory, its `done.json` and its place in `all` follow from the
registration.

Concepts
--------
* `name`      the subcommand (`validate`, `fit-cvae`, ...).
* `group`     several stages run under one subcommand: Phase 2a, 2b and 4a
              (#12, #13, #15) each register their own stage with
              `group="preprocess"`, and `augment_pipeline.py preprocess` runs
              all three in order.
* `dirname`   the stage directory under `--out`. One stage = one directory =
              one `done.json`. Two stages may not share a directory or nest one
              inside the other. Suggested layout (Plan §Architecture, "Output
              root"): `prep/<stage>`, `pool/...`, `gen/<model>`,
              `synth/<source>`, `export/<source>`, `lopo_utility`.
* `requires`  upstream stages whose `done.json` must exist. Their fingerprints
              are chained into this stage's, so re-running an upstream stage
              re-runs everything downstream of it. `ctx.upstream_dir(name)`
              only resolves declared upstreams, so every dependency is visible.
* `order`     tie-break for stages with no dependency between them.
* `purposes`  the `SplitGuard` purposes the stage may use (`io.PURPOSE_SPLITS`).
              Default: every train-only purpose. A stage that reads val/test
              ("structural", "final_eval") must say so here, in one reviewable line.
* `in_all`    False keeps a stage out of `all` (the demo stage).
"""

from __future__ import annotations

import heapq
import importlib
import pkgutil
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Type

from hsi_augment.config import ConfigError, StageConfig, check_config_class
from hsi_augment.io import PURPOSE_SPLITS, TRAIN_ONLY_PURPOSES

ALL = "all"
NAME_RE = re.compile(r"^[a-z][a-z0-9]*(-[a-z0-9]+)*$")
DIRNAME_RE = re.compile(r"^[A-Za-z0-9_.-]+(/[A-Za-z0-9_.-]+)*$")

# The plan's subcommands (Plan §Architecture) and the issues that implement
# them. Until a stage or group with that name is registered, the CLI lists it
# as planned and refuses to run it with a pointer to the issue. Registering it
# needs no edit here.
PLANNED_COMMANDS: Dict[str, str] = {
    "validate": "#11 (Phase 1)",
    "preprocess": "#12, #13, #14, #15, #20 (Phases 2a, 2b, 3, 4a; aug-calibration)",
    "pool": "#16 (Phase 4b)",
    "fit-cvae": "#23 (Phase 6)",
    "fit-ae": "#24 (Phase 7)",
    "fit-ldm": "#24 (Phase 7)",
    "generate": "#29 (Phase 8a)",
    "qc": "#28, #29 (Phase 8b)",
    "export": "#29 (Phase 9)",
    "lopo-utility": "#30 (Phase 8c)",
    "bench": "#33 (Phase 10)",
}

StageFn = Callable[[Any, StageConfig], Optional[Dict[str, Any]]]


class RegistryError(ValueError):
    """An invalid stage registration or target."""


@dataclass(frozen=True)
class StageSpec:
    """A registered stage. `fn(ctx, cfg)` does the work; see stage.StageContext."""
    name: str
    fn: StageFn = field(repr=False, compare=False)
    config_cls: Type[StageConfig] = field(repr=False)
    help: str = ""
    dirname: str = ""
    requires: Tuple[str, ...] = ()
    group: Optional[str] = None
    order: int = 100
    purposes: Tuple[str, ...] = TRAIN_ONLY_PURPOSES
    in_all: bool = True
    module: str = ""

    def reads_held_out(self) -> bool:
        """True if any declared purpose may read val or test."""
        return any(PURPOSE_SPLITS[p] != ("train",) for p in self.purposes)


class Registry:
    """Stages by name. `REGISTRY` is the one the CLI uses; tests build their own."""

    def __init__(self) -> None:
        self._specs: Dict[str, StageSpec] = {}

    # -- registration -----------------------------------------------------------
    def register(self, spec: StageSpec) -> StageSpec:
        if not NAME_RE.match(spec.name):
            raise RegistryError(f"stage name {spec.name!r} must be lowercase words joined by '-'")
        if spec.name == ALL:
            raise RegistryError(f"{ALL!r} is reserved")
        if spec.group is not None and not NAME_RE.match(spec.group):
            raise RegistryError(f"group name {spec.group!r} must be lowercase words joined by '-'")
        if spec.group in (ALL, spec.name):
            raise RegistryError(f"stage {spec.name!r}: invalid group {spec.group!r}")
        if not DIRNAME_RE.match(spec.dirname) or ".." in spec.dirname.split("/"):
            raise RegistryError(f"stage {spec.name!r}: dirname {spec.dirname!r} must be a relative path")
        unknown = sorted(set(spec.purposes) - set(PURPOSE_SPLITS))
        if unknown or not spec.purposes:
            raise RegistryError(f"stage {spec.name!r}: unknown/empty purposes {unknown or '()'}; "
                                f"known: {sorted(PURPOSE_SPLITS)}")
        if spec.name in spec.requires:
            raise RegistryError(f"stage {spec.name!r} requires itself")
        try:
            check_config_class(spec.config_cls)
        except ConfigError as e:
            raise RegistryError(f"stage {spec.name!r}: {e}") from e

        old = self._specs.get(spec.name)
        if old is not None and (old.module, getattr(old.fn, "__qualname__", None)) != \
                (spec.module, getattr(spec.fn, "__qualname__", None)):
            raise RegistryError(f"stage {spec.name!r} is already registered by {old.module}")
        for other in self._specs.values():
            if other.name == spec.name:
                continue  # re-registration of the same function (module reload)
            if spec.name == other.group or spec.group == other.name:
                raise RegistryError(f"{spec.name!r}/{spec.group!r}: a stage and a group share a name")
            a, b = spec.dirname + "/", other.dirname + "/"
            if a.startswith(b) or b.startswith(a):
                raise RegistryError(f"stage {spec.name!r} dir {spec.dirname!r} overlaps stage "
                                    f"{other.name!r} dir {other.dirname!r}")
        self._specs[spec.name] = spec
        return spec

    def stage(self, name: str, *, config: Type[StageConfig], help: str = "",
              dirname: Optional[str] = None, requires: Sequence[str] = (), group: Optional[str] = None,
              order: int = 100, purposes: Sequence[str] = TRAIN_ONLY_PURPOSES,
              in_all: bool = True) -> Callable[[StageFn], StageFn]:
        """Decorator form of `register`. Returns the function unchanged."""
        def deco(fn: StageFn) -> StageFn:
            self.register(StageSpec(name=name, fn=fn, config_cls=config, help=help,
                                    dirname=dirname or name, requires=tuple(requires), group=group,
                                    order=int(order), purposes=tuple(purposes), in_all=bool(in_all),
                                    module=getattr(fn, "__module__", "")))
            return fn
        return deco

    # -- lookup -----------------------------------------------------------------
    def __contains__(self, name: str) -> bool:
        return name in self._specs

    def get(self, name: str) -> StageSpec:
        try:
            return self._specs[name]
        except KeyError:
            raise RegistryError(f"no stage named {name!r}") from None

    def specs(self) -> List[StageSpec]:
        """Every stage, in run order."""
        return self.ordered(self._specs.values())

    def groups(self) -> Dict[str, List[StageSpec]]:
        out: Dict[str, List[StageSpec]] = {}
        for s in self.specs():
            if s.group:
                out.setdefault(s.group, []).append(s)
        return out

    def command_names(self) -> List[str]:
        return sorted(set(self._specs) | set(self.groups()) | {ALL})

    def planned_only(self) -> Dict[str, str]:
        """Planned subcommands nobody has registered yet."""
        have = set(self._specs) | set(self.groups())
        return {k: v for k, v in PLANNED_COMMANDS.items() if k not in have}

    # -- ordering ----------------------------------------------------------------
    def ordered(self, specs: Iterable[StageSpec]) -> List[StageSpec]:
        """Topological order by `requires` (only among `specs`; an upstream
        outside the set must already be done, which the runner checks), ties
        broken by (`order`, name). A cycle is an error."""
        chosen = {s.name: s for s in specs}
        indeg = {n: 0 for n in chosen}
        children: Dict[str, List[str]] = {n: [] for n in chosen}
        for s in chosen.values():
            for r in s.requires:
                if r in chosen:
                    indeg[s.name] += 1
                    children[r].append(s.name)
        heap = [(chosen[n].order, n) for n, d in indeg.items() if d == 0]
        heapq.heapify(heap)
        out: List[StageSpec] = []
        while heap:
            _, n = heapq.heappop(heap)
            out.append(chosen[n])
            for c in children[n]:
                indeg[c] -= 1
                if indeg[c] == 0:
                    heapq.heappush(heap, (chosen[c].order, c))
        if len(out) != len(chosen):
            raise RegistryError(f"dependency cycle among {sorted(set(chosen) - {s.name for s in out})}")
        return out

    def resolve(self, targets: Sequence[str]) -> List[StageSpec]:
        """Stage names, group names and/or 'all' -> the stages to run, in order."""
        picked: Dict[str, StageSpec] = {}
        groups = self.groups()
        for t in targets:
            if t == ALL:
                picked.update({s.name: s for s in self._specs.values() if s.in_all})
            elif t in self._specs:
                picked[t] = self._specs[t]
            elif t in groups:
                picked.update({s.name: s for s in groups[t]})
            elif t in PLANNED_COMMANDS:
                raise RegistryError(f"{t!r} is planned but not implemented yet ({PLANNED_COMMANDS[t]})")
            else:
                import difflib
                close = difflib.get_close_matches(t, self.command_names() + list(PLANNED_COMMANDS), n=3)
                hint = f"; did you mean {close}?" if close else ""
                raise RegistryError(f"unknown command {t!r}{hint}")
        return self.ordered(picked.values())


REGISTRY = Registry()


def register_stage(name: str, *, config: Type[StageConfig], help: str = "",
                   dirname: Optional[str] = None, requires: Sequence[str] = (),
                   group: Optional[str] = None, order: int = 100,
                   purposes: Sequence[str] = TRAIN_ONLY_PURPOSES,
                   in_all: bool = True) -> Callable[[StageFn], StageFn]:
    """Register a stage with the CLI's registry (`REGISTRY`). See the module
    docstring for an example and `StageSpec` for the fields."""
    return REGISTRY.stage(name, config=config, help=help, dirname=dirname, requires=requires,
                          group=group, order=order, purposes=purposes, in_all=in_all)


def load_builtin_stages(package: str = "hsi_augment.stages") -> Registry:
    """Import every module of `package` (sorted) so their `@register_stage`
    decorators run. An import error is NOT swallowed: a broken stage file must
    fail loudly, not vanish from the CLI."""
    pkg = importlib.import_module(package)
    for m in sorted(pkgutil.iter_modules(pkg.__path__), key=lambda m: m.name):
        if not m.name.startswith("_"):
            importlib.import_module(f"{package}.{m.name}")
    return REGISTRY
