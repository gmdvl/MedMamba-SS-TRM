# -*- coding: utf-8 -*-
"""
hsi_augment/config.py
=====================
Stage configuration: one frozen dataclass per stage.

Why a dataclass and not the argparse namespace
----------------------------------------------
A stage's outputs are reused only while its configuration is unchanged
(`stage.py` skips a stage whose `done.json` records the same `config_hash`).
That needs a configuration with a stable, canonical identity. A namespace mixes
in flags that do not change the result (`--force`, `--resume`) and has no
types. A frozen dataclass has exactly the fields that matter, with types:

  * `to_dict()` / `from_dict()` and `to_json()` / `from_json()` round-trip, so
    the resolved `config.json` a stage writes can be fed back with `--config`;
  * `config_hash()` is the sha256 of the canonical JSON (sorted keys, no
    whitespace, NaN refused). Values are coerced to the field's type first, so
    `1` and `1.0` in a float field, or a list and a tuple in a tuple field,
    hash the same;
  * unknown keys are an error, so a typo in a config file cannot silently fall
    back to a default.

Precedence (Plan §Architecture, "Configuration and seeds")
----------------------------------------------------------
dataclass default  <  `--config file.json`  <  flags typed on the command line.

The flags are generated from the fields (`add_config_flags`) with
`default=argparse.SUPPRESS`, so the parsed namespace holds only what was
typed, and "explicit flags win" needs no second parse.

Writing a stage config
----------------------
Put it next to the stage, in that stage's own module, not in this file, so
branches that add stages in parallel never edit the same file:

    @dataclass(frozen=True)
    class TissueMaskConfig(StageConfig):
        sample_rows: int = param(500_000, help="stratified train sample for the GMM")
        method: str = param("gmm", help="threshold method", choices=("gmm", "otsu"))
        quantiles: Tuple[float, ...] = param((0.005, 0.995), help="robust range")

        def validate(self) -> None:
            if self.sample_rows <= 0:
                raise ConfigError("sample_rows must be > 0")

Supported field types: bool, int, float, str, Optional[...] of those,
Tuple[X, ...] and fixed Tuple[X, Y], Literal[...], and JSON-able dicts
(config file only, no flag). Field names must not shadow the common CLI flags
(`RESERVED_FIELD_NAMES`).
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import types
import typing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Type, TypeVar

# Common flags of `augment_pipeline.py`. A stage field with one of these names
# would be unreachable from the command line, so `check_config_class` refuses it.
RESERVED_FIELD_NAMES = frozenset({"data_dir", "out", "seed", "config", "force", "resume", "list",
                                  "help", "stages"})

# Top-level keys of a `--config` file. "stages" holds one section per stage.
CONFIG_FILE_KEYS = ("data_dir", "out", "seed", "stages")
# Keys the pipeline itself writes into a resolved `config.json`; ignored on load
# so that file can be passed straight back with `--config`.
CONFIG_FILE_METADATA_KEYS = ("schema_version", "config_hashes", "comment", "_comment")

# argparse dest prefix for generated flags, so a field can never collide with a
# common flag's dest.
FLAG_DEST_PREFIX = "cfg__"

# `Optional[X]` and `X | None` have different origins.
_UNION_ORIGINS = (typing.Union, types.UnionType)

_BOOL_TRUE = ("on", "true", "yes", "1")
_BOOL_FALSE = ("off", "false", "no", "0")

T = TypeVar("T", bound="StageConfig")


class ConfigError(ValueError):
    """A configuration value or file is invalid."""


# ============================================================================
# canonical JSON
# ============================================================================

def to_jsonable(obj: Any) -> Any:
    """Plain-JSON form: tuples -> lists, numpy scalars/arrays -> Python, Path ->
    str, dict keys -> str. `training.npy_atomic.save_json_atomic` falls back to
    `default=str`, which would write an np.float32 as the STRING "1.5", so
    everything bound for JSON goes through here first."""
    if isinstance(obj, (str, bool, type(None))):
        return obj
    if isinstance(obj, (int, float)):
        return obj
    if isinstance(obj, Mapping):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    try:  # numpy without importing it here
        import numpy as np
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except ImportError:  # pragma: no cover
        pass
    raise ConfigError(f"value {obj!r} of type {type(obj).__name__} is not JSON-serialisable")


def canonical_json(obj: Any) -> str:
    """Sorted keys, no whitespace, ASCII, NaN/Inf refused: the same value always
    gives the same bytes."""
    try:
        return json.dumps(to_jsonable(obj), sort_keys=True, separators=(",", ":"),
                          ensure_ascii=True, allow_nan=False)
    except ValueError as e:
        raise ConfigError(f"cannot hash {obj!r}: {e}") from e


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ============================================================================
# field declaration and type coercion
# ============================================================================

def param(default: Any = dataclasses.MISSING, *, help: str = "",
          choices: Optional[Iterable[Any]] = None, flag: bool = True,
          default_factory: Any = dataclasses.MISSING) -> Any:
    """`dataclasses.field` with the metadata the flag generator reads.
    `flag=False` keeps a field out of the command line (config file only)."""
    meta = {"help": help, "choices": tuple(choices) if choices is not None else None, "flag": flag}
    if default_factory is not dataclasses.MISSING:
        return dataclasses.field(default_factory=default_factory, metadata=meta)
    return dataclasses.field(default=default, metadata=meta)


_HINTS_CACHE: Dict[type, Dict[str, Any]] = {}


def _type_hints(cls: type) -> Dict[str, Any]:
    if cls not in _HINTS_CACHE:
        _HINTS_CACHE[cls] = typing.get_type_hints(cls)
    return _HINTS_CACHE[cls]


def _coerce(value: Any, tp: Any, where: str) -> Any:
    """`value` converted to annotation `tp`, or ConfigError."""
    origin = typing.get_origin(tp)
    args = typing.get_args(tp)

    if tp is Any:
        return value
    if origin in _UNION_ORIGINS:
        if value is None and type(None) in args:
            return None
        inner = [a for a in args if a is not type(None)]
        if len(inner) == 1:
            return _coerce(value, inner[0], where)
        for a in inner:  # first member that accepts it
            try:
                return _coerce(value, a, where)
            except ConfigError:
                continue
        raise ConfigError(f"{where}: {value!r} matches none of {tp}")
    if origin is typing.Literal:
        if value not in args:
            raise ConfigError(f"{where}: {value!r} is not one of {list(args)}")
        return value
    if tp is bool:
        if isinstance(value, bool):
            return value
        raise ConfigError(f"{where}: expected a bool, got {value!r}")
    if tp is int:
        if isinstance(value, bool):
            raise ConfigError(f"{where}: expected an int, got the bool {value!r}")
        if isinstance(value, int):
            return int(value)
        if isinstance(value, float) and value.is_integer():  # 2e4 in a JSON file
            return int(value)
        try:
            import numpy as np
            if isinstance(value, np.integer):
                return int(value)
        except ImportError:  # pragma: no cover
            pass
        raise ConfigError(f"{where}: expected an int, got {value!r}")
    if tp is float:
        if isinstance(value, bool):
            raise ConfigError(f"{where}: expected a float, got the bool {value!r}")
        if isinstance(value, (int, float)):
            return float(value)
        try:
            import numpy as np
            if isinstance(value, (np.integer, np.floating)):
                return float(value)
        except ImportError:  # pragma: no cover
            pass
        raise ConfigError(f"{where}: expected a float, got {value!r}")
    if tp is str:
        if isinstance(value, str):
            return value
        raise ConfigError(f"{where}: expected a str, got {value!r}")
    if origin is tuple or tp is tuple:
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{where}: expected a list, got {value!r}")
        if not args:
            return tuple(value)
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_coerce(v, args[0], f"{where}[{i}]") for i, v in enumerate(value))
        if len(value) != len(args):
            raise ConfigError(f"{where}: expected {len(args)} values, got {len(value)}")
        return tuple(_coerce(v, a, f"{where}[{i}]") for i, (v, a) in enumerate(zip(value, args)))
    if origin in (dict, Mapping) or tp is dict:
        if not isinstance(value, Mapping):
            raise ConfigError(f"{where}: expected a dict, got {value!r}")
        canonical_json(value)  # must be JSON-able
        return dict(value)
    # anything else must at least be JSON-able, or the hash is undefined
    canonical_json(value)
    return value


# ============================================================================
# the base class
# ============================================================================

@dataclass(frozen=True)
class StageConfig:
    """Base for every stage's configuration. Subclass with
    `@dataclass(frozen=True)`; override `validate()` for range checks."""

    def __post_init__(self) -> None:
        hints = _type_hints(type(self))
        for f in dataclasses.fields(self):
            value = _coerce(getattr(self, f.name), hints.get(f.name, Any),
                            f"{type(self).__name__}.{f.name}")
            object.__setattr__(self, f.name, value)
            choices = (f.metadata or {}).get("choices")
            if choices and value is not None:
                for v in (value if isinstance(value, tuple) else (value,)):
                    if v not in choices:
                        raise ConfigError(f"{type(self).__name__}.{f.name}: {v!r} is not one of "
                                          f"{list(choices)}")
        self.validate()

    def validate(self) -> None:
        """Raise ConfigError for an invalid combination of values."""

    # -- dict / JSON ----------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {f.name: to_jsonable(getattr(self, f.name)) for f in dataclasses.fields(self)}

    @classmethod
    def from_dict(cls: Type[T], data: Optional[Mapping[str, Any]] = None) -> T:
        data = dict(data or {})
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ConfigError(f"{cls.__name__}: unknown key(s) {unknown}; known: {sorted(known)}")
        return cls(**data)

    def to_json(self, indent: Optional[int] = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, allow_nan=False)

    @classmethod
    def from_json(cls: Type[T], text: str) -> T:
        return cls.from_dict(json.loads(text))

    def canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def config_hash(self) -> str:
        """sha256 of the canonical JSON of the fields. The class name is not
        part of it: the stage directory already says which stage this is."""
        return sha256_hex(self.canonical_json())

    def replace(self: T, **changes: Any) -> T:
        return dataclasses.replace(self, **changes)


def check_config_class(cls: type) -> None:
    """Called at stage registration: a frozen StageConfig dataclass whose field
    names do not shadow the common flags and whose defaults construct."""
    if not (isinstance(cls, type) and issubclass(cls, StageConfig) and dataclasses.is_dataclass(cls)):
        raise ConfigError(f"{cls!r} must be a @dataclass(frozen=True) subclass of StageConfig")
    if not cls.__dataclass_params__.frozen:  # type: ignore[attr-defined]
        raise ConfigError(f"{cls.__name__} must be declared with @dataclass(frozen=True)")
    clash = sorted(f.name for f in dataclasses.fields(cls) if f.name in RESERVED_FIELD_NAMES)
    if clash:
        raise ConfigError(f"{cls.__name__}: field name(s) {clash} shadow common pipeline flags")
    cls()  # every field needs a default, and the defaults must validate


# ============================================================================
# command-line flags generated from the fields
# ============================================================================

def parse_bool(text: str) -> bool:
    t = str(text).strip().lower()
    if t in _BOOL_TRUE:
        return True
    if t in _BOOL_FALSE:
        return False
    raise argparse.ArgumentTypeError(f"expected on/off, got {text!r}")


def _optional(parse: Callable[[str], Any]) -> Callable[[str], Any]:
    def _p(text: str) -> Any:
        return None if str(text).strip().lower() in ("none", "null") else parse(text)
    _p.__name__ = getattr(parse, "__name__", "value")
    return _p


def _flag_spec(tp: Any) -> Optional[Dict[str, Any]]:
    """argparse kwargs for annotation `tp`, or None if it has no flag form."""
    origin, args = typing.get_origin(tp), typing.get_args(tp)
    if origin in _UNION_ORIGINS:
        inner = [a for a in args if a is not type(None)]
        if len(inner) == 1 and type(None) in args:
            spec = _flag_spec(inner[0])
            if spec is None or "nargs" in spec:
                return None
            spec["type"] = _optional(spec.get("type", str))
            spec.pop("choices", None)
            return spec
        return None
    if origin is typing.Literal:
        kinds = {type(a) for a in args}
        return {"type": kinds.pop() if len(kinds) == 1 else str, "choices": list(args)}
    if tp is bool:
        return {"type": parse_bool, "metavar": "{on,off}"}
    if tp in (int, float, str):
        return {"type": tp}
    if origin is tuple and args:
        elem = args[0]
        if (len(args) == 2 and args[1] is Ellipsis) or all(a is elem for a in args):
            if elem in (int, float, str):
                spec = {"type": elem, "nargs": "+" if args[-1] is Ellipsis else len(args)}
                return spec
            if elem is bool:
                return {"type": parse_bool, "nargs": "+" if args[-1] is Ellipsis else len(args)}
        return None
    return None


def _checked_choice(parse: Callable[[str], Any], allowed: Sequence[Any]) -> Callable[[str], Any]:
    def _p(text: str) -> Any:
        value = parse(text)
        if value is not None and value not in allowed:
            raise argparse.ArgumentTypeError(f"invalid choice {value!r} (choose from {list(allowed)})")
        return value
    _p.__name__ = getattr(parse, "__name__", "value")
    return _p


def add_config_flags(parser: argparse.ArgumentParser, cls: Type[StageConfig],
                     title: Optional[str] = None) -> argparse._ArgumentGroup:
    """One `--<field>` flag per field of `cls` (skipping `param(flag=False)`
    and types without a flag form). Defaults are SUPPRESSed: the namespace
    only holds typed flags, which is what makes them win over `--config`."""
    group = parser.add_argument_group(title or f"{cls.__name__} (stage settings)")
    hints = _type_hints(cls)
    defaults = cls().to_dict()
    for f in dataclasses.fields(cls):
        meta = f.metadata or {}
        if meta.get("flag") is False:
            continue
        spec = _flag_spec(hints.get(f.name, Any))
        if spec is None:
            continue
        if meta.get("choices"):
            # Checked inside the type, not with argparse `choices`: argparse would
            # compare the CONVERTED value, and an Optional field's "none" -> None
            # is never in the list.
            allowed = list(meta["choices"])
            spec["type"] = _checked_choice(spec.get("type", str), allowed)
            spec["metavar"] = "{" + ",".join(map(str, allowed)) + "}"
            spec.pop("choices", None)
        if "choices" not in spec:
            spec.setdefault("metavar", f.name.upper())  # not the internal cfg__ dest
        default_txt = json.dumps(defaults[f.name])
        help_txt = (meta.get("help") or "").strip()
        spec["help"] = f"{help_txt} (default: {default_txt})" if help_txt else f"default: {default_txt}"
        group.add_argument(f"--{f.name}", dest=FLAG_DEST_PREFIX + f.name, default=argparse.SUPPRESS,
                           **spec)
    return group


def explicit_values(namespace: argparse.Namespace, cls: Type[StageConfig]) -> Dict[str, Any]:
    """{field: value} for the generated flags that were typed."""
    out = {}
    for f in dataclasses.fields(cls):
        dest = FLAG_DEST_PREFIX + f.name
        if hasattr(namespace, dest):
            out[f.name] = getattr(namespace, dest)
    return out


# ============================================================================
# --config file.json
# ============================================================================

def load_config_file(path: Optional[str]) -> Dict[str, Any]:
    """Reads a `--config` file:

        {"data_dir": "...", "out": "...", "seed": 42,
         "stages": {"<stage name>": {"<field>": value, ...}, ...}}

    Every key is optional. The resolved `config.json` a stage writes has this
    shape too, so it can be passed back to reproduce the stage."""
    if not path:
        return {}
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, ValueError) as e:
        raise ConfigError(f"--config {path}: cannot read JSON ({e})") from e
    if not isinstance(data, dict):
        raise ConfigError(f"--config {path}: the top level must be an object")
    unknown = sorted(set(data) - set(CONFIG_FILE_KEYS) - set(CONFIG_FILE_METADATA_KEYS))
    if unknown:
        raise ConfigError(f"--config {path}: unknown key(s) {unknown}; allowed: {list(CONFIG_FILE_KEYS)}")
    stages = data.get("stages", {})
    if not isinstance(stages, dict) or not all(isinstance(v, dict) for v in stages.values()):
        raise ConfigError(f"--config {path}: 'stages' must map stage names to objects")
    return {k: data[k] for k in CONFIG_FILE_KEYS if k in data}


def build_config(cls: Type[T], file_section: Optional[Mapping[str, Any]] = None,
                 explicit: Optional[Mapping[str, Any]] = None) -> T:
    """defaults < file section < explicit flags."""
    merged = dict(file_section or {})
    merged.update(explicit or {})
    return cls.from_dict(merged)
