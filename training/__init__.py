"""
training/
=========
The MedMamba-SS-TRM training package. Entry points live in the repository root:
`train.py` (one run, `--profile ...`) and `run_experiments.py` (sweeps). The
pipeline they drive is `training/train_pipeline.py`; the trainer class is
`training/trainerg.py:TrainerG`.

Renamed modules
---------------
`RENAMED_MODULES` maps a module's old, version-suffixed name to the module that
holds its code now.
The archived entry points (`archive/train_example_v16*.py`, `_v18.py`,
`prepare_histologyhsi_bc_v8.py`) are frozen and still import the old names, and
`tests/test_train_e2e_parity.py` runs them, so each old name keeps resolving
here: `import training.gates_v16` returns the very same module object as
`import training.gates` (one module, two names, so a monkeypatch through either
name is seen through both). New code should use the current names.
"""

import importlib
import importlib.abc
import importlib.util
import sys

# old name -> current name, both relative to this package (renamed 2026-10-01)
RENAMED_MODULES = {
    "compile_cache_v17": "compile_cache",
    "ema_v16": "ema",
    "fd_limit_v16": "fd_limit",
    "flops_report_v18": "flops_report",
    "gates_v16": "gates",
    "grad_health_v16": "fused_stability_checks",
    "heldout_metrics_v19": "heldout_metrics",
    "hsi_baselines_v19": "hsi_baselines",
    "npy_dataset_v16": "npy_data",  # merged into npy_data.py
    "prep_progress_v16": "prep_progress",
    "progress_v16": "progress",
}


class _RenamedModuleFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Answers `import training.<old name>` with the already-imported current module."""

    def find_spec(self, fullname, path=None, target=None):
        package, _, old = fullname.rpartition(".")
        if package == __name__ and old in RENAMED_MODULES:
            return importlib.util.spec_from_loader(fullname, self)
        return None

    _real_specs = {}

    def create_module(self, spec):
        module = importlib.import_module(f"{__name__}.{RENAMED_MODULES[spec.name.rpartition('.')[2]]}")
        # The import system overwrites `__spec__` on whatever this returns; keep the
        # module's own, so it still reports its current name and file.
        self._real_specs[spec.name] = module.__spec__
        return module

    def exec_module(self, module):
        module.__spec__ = self._real_specs.pop(module.__spec__.name)


# Compared by name, not isinstance: `importlib.reload(training)` makes a new class.
if not any(type(f).__module__ == __name__ and type(f).__name__ == "_RenamedModuleFinder"
           for f in sys.meta_path):
    sys.meta_path.append(_RenamedModuleFinder())
