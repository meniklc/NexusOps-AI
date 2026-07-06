"""Tiny zero-dependency test runner for the offline suite.

Real usage is `pytest -q` (see requirements.txt). This runner exists only so
the pure-Python tests can be executed in environments without pytest installed.
It imports each `tests.test_*` module and runs its `test_*` functions, skipping
modules whose third-party imports are unavailable.

    python -m tests.run_offline
"""
from __future__ import annotations

import importlib
import pkgutil
import traceback

import tests


def main() -> int:
    passed = failed = skipped = 0
    for mod in pkgutil.iter_modules(tests.__path__):
        if not mod.name.startswith("test_"):
            continue
        full = f"tests.{mod.name}"
        try:
            module = importlib.import_module(full)
        except Exception as exc:  # missing optional dep -> skip
            skipped += 1
            print(f"SKIP {full} ({exc.__class__.__name__}: {exc})")
            continue
        for name in sorted(dir(module)):
            if name.startswith("test_"):
                try:
                    getattr(module, name)()
                    passed += 1
                    print(f"PASS {full}.{name}")
                except Exception:
                    failed += 1
                    print(f"FAIL {full}.{name}")
                    traceback.print_exc()
    print(f"\n{passed} passed, {failed} failed, {skipped} skipped")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
