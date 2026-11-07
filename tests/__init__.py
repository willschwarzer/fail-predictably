"""CPU unit tests. Run one file from the repository root, e.g.

    python -m tests.test_ols_bootstrap

or all of them with ``pytest`` if it is installed. Importing this package puts
``src/`` on ``sys.path``, so the tests run without installing the package.
"""
import sys
from pathlib import Path

_SRC = str(Path(__file__).resolve().parent.parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
