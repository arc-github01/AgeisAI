"""Makes the repository root importable so tests can ``import src.*``.

Kept at the root (rather than shipping a packaging config) because the project
is run in place: ``pytest``, ``python -m src...`` and ``streamlit run app.py``
all execute from the repository root.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
