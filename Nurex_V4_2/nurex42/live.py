"""
nurex42.live — Nurex V4.2 live-trading execution module.

SOURCE STATUS: Python 3.14 bytecode only (live.pyc, compiled 2026-09-16).
No .py source is available. This stub was created by Phase 0 of the
transformation plan on 2026-09-21 to make the package auditable.

To recover the full source:
  1. On a machine with pycdc or decompyle4 that supports Python 3.14:
       pycdc Nurex_V4_2/nurex42/__pycache__/live.cpython-314.pyc > live_recovered.py
  2. Or run _decompile_phase0.py with the project's Python 3.14 venv:
       Nurex_V4_2\\.venv\\Scripts\\python.exe _decompile_phase0.py

NOTE: This module is NOT imported by the V4.1 trading flow (train_v4_1.py).
It appears to be a V4.2 live-execution component, separate from the V4.1
paper-trading and decision-journal flow. The V4.1 system uses only:
  - nurex42.timeutil
  - nurex42.storage
  - nurex42.decision (via v4_1_intraday.pipeline_id)
  - nurex42.models (via v4_1_intraday.pipeline_id)

Bytecode metadata:
  magic : 0x0e2b (Python 3.14)
  size  : 11187 bytes source
  ts    : 2026-09-16 22:11:01 UTC
"""

# This stub raises ImportError to prevent silent failures.
# Replace with the real source when available.
raise ImportError(
    "nurex42.live source not available — Python 3.14 bytecode only. "
    "See docstring for recovery instructions."
)
