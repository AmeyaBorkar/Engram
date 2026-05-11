"""Stage 8 reconciliation surface.

A `Reconciler` resolves detected `Conflict` rows: it picks a winner
per the chosen `Resolution` policy, invalidates the loser through the
storage layer, and flips the conflict row OPEN -> RESOLVED.

Public surface lives on `Memory.reconcile`; this module is the engine
behind it (the same shape as `decay/_engine.py` and
`consolidation/_engine.py`).
"""

from engram.reconcile._engine import Reconciler

__all__ = ["Reconciler"]
