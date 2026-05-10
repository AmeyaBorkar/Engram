"""Internal security testing artifacts.

Private (`_`-prefixed) module: the contents are stable enough for
in-tree consumption but not part of the public API. Stage 5+ uses the
prompt-injection corpus to regression-test the consolidation prompt.
"""

from engram._security.prompt_injection import CORPUS, InjectionAttempt

__all__ = ["CORPUS", "InjectionAttempt"]
