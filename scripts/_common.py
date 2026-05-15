"""Shared helpers for the `scripts/` retrieval / eval / trace battery.

Pre-audit (H-85, H-87, M-17, M-122, M-169) every script had its own
copies of `_load_env`, `_build_embedder`, `_build_reranker`, the
`_split_config` knob mapping, and the `_REPO_ROOT` sys.path setup --
six or seven near-identical functions per concern across 7 scripts.
A security fix to the `.env` parser had to be applied seven times to
land; a knob-name fix could land in one script and silently diverge
the rest. Centralising here removes that drift.

Public surface (everything else is `_`-prefixed and considered internal):

  * ``ensure_repo_on_path()``     -- adds repo root + ``src/`` to sys.path
  * ``load_env_file(path)``       -- best-effort .env loader (uses python-dotenv
                                     when available, falls back to a built-in
                                     parser with proper paired-quote handling)
  * ``build_embedder(...)``       -- LocalEmbedder factory with dtype mapping
  * ``build_reranker(...)``       -- BGEReranker factory; ``None`` to skip
  * ``build_chat(name, model)``   -- thin re-export of bench.build_chat
  * ``split_config(cfg)``         -- standard knob mapping; underscore-prefixed
                                     keys (bench-level flags) split out separately
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any


def ensure_repo_on_path() -> Path:
    """Add the repo root and src/ to sys.path. Returns the repo root.

    Every script in scripts/ needs this so `from engram import ...`
    and `from benchmarks.suites.* import ...` resolve when the script
    is invoked directly (`python scripts/foo.py`). Centralising the
    path manipulation means a single `ensure_repo_on_path()` call
    replaces the 8-line block each script copied.
    """
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    src = repo_root / "src"
    if src.exists() and str(src) not in sys.path:
        sys.path.insert(0, str(src))
    return repo_root


def _strip_matched_quotes(value: str) -> str:
    """Strip a single matched pair of ``"..."`` or ``'...'`` quotes.

    The pre-audit fallback used ``value.strip().strip('"').strip("'")``,
    which strips ANY characters in those sets from BOTH ends -- so
    ``"it's"`` became ``it`` (lost the trailing ``'s`` because the
    second strip then ate the opening of ``'s``), and unpaired quotes
    were silently consumed. This function only strips when the first
    and last char are the SAME quote character.
    """
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
        return value[1:-1]
    return value


def load_env_file(path: Path = Path(".env")) -> bool:
    """Best-effort `.env` loader. Existing env vars take precedence.

    Returns True iff the file was found (regardless of whether any
    keys were loaded). Uses `python-dotenv` when available; otherwise
    falls back to a tiny built-in parser that handles paired-quote
    stripping correctly (M-17).
    """
    if not path.exists():
        return False
    try:
        from dotenv import load_dotenv
    except ImportError:
        with path.open("r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = _strip_matched_quotes(value.strip())
                if key and key not in os.environ:
                    os.environ[key] = value
        return True
    load_dotenv(path, override=False)
    return True


# Standard CLI dtype -> embedder dtype mapping. The CLI uses "fp16"
# / "fp32" as shorthand; the LocalEmbedder + BGEReranker accept the
# torch-native names. Defined once here so adding a new shorthand
# (e.g. "bf16") doesn't require editing every script.
_DTYPE_MAP: dict[str, str] = {"auto": "auto", "fp16": "float16", "fp32": "float32"}


def _resolve_dtype(dtype: str) -> str:
    return _DTYPE_MAP.get(dtype, "auto")


def build_embedder(
    model: str,
    device: str | None,
    dtype: str = "auto",
) -> Any:
    """Construct a `LocalEmbedder` with the given model/device/dtype.

    Lazy-imports `engram.providers.local` so the wider `_common` module
    has no torch dependency at import time. Caller is responsible for
    sys.path setup (call `ensure_repo_on_path()` first).
    """
    from engram.providers.local import LocalEmbedder

    return LocalEmbedder(
        model=model,
        device=device,
        dtype=_resolve_dtype(dtype),  # type: ignore[arg-type]
    )


def build_reranker(
    model: str | None,
    device: str | None,
    dtype: str = "auto",
) -> Any:
    """Construct a `BGEReranker`, or `None` if `model` is None / 'none'.

    Lazy-imports `engram.retrieve._bge_reranker` for the same reason
    as `build_embedder`.
    """
    if model is None or model.lower() == "none":
        return None
    from engram.retrieve._bge_reranker import BGEReranker

    return BGEReranker(
        model=model,
        device=device,
        dtype=_resolve_dtype(dtype),  # type: ignore[arg-type]
    )


def build_chat(chat_name: str, chat_model: str | None) -> Any:
    """Construct a chat provider via the bench's standard catalog.

    Thin re-export of `engram.bench._real_provider.build_chat`; lives
    here so scripts only need one import.
    """
    from engram.bench._real_provider import build_chat as _bench_build_chat

    return _bench_build_chat(chat_name, chat_model)


# Keys reserved for bench-level flags (NOT RetrieveParams kwargs).
# By convention these are prefixed with `_` in the CONFIGS dicts.
_BENCH_LEVEL_FLAGS: frozenset[str] = frozenset({"_auto_temporal"})


def split_config(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, bool]]:
    """Split a CONFIGS-style dict into (retrieve_params_overrides, bench_flags).

    Centralises the convention "keys starting with `_` are bench-level
    flags, everything else is a RetrieveParams field". Three scripts
    used to copy-paste their own version of this; a knob-name typo
    in one wouldn't catch in the other.

    `bench_flags` is a `dict[str, bool]` containing every recognised
    underscore-prefixed key. Currently:
      * ``_auto_temporal``: per-question year-token lexical filter.
    """
    rp_overrides: dict[str, Any] = {}
    bench_flags: dict[str, bool] = {}
    for k, v in config.items():
        if k in _BENCH_LEVEL_FLAGS:
            bench_flags[k] = bool(v)
        elif k.startswith("_"):
            # Unknown underscore-prefixed flag: record it under bench
            # but bool-cast, since RetrieveParams will reject it.
            bench_flags[k] = bool(v)
        else:
            rp_overrides[k] = v
    return rp_overrides, bench_flags
