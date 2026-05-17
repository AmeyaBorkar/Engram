"""Trace multiple LongMemEval qids in one Python process.

`scripts/retrieval_trace.py` traces ONE question per invocation; for a
list of 20-30 qids that means reloading the embedder + reranker each
time, which dominates wall time. This script does the same work but
loads the models ONCE and reuses them across all qids.

Inputs:
  - A file with one qid per line (`--qid-file`), OR a comma-separated
    `--qids` argument.
  - Output directory; each trace goes to `<dir>/trace_<qid>.txt`.

Reuses `_trace_question` from `scripts.retrieval_trace`, so the output
format is identical to per-question invocations.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# M-122: shared helpers in scripts/_common.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ensure_repo_on_path, load_env_file  # noqa: E402

ensure_repo_on_path()
load_env_file()

from engram import Memory, SqliteStorage  # noqa: E402
from engram.providers._fake import FakeChat  # noqa: E402

from benchmarks.suites.longmemeval import (  # noqa: E402
    DATASET_ROOT,
    DEFAULT_FILENAME,
    _ingest_haystack,
    _load_dataset,
)
from scripts.retrieval_trace import (  # noqa: E402
    _build_embedder,
    _build_reranker,
    _trace_question,
)

_LOG = logging.getLogger("engram.batch_trace")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Batch-trace many LongMemEval qids in one process."
    )
    parser.add_argument("--qid-file", type=Path, default=None)
    parser.add_argument("--qids", default=None, help="Comma-separated.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Each trace goes to <dir>/trace_<qid>.txt.",
    )
    parser.add_argument(
        "--configs",
        default="baseline",
        help="Comma-separated configs to trace per question.",
    )
    parser.add_argument("--embed-model", default="BAAI/bge-large-en-v1.5")
    parser.add_argument("--embed-device", default=None)
    parser.add_argument("--dtype", default="fp32", choices=("auto", "fp16", "fp32"))
    parser.add_argument("--reranker", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--deep-n", type=int, default=50)
    parser.add_argument("--log-level", default=os.environ.get("ENGRAM_LOG_LEVEL", "INFO"))
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )

    # Resolve qids
    qids: list[str] = []
    if args.qid_file:
        for line in args.qid_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                qids.append(line)
    if args.qids:
        qids.extend(q.strip() for q in args.qids.split(",") if q.strip())
    if not qids:
        print("no qids provided (use --qid-file or --qids)", file=sys.stderr)
        return 2

    config_names = [c.strip() for c in args.configs.split(",") if c.strip()]

    # Load dataset and index by qid
    all_questions = _load_dataset(DATASET_ROOT / DEFAULT_FILENAME)
    by_qid = {q.qid: q for q in all_questions}

    # Resolve qid prefixes (accept partial matches)
    resolved: list[str] = []
    for qid in qids:
        if qid in by_qid:
            resolved.append(qid)
            continue
        matches = [k for k in by_qid if k.startswith(qid)]
        if len(matches) == 1:
            resolved.append(matches[0])
        elif len(matches) == 0:
            _LOG.warning("qid not found: %s", qid)
        else:
            _LOG.warning("qid prefix ambiguous (%d matches): %s", len(matches), qid)
            resolved.append(matches[0])

    if not resolved:
        print("no valid qids resolved", file=sys.stderr)
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _LOG.info("Loading embedder + reranker once for %d qids...", len(resolved))
    embedder = _build_embedder(args.embed_model, args.embed_device, args.dtype)
    reranker = _build_reranker(args.reranker, args.embed_device, args.dtype)
    chat = FakeChat()

    t_start = time.perf_counter()
    for i, qid in enumerate(resolved, start=1):
        q = by_qid[qid]
        storage = SqliteStorage(":memory:")
        storage.initialize()
        try:
            memory = Memory(storage=storage, embedder=embedder, chat=chat, reranker=reranker)
            t_ingest = time.perf_counter()
            turns = _ingest_haystack(memory, q)
            ingest_s = time.perf_counter() - t_ingest
            t_trace = time.perf_counter()
            text = _trace_question(
                q=q,
                storage=storage,
                embedder=embedder,
                reranker=reranker,
                chat=chat,
                config_names=config_names,
                deep_n=args.deep_n,
                k=args.k,
            )
            trace_s = time.perf_counter() - t_trace
            out = args.output_dir / f"trace_{qid}.txt"
            out.write_text(text, encoding="utf-8")
            _LOG.info(
                "%3d/%3d qid=%s [%s] turns=%d ingest=%.1fs trace=%.1fs -> %s",
                i,
                len(resolved),
                qid[:12],
                q.qtype,
                turns,
                ingest_s,
                trace_s,
                out.name,
            )
        finally:
            storage.close()

    total = time.perf_counter() - t_start
    _LOG.info("Done: %d traces in %.1f min", len(resolved), total / 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
