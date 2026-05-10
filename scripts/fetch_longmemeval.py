"""Download the LongMemEval-S split into ``benchmarks/datasets/longmemeval/``.

Source: HuggingFace ``xiaowu0162/longmemeval-cleaned`` (the cleaned variant
that the dataset author recommends; supersedes the original
``xiaowu0162/longmemeval`` which had noisy haystack sessions).

Run:
    python scripts/fetch_longmemeval.py            # fetches longmemeval_s_cleaned.json
    python scripts/fetch_longmemeval.py --split m  # fetches the much larger _m split
    python scripts/fetch_longmemeval.py --split oracle

The dataset is research-licensed and not vendored in this repo;
``benchmarks/datasets/`` is gitignored. Re-running this script is a
no-op when the file already exists (use ``--force`` to overwrite).

Hugging Face Hub auth is NOT required for this dataset. If the request
returns 401, set ``HF_TOKEN`` in your shell.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

HF_BASE = "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main"
SPLIT_FILES: dict[str, str] = {
    "s": "longmemeval_s_cleaned.json",
    "m": "longmemeval_m_cleaned.json",
    "oracle": "longmemeval_oracle.json",
}
DEST_DIR = Path("benchmarks/datasets/longmemeval")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"  fetching {url}", file=sys.stderr)
    with urllib.request.urlopen(url) as resp, tmp.open("wb") as f:  # noqa: S310 - pinned URL
        total = 0
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            total += len(chunk)
            print(f"\r    {total / (1 << 20):.1f} MiB", end="", file=sys.stderr)
        print("", file=sys.stderr)
    tmp.replace(dest)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split",
        choices=sorted(SPLIT_FILES),
        default="s",
        help="LongMemEval split to download (default: s, the 500-question evaluation set).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the destination file exists.",
    )
    args = parser.parse_args()

    filename = SPLIT_FILES[args.split]
    url = f"{HF_BASE}/{filename}"
    dest = DEST_DIR / filename

    if dest.exists() and not args.force:
        print(
            f"already present: {dest} (sha256={_sha256(dest)[:16]}...)",
            file=sys.stderr,
        )
        return 0

    _download(url, dest)
    digest = _sha256(dest)
    size_mib = dest.stat().st_size / (1 << 20)
    print(f"saved to {dest}", file=sys.stderr)
    print(f"  size:   {size_mib:.1f} MiB", file=sys.stderr)
    print(f"  sha256: {digest}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
