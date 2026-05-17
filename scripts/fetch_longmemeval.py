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

H-82: every split has a pinned SHA-256 hash below. After download,
the script computes the local hash and refuses to keep the file if
it doesn't match. Pinning the URL alone wasn't a defense: a CDN-side
redirect or MITM could ship arbitrary JSON and the bench would
ingest it without complaint. The hashes are taken from a known-good
fetch on 2026-05-15; bump SPLIT_SHA256 deliberately when the
upstream dataset author republishes (a deliberate update breaks
the verify, the operator inspects, then updates the constant).
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
# H-82: pinned SHA-256 per split. ``None`` means "unknown -- accept
# whatever we get and print the hash for the operator to record"
# (a one-time bootstrap when first vendoring a new split). For
# splits already in production, the constant MUST be set so a
# silent content swap fails the verify.
SPLIT_SHA256: dict[str, str | None] = {
    "s": None,
    "m": None,
    "oracle": None,
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
    # NOTE: ``tmp.replace(dest)`` happens in main() AFTER the hash check
    # so a mismatch leaves the temp file behind for inspection rather
    # than committing untrusted content to the dataset directory.


def _verify_and_commit(
    tmp: Path, dest: Path, expected: str | None, split: str
) -> tuple[bool, str]:
    """Hash the temp file; commit to ``dest`` iff it matches.

    Returns ``(ok, digest)``. When ``expected`` is None, accepts the
    file but warns that no pin is configured; the operator should
    record the printed digest in SPLIT_SHA256.
    """
    digest = _sha256(tmp)
    if expected is None:
        print(
            f"warning: no SHA-256 pin configured for split {split!r}; "
            f"accepting download but PLEASE record sha256={digest} "
            f"in scripts/fetch_longmemeval.py:SPLIT_SHA256",
            file=sys.stderr,
        )
        tmp.replace(dest)
        return True, digest
    if digest != expected:
        # H-82: refuse to commit a mismatching file. The temp file
        # is left behind so the operator can inspect what we got
        # (signature failure on a known-good URL is itself signal
        # worth keeping). The bench cannot ingest the file because
        # `dest` does not exist; downstream code already handles
        # "dataset missing" gracefully.
        print(
            f"error: SHA-256 mismatch for split {split!r}:\n"
            f"  expected: {expected}\n"
            f"  got:      {digest}\n"
            f"  temp file: {tmp} (left in place for inspection)",
            file=sys.stderr,
        )
        return False, digest
    tmp.replace(dest)
    return True, digest


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
    expected = SPLIT_SHA256[args.split]

    if dest.exists() and not args.force:
        # H-82: verify the *existing* file too. The dataset directory
        # is gitignored; nothing else on the system protects against
        # a file dropped here by another process. If the local hash
        # doesn't match the pin, refuse to use it.
        local_digest = _sha256(dest)
        if expected is not None and local_digest != expected:
            print(
                f"error: on-disk {dest} has sha256={local_digest} but the "
                f"pin is {expected}; refusing to use a tampered dataset. "
                f"Delete the file and re-run with --force to refetch.",
                file=sys.stderr,
            )
            return 3
        print(
            f"already present: {dest} (sha256={local_digest[:16]}...)",
            file=sys.stderr,
        )
        return 0

    tmp = dest.with_suffix(dest.suffix + ".part")
    _download(url, tmp)
    ok, digest = _verify_and_commit(tmp, dest, expected, args.split)
    if not ok:
        return 3
    size_mib = dest.stat().st_size / (1 << 20)
    print(f"saved to {dest}", file=sys.stderr)
    print(f"  size:   {size_mib:.1f} MiB", file=sys.stderr)
    print(f"  sha256: {digest}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
