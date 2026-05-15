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
# Per-split expected SHA256. Pre-audit there was no checksum
# verification, so a MITM, an HF redirect, or a silent content
# revision would land arbitrary JSON into `benchmarks/datasets/`
# without any signal. The script now verifies the local digest after
# download and refuses to publish a mismatched file.
#
# Empty string means "no pinned digest yet"; the script will print
# the observed digest and accept the file (skipping verification for
# splits where pinning hasn't been done). Replace the empty string
# with the canonical digest once the file is known-good. To re-pin
# after an upstream dataset bump, run with `--accept-new-checksum`
# (alias: `--force` already exists) to download AND print the new
# digest without verifying.
SPLIT_SHA256: dict[str, str] = {
    "s": "",
    "m": "",
    "oracle": "",
}
DEST_DIR = Path("benchmarks/datasets/longmemeval")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(
    url: str, dest: Path, expected_sha256: str = "", *, accept_new: bool = False
) -> str:
    """Download `url` to `dest` atomically. Verify SHA256 if pinned.

    Returns the observed sha256 hex digest. Raises RuntimeError on
    mismatch (unless `accept_new=True`, which prints the digest and
    accepts the file -- intended for pinning a fresh upstream bump).
    """
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
    observed = _sha256(tmp)
    if expected_sha256 and not accept_new:
        if observed != expected_sha256:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(
                f"sha256 mismatch for {url}\n"
                f"  expected: {expected_sha256}\n"
                f"  observed: {observed}\n"
                f"  Refusing to publish {dest}. If the upstream file legitimately "
                f"changed, re-run with `--accept-new-checksum` to download and "
                f"print the new digest; then update SPLIT_SHA256 in this script."
            )
    elif not expected_sha256:
        print(
            f"  (no pinned sha256 for this split; observed {observed[:16]}...)",
            file=sys.stderr,
        )
    tmp.replace(dest)
    return observed


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
    parser.add_argument(
        "--accept-new-checksum",
        action="store_true",
        help=(
            "Skip SHA256 verification and print the observed digest. "
            "Intended for the workflow `<upstream bumped the dataset> ->"
            " download with this flag -> paste the new digest into "
            "SPLIT_SHA256 -> commit`. Do NOT use to silence unexpected "
            "mismatches in production runs."
        ),
    )
    args = parser.parse_args()

    filename = SPLIT_FILES[args.split]
    url = f"{HF_BASE}/{filename}"
    dest = DEST_DIR / filename
    expected = SPLIT_SHA256.get(args.split, "")

    if dest.exists() and not args.force:
        local = _sha256(dest)
        if expected and local != expected:
            print(
                f"WARN: existing {dest} sha256={local[:16]}... does NOT match "
                f"the pinned digest {expected[:16]}.... Re-run with --force "
                f"to redownload.",
                file=sys.stderr,
            )
        else:
            print(
                f"already present: {dest} (sha256={local[:16]}...)",
                file=sys.stderr,
            )
        return 0

    digest = _download(
        url,
        dest,
        expected_sha256=expected,
        accept_new=args.accept_new_checksum,
    )
    size_mib = dest.stat().st_size / (1 << 20)
    print(f"saved to {dest}", file=sys.stderr)
    print(f"  size:   {size_mib:.1f} MiB", file=sys.stderr)
    print(f"  sha256: {digest}", file=sys.stderr)
    if not expected:
        print(
            f"  Pin this digest in SPLIT_SHA256[{args.split!r}] for future runs.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
