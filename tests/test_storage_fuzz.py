"""Random byte-payload fuzz: storage layer never panics on adversarial input.

Storage accepts user-controlled `content` and `metadata`. We don't trust
the network, so we don't trust the bytes — random Unicode strings, embedded
nulls, weird control characters, and oversized JSON metadata must all
roundtrip cleanly or fail with a typed error, never a segfault or a
DBAPI panic.
"""

from __future__ import annotations

import json
import os
import string
from typing import Any

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from engram.schemas import Event, Level, MemoryItem
from engram.storage import SqliteStorage

_max_examples = 100
_settings = settings(
    max_examples=_max_examples,
    deadline=4000,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)


# Generate content that includes nulls, control chars, surrogates-via-utf8,
# oversized strings, and ordinary text.
#
# Why `blacklist_categories=("Cs",)` (unpaired surrogates) is excluded:
# the SQLite Python binding encodes TEXT columns as UTF-8.  Unpaired
# surrogates (U+D800..U+DFFF as code points, not as part of a valid
# surrogate pair) are not legal Unicode and `str.encode("utf-8")`
# raises `UnicodeEncodeError` on them.  The DB-API binding surfaces
# that as an exception before the storage layer ever runs, so the
# fuzz round-trip would only ever measure "the Python encoder rejects
# these," not anything about storage behavior.
#
# This is a documented out-of-scope: the test surface is "everything
# storage accepts must round-trip cleanly," and the encoder gate is a
# layer above storage in the call stack.  If you ever lower the
# encoder bypass (e.g. switch to a binary-safe BLOB column for content),
# remove the blacklist and re-run — that's a real attack surface.
_adversarial_text = st.one_of(
    st.text(min_size=0, max_size=64),
    st.text(alphabet=string.printable, min_size=0, max_size=2048),
    st.text(
        alphabet=st.characters(
            blacklist_categories=("Cs",),  # see docstring above
        ),
        min_size=0,
        max_size=512,
    ),
)


@given(content=_adversarial_text)
@_settings
def test_event_content_roundtrip_fuzz(storage: SqliteStorage, content: str) -> None:
    e = Event(content=content)
    storage.insert_event(e)
    got = storage.get_event(e.id)
    assert got is not None
    assert got.content == content


_json_value: st.SearchStrategy[Any] = st.recursive(
    st.one_of(
        st.none(),
        st.booleans(),
        st.integers(min_value=-(2**53), max_value=2**53),
        st.floats(allow_nan=False, allow_infinity=False, width=64),
        st.text(min_size=0, max_size=64),
    ),
    lambda children: st.one_of(
        st.lists(children, max_size=5),
        st.dictionaries(st.text(min_size=0, max_size=16), children, max_size=5),
    ),
    max_leaves=15,
)


@given(metadata=st.dictionaries(st.text(max_size=16), _json_value, max_size=10))
@_settings
def test_metadata_arbitrary_json_roundtrip(
    storage: SqliteStorage, metadata: dict[str, Any]
) -> None:
    e = Event(content="x", metadata=metadata)
    storage.insert_event(e)
    got = storage.get_event(e.id)
    assert got is not None
    # Compare via JSON canonicalization to handle float NaN-style edge cases
    # (already excluded by strategy, but defensive).
    assert json.dumps(got.metadata, sort_keys=True) == json.dumps(metadata, sort_keys=True)


def test_random_byte_payloads_in_memory_item_content(storage: SqliteStorage) -> None:
    """Memory item content with random bytes-as-strings (errors='replace') survives roundtrip."""
    for _ in range(50):
        raw = os.urandom(64)
        text = raw.decode("utf-8", errors="replace")
        item = MemoryItem(level=Level.EVENT, content=text)
        storage.insert_memory_item(item)
        got = storage.get_memory_item(item.id)
        assert got is not None
        assert got.content == text
