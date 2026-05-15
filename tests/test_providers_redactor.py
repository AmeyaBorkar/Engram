"""Tests for `engram.providers.Redactor`."""

from __future__ import annotations

import re

from engram.providers import Redactor


def test_redacts_openai_key() -> None:
    r = Redactor.default()
    redacted = r.redact("Use sk-abcdefghijklmnopqrstuvwxyz to authenticate.")
    assert "sk-abc" not in redacted
    assert "[REDACTED]" in redacted


def test_redacts_anthropic_key() -> None:
    r = Redactor.default()
    redacted = r.redact("ANTHROPIC_API_KEY=sk-ant-api01-abcdefghijklmnop12345")
    assert "sk-ant" not in redacted


def test_redacts_aws_access_key() -> None:
    r = Redactor.default()
    redacted = r.redact("AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE")
    assert "AKIA" not in redacted


def test_redacts_bearer_token() -> None:
    r = Redactor.default()
    redacted = r.redact("Authorization: Bearer abc.def.ghi-123")
    assert "abc.def.ghi" not in redacted


def test_redacts_bearer_token_with_base64_chars() -> None:
    """Base64 / JWT payloads contain `=`, `+`, `/` -- the bearer pattern
    must cover the full token, not truncate at the first such char."""
    r = Redactor.default()
    # JWT-shaped: three dot-separated base64url segments, trailing `=`
    redacted = r.redact("Bearer eyJhbGc.eyJzdWI=.signaturewith+slash/here==")
    assert "eyJ" not in redacted
    assert "signature" not in redacted
    assert "==" not in redacted


def test_redacts_authorization_header_arbitrary_case() -> None:
    """Authorization headers should be scrubbed regardless of case + scheme."""
    r = Redactor.default()
    for line in [
        "AUTHORIZATION: Token abc123xyz456def",
        "authorization: ApiKey deadbeefcafe1234",
        "Authorization:Basic dXNlcjpwYXNzd29yZA==",
    ]:
        redacted = r.redact(line)
        assert "abc123" not in redacted
        assert "deadbeef" not in redacted
        assert "dXNlcjp" not in redacted


def test_redacts_x_api_key_header() -> None:
    r = Redactor.default()
    for line in [
        "x-api-key: abc123def456ghi",
        "X-API-Key: longtokenvalue9876",
        "anthropic-api-key=sk-ant-api01-abcdefghijklmnop12345",
        "openai-api-key: sk-thisIsLongEnoughToMatch12345",
    ]:
        redacted = r.redact(line)
        assert "abc123" not in redacted
        assert "longtoken" not in redacted
        # The sk-ant rule fires first on the anthropic line and that's fine.
        assert "sk-ant-api01-abc" not in redacted
        assert "sk-this" not in redacted


def test_redacts_email() -> None:
    r = Redactor.default()
    redacted = r.redact("Contact ameya@example.com please.")
    assert "ameya@example" not in redacted


def test_redacts_phone() -> None:
    r = Redactor.default()
    redacted = r.redact("Call 555-123-4567 today.")
    assert "555-123-4567" not in redacted


def test_redacts_ssn() -> None:
    r = Redactor.default()
    redacted = r.redact("SSN 123-45-6789 on file.")
    assert "123-45-6789" not in redacted


def test_redacts_credit_card_shaped_digits() -> None:
    r = Redactor.default()
    redacted = r.redact("Card 4242 4242 4242 4242 charged.")
    assert "4242 4242 4242 4242" not in redacted


def test_leaves_normal_prose_alone() -> None:
    r = Redactor.default()
    text = "User has a golden retriever named Max."
    assert r.redact(text) == text


def test_custom_patterns_via_from_patterns() -> None:
    r = Redactor.from_patterns([r"hidden-\d+"], replacement="<gone>")
    assert r.redact("see hidden-42 here") == "see <gone> here"


def test_from_patterns_accepts_compiled_regex() -> None:
    r = Redactor.from_patterns([re.compile(r"X+", re.IGNORECASE)], replacement="?")
    assert r.redact("aXxX b") == "a? b"


def test_redact_obj_recurses_into_dict_and_list() -> None:
    r = Redactor.default()
    payload = {
        "user": "ameya@example.com",
        "messages": [
            "hi",
            "my key is sk-thisIsLongEnoughToMatch12345",
        ],
        "nested": {"phone": "555-123-4567"},
        "count": 3,
    }
    out = r.redact_obj(payload)
    assert "ameya@example.com" not in str(out)
    assert "sk-this" not in str(out)
    assert "555-123-4567" not in str(out)
    assert out["count"] == 3  # non-string passes through


def test_redact_obj_passes_through_non_strings() -> None:
    r = Redactor.default()
    assert r.redact_obj(42) == 42
    assert r.redact_obj(None) is None
    assert r.redact_obj(3.14) == 3.14
    assert r.redact_obj((1, "a", 2)) == (1, "a", 2)


def test_anthropic_pattern_wins_over_generic_sk() -> None:
    """Both patterns would match an Anthropic key; the more specific one fires first
    and replaces the whole thing in one shot rather than leaving partial text."""
    r = Redactor.default()
    redacted = r.redact("sk-ant-api01-abcdefghijklmnop12345")
    assert redacted == "[REDACTED]"
