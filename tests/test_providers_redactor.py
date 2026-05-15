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


def test_redacts_huggingface_token() -> None:
    r = Redactor.default()
    redacted = r.redact("HF_TOKEN=hf_abcdefghijklmnopqrstuvwxyz")
    assert "hf_abc" not in redacted


def test_redacts_github_pat() -> None:
    r = Redactor.default()
    redacted = r.redact("GH_TOKEN=ghp_abcdefghijklmnopqrstuvwxyz1234567890")
    assert "ghp_abc" not in redacted


def test_redacts_slack_bot_token() -> None:
    r = Redactor.default()
    redacted = r.redact("xoxb-1234567890-deadbeef0001")
    assert "xoxb-" not in redacted


def test_redacts_jwt() -> None:
    r = Redactor.default()
    jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0In0.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
    redacted = r.redact(f"Authorization is {jwt} done")
    assert "eyJ" not in redacted


def test_bearer_redaction_does_not_truncate_at_padding() -> None:
    """A base64url token containing `=` or `+` should be redacted whole, not cut."""
    r = Redactor.default()
    redacted = r.redact("Bearer abc/def+ghi=jkl")
    # Whole token gone, not just the prefix.
    assert "abc/def+ghi" not in redacted
    assert "=jkl" not in redacted
