"""Regression tests for engram._prompt_util.

These pin the security-critical invariants:
  - `inline` collapses every line-breaking character (LF, CR, CRLF,
    U+2028, U+2029, TAB) and double-escapes backslash so the escapes
    survive.
  - `strip_code_fence` pulls ```json ... ``` content out unchanged,
    passes through unfenced input.
  - `render_prompt` does ONE substitution pass — user content in slot
    `a` cannot impersonate a literal `{b}` and steer the second pass.
"""

from __future__ import annotations

from engram._prompt_util import inline, render_prompt, strip_code_fence


class TestInline:
    def test_collapses_newlines(self) -> None:
        assert inline("a\nb") == "a\\nb"

    def test_collapses_crlf(self) -> None:
        assert inline("a\r\nb") == "a\\nb"  # CRLF -> one \n, not two

    def test_collapses_cr_alone(self) -> None:
        assert inline("a\rb") == "a\\nb"

    def test_collapses_tab(self) -> None:
        assert inline("a\tb") == "a\\tb"

    def test_collapses_u2028_line_separator(self) -> None:
        assert inline("a" + chr(0x2028) + "b") == "a\\nb"

    def test_collapses_u2029_paragraph_separator(self) -> None:
        assert inline("a" + chr(0x2029) + "b") == "a\\nb"

    def test_escapes_backslash_first(self) -> None:
        # If we escaped \n before \\, the resulting \\n would be re-escaped
        # to \\\\n. Backslash MUST be handled first.
        assert inline("a\\nb") == "a\\\\nb"

    def test_idempotent_on_safe_text(self) -> None:
        assert inline("normal text 123") == "normal text 123"


class TestStripCodeFence:
    def test_json_fence(self) -> None:
        assert strip_code_fence('```json\n{"x":1}\n```') == '{"x":1}'

    def test_uppercase_fence(self) -> None:
        assert strip_code_fence('```JSON\n{"x":1}\n```') == '{"x":1}'

    def test_bare_fence(self) -> None:
        assert strip_code_fence('```\n{"x":1}\n```') == '{"x":1}'

    def test_no_fence_passes_through(self) -> None:
        assert strip_code_fence('{"x":1}') == '{"x":1}'

    def test_preamble_does_not_match(self) -> None:
        # The fence regex is anchored at both ends; a fence with prose
        # outside it falls through unchanged so the JSON parser surfaces
        # the real error.
        result = strip_code_fence("here is your json:\n```json\n{}\n```")
        assert "here is your json" in result


class TestRenderPromptSinglePass:
    """The H-03 invariant: one substitution pass, no second-replace bypass."""

    def test_simple_substitute(self) -> None:
        out = render_prompt("Q: {q}\nA: {a}", q="hello", a="world")
        assert out == "Q: hello\nA: world"

    def test_content_containing_other_placeholder_does_not_redirect(self) -> None:
        # The audit's H-03 PoC: a's content contains a literal `{b}`. A
        # chained .replace("{a}",a).replace("{b}",b) would substitute b
        # INTO a's contribution. render_prompt must NOT.
        out = render_prompt("A: {a} B: {b}", a="I am {b}", b="real-B")
        assert out == "A: I am {b} B: real-B"

    def test_unset_placeholder_left_literal(self) -> None:
        # Partial render is common (e.g. template with `{now}` filled by
        # caller but `{query}` still missing). Don't KeyError; leave it.
        out = render_prompt("Hi {known} {unknown}", known="x")
        assert out == "Hi x {unknown}"

    def test_mapping_form(self) -> None:
        out = render_prompt("Q: {q}", {"q": "hello"})
        assert out == "Q: hello"

    def test_kwargs_win_on_collision_with_mapping(self) -> None:
        out = render_prompt("X: {x}", {"x": "from-map"}, x="from-kwargs")
        assert out == "X: from-kwargs"

    def test_empty_fields_returns_template_unchanged(self) -> None:
        assert render_prompt("Hello {x}") == "Hello {x}"
