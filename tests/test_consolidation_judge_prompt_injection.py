"""Stage 8 prompt-injection regression suite for the judge prompt.

Scope (per cross-cutting standards): "Prompt-injection corpus: events
that try to claim system-instruction status do not get promoted." The
abstraction prompt has `test_consolidation_prompt_injection.py`; this
file is the matching corpus for the judge prompt that decides whether
two memory items contradict.

The judge has a different attack surface from the abstraction prompt:

  * The output is a *closed enum* (`agree` / `contradict` / `unrelated`),
    so injection cannot smuggle arbitrary text into the result.
  * The risk is **verdict manipulation**: trick the judge into returning
    `unrelated` when two statements actually CONTRADICT, so the conflict
    never gets recorded -> silent contradiction sneaks into the
    hierarchy.
  * The risk is also **denial of service via parse failure**: malformed
    judge responses fall back to `UNRELATED` (the safe default in
    `judge(...)`), which is the same effect as a successful verdict-
    manipulation attack on a contradiction.

Three layers exercised:

  1. Prompt rendering treats A/B as DATA. `CRITICAL RULES` precedes
     the `STATEMENTS` block; newlines/tabs in payloads are escaped.
  2. Parse-time rejection. Malformed JSON, missing `verdict`, invalid
     enum values all raise `AbstractionParseError`. The `judge(...)`
     wrapper catches these and returns `Verdict.UNRELATED`.
  3. End-to-end. Every CORPUS payload, planted as statement A or B,
     produces a structured prompt; the judge function never crashes.
"""

from __future__ import annotations

import json

import pytest

from engram._security.prompt_injection import CORPUS
from engram.consolidation import (
    AbstractionParseError,
    Verdict,
    judge,
    parse_judge_response,
    render_judge_prompt,
)
from engram.providers._cache import content_hash
from engram.providers._fake import FakeChat

SAFE_A = "The user owns a golden retriever named Rex."
SAFE_B = "The user is allergic to cats."


# ---------------------------------------------------------------------------
# 1. Prompt rendering treats A and B as DATA
# ---------------------------------------------------------------------------


class TestJudgePromptRenderingIsHardened:
    def test_critical_rules_precede_statements(self) -> None:
        """Hardening header must come before the STATEMENTS block."""
        for attack in CORPUS:
            prompt = render_judge_prompt(a=attack.payload, b=SAFE_B)
            rules_idx = prompt.find("CRITICAL RULES")
            statements_idx = prompt.find("STATEMENTS")
            assert rules_idx >= 0, "judge prompt missing CRITICAL RULES header"
            assert statements_idx > rules_idx, (
                f"{attack.name}: STATEMENTS must appear after CRITICAL RULES"
            )

    def test_newlines_in_payload_are_escaped(self) -> None:
        """Multi-line payloads must be inlined so they cannot inject a
        new section heading like '\n\nSYSTEM:'."""
        for attack in CORPUS:
            if "\n" not in attack.payload:
                continue
            prompt = render_judge_prompt(a=attack.payload, b=SAFE_B)
            # The raw newline from the payload must not appear inside the
            # STATEMENTS block as a new line; the literal `\n` escape
            # sequence must be there instead.
            statements_block = prompt[prompt.find("STATEMENTS"):]
            assert "\\n" in statements_block, (
                f"{attack.name}: newline not escaped in judge prompt"
            )

    def test_payload_appears_only_in_statement_slots(self) -> None:
        """The injection payload must land in the A: or B: slot, never
        before CRITICAL RULES."""
        for attack in CORPUS:
            prompt = render_judge_prompt(a=attack.payload, b=SAFE_B)
            # The CRITICAL RULES block is the first textual content the
            # model sees; payload (even when escaped) can't appear there.
            rules_block = prompt[: prompt.find("STATEMENTS")]
            # Compare on a sentinel substring that's unique to each payload
            # but would still appear post-escape (use a non-newline char).
            sentinel = attack.payload.split("\n", 1)[0][:30]
            if len(sentinel) >= 5 and "{" not in sentinel:
                assert sentinel not in rules_block, (
                    f"{attack.name}: payload sentinel {sentinel!r} leaked "
                    "into CRITICAL RULES block"
                )

    def test_chat_template_tokens_are_preserved_as_literals(self) -> None:
        """The `chat_template_injection` payload contains <|im_start|>
        tokens. They must appear as literal text in the rendered prompt
        - the downstream tokenizer should not interpret them as actual
        chat boundaries."""
        attack = next(a for a in CORPUS if a.name == "chat_template_injection")
        prompt = render_judge_prompt(a=attack.payload, b=SAFE_B)
        assert "<|im_start|>" in prompt
        assert "<|im_end|>" in prompt
        # Most importantly, the tokens appear only inside the STATEMENTS
        # block (escaped via _inline keeps newlines from breaking the
        # prompt structure but tokens stay literal).
        statements_block = prompt[prompt.find("STATEMENTS"):]
        assert "<|im_start|>" in statements_block


# ---------------------------------------------------------------------------
# 2. Parse-time rejection
# ---------------------------------------------------------------------------


class TestParseRejectsMalformedJudgeResponses:
    def test_empty_response_raises(self) -> None:
        with pytest.raises(AbstractionParseError, match="empty"):
            parse_judge_response("")
        with pytest.raises(AbstractionParseError, match="empty"):
            parse_judge_response("   \n\t  ")

    def test_non_json_raises(self) -> None:
        with pytest.raises(AbstractionParseError, match="invalid judge JSON"):
            parse_judge_response("the verdict is contradict")
        with pytest.raises(AbstractionParseError, match="invalid judge JSON"):
            parse_judge_response("contradict")

    def test_missing_verdict_field_raises(self) -> None:
        with pytest.raises(AbstractionParseError, match="schema mismatch"):
            parse_judge_response(json.dumps({"reason": "they agree"}))

    def test_invalid_verdict_value_raises(self) -> None:
        with pytest.raises(AbstractionParseError, match="schema mismatch"):
            parse_judge_response(json.dumps({"verdict": "definitely_yes"}))

    def test_extra_fields_are_ignored(self) -> None:
        """The model is permissive on extras; only `verdict` matters."""
        out = parse_judge_response(
            json.dumps({"verdict": "contradict", "INJECTED": "extra"})
        )
        assert out.verdict is Verdict.CONTRADICT

    def test_strips_code_fence(self) -> None:
        fenced = "```json\n" + json.dumps({"verdict": "agree"}) + "\n```"
        out = parse_judge_response(fenced)
        assert out.verdict is Verdict.AGREE

    def test_case_sensitivity_of_verdict_enum(self) -> None:
        """The verdict enum values are lowercase. The judge should
        reject upper-case variants that don't match exactly -- callers
        shouldn't have to second-guess casing rules."""
        with pytest.raises(AbstractionParseError, match="schema mismatch"):
            parse_judge_response(json.dumps({"verdict": "CONTRADICT"}))


# ---------------------------------------------------------------------------
# 3. `judge()` falls back to UNRELATED on parse failure
# ---------------------------------------------------------------------------


class TestJudgeFallbackOnFailure:
    def test_corrupt_response_falls_back_to_unrelated(self) -> None:
        """The safe default on parse failure is UNRELATED (we'd rather
        miss a real contradiction than flag a spurious one). This is
        the documented behavior of `judge(...)`."""
        # FakeChat returns garbage; judge() should retry up to
        # max_retries then fall back.
        chat = FakeChat(default="garbage that is not json")
        verdict = judge(a=SAFE_A, b=SAFE_B, chat=chat, max_retries=1)
        assert verdict is Verdict.UNRELATED

    def test_retries_then_falls_back(self) -> None:
        """If the first response is malformed, the wrapper retries.
        After all retries fail, the verdict is UNRELATED."""
        chat = FakeChat(default="{}")  # schema mismatch every time
        verdict = judge(a=SAFE_A, b=SAFE_B, chat=chat, max_retries=2)
        assert verdict is Verdict.UNRELATED

    def test_successful_parse_returns_actual_verdict(self) -> None:
        """Sanity: a well-formed CONTRADICT response surfaces unchanged."""
        prompt = render_judge_prompt(a=SAFE_A, b=SAFE_B)
        chat = FakeChat(
            scripts={content_hash(prompt): json.dumps({"verdict": "contradict"})}
        )
        verdict = judge(a=SAFE_A, b=SAFE_B, chat=chat, max_retries=0)
        assert verdict is Verdict.CONTRADICT


# ---------------------------------------------------------------------------
# 4. End-to-end: every CORPUS payload survives prompt assembly + judge call
# ---------------------------------------------------------------------------


class TestEndToEndJudgeAgainstInjectionCorpus:
    """End-to-end behavior of judge() against the prompt-injection corpus.

    Post-M-04: the judge short-circuits to UNRELATED when either
    statement looks like an injection attempt, before the chat call.
    The previous contract (LLM is trusted to be robust; FakeChat reply
    is surfaced verbatim) was permissive — feeding an obviously-hostile
    payload through the LLM bills tokens for a verdict we discard, and
    gives the model a chance to leak system context through future
    reasoning fields if the schema is widened.
    """

    def test_every_attack_as_statement_a_short_circuits(self) -> None:
        """For every entry in the shared CORPUS, planting it as A
        forces judge() to return UNRELATED without ever calling chat.
        """
        for attack in CORPUS:
            # FakeChat will return AGREE if asked — but the
            # short-circuit means chat is never called.
            chat = FakeChat(default=json.dumps({"verdict": "agree"}))
            verdict = judge(
                a=attack.payload, b=SAFE_B, chat=chat, max_retries=0
            )
            assert verdict is Verdict.UNRELATED, attack.name

    def test_every_attack_as_statement_b_short_circuits(self) -> None:
        """Symmetric: payload as B, benign as A."""
        for attack in CORPUS:
            chat = FakeChat(default=json.dumps({"verdict": "agree"}))
            verdict = judge(
                a=SAFE_A, b=attack.payload, chat=chat, max_retries=0
            )
            assert verdict is Verdict.UNRELATED, attack.name

    def test_injection_in_both_statements_short_circuits(self) -> None:
        """Both statements being injection payloads also short-circuits."""
        a_attack = next(a for a in CORPUS if a.name == "ignore_prior_instructions")
        b_attack = next(a for a in CORPUS if a.name == "fake_system_role")
        chat = FakeChat(default=json.dumps({"verdict": "contradict"}))
        verdict = judge(
            a=a_attack.payload,
            b=b_attack.payload,
            chat=chat,
            max_retries=0,
        )
        assert verdict is Verdict.UNRELATED
