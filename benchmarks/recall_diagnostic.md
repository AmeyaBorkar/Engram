# Retrieval-recall diagnostic — refining the path to 90%

**Run analyzed:** n=500 v3a SOTA manifest `20260518T033410_441206+0000-bb7c8412-dirty-longmemeval.json`
**Initial claim:** 16 questions failed because the model said "did not mention X" when gold IS in haystack — a retrieval-recall problem.
**Refined finding:** Only 6 of the 16 are pure retrieval failures. The rest split between multi-hop reasoning (7) and coincidental string-matches (2). One is already counted in Bucket A judge-strict FN.

## Decomposition of the 16 abstain-when-shouldn't failures

### Category 1 — True retrieval failures (6): gold buried in off-topic session

In each case the gold IS a literal string in the haystack, inside the answer session(s), but the session's main topic is unrelated. Embedding retrieval scores the whole session against the query; with the gold being a side mention, the session loses to other (wrong) sessions whose topic embedding aligns better with the query.

| qid | qtype | Q topic | Answer session topic | Gold |
|---|---|---|---|---|
| `51a45a95` | sss-user | "$5 coupon coffee creamer" | coupon organization tips | "Target" (mentioned via Cartwheel app) |
| `5d3d2817` | sss-user | "previous occupation" | project management tools | "Marketing specialist at small startup" |
| `ec81a493` | sss-user | "favorite artist debut album copies" | vinyl record storage | "500" |
| `71017277` | temporal | "jewelry received Saturday from whom" | antique furniture maintenance | "my aunt" |
| `gpt4_468eb064` | temporal | "lunch Tuesday with whom" | digital marketing workshop | "Emma" |
| `a2f3aa27` | KU | "Instagram followers now" | Instagram caption help | "1300" (sess #39, latest) |

### Category 2 — Multi-hop reasoning (7): answer must be derived

Retrieval likely *did* find relevant sessions, but the model needed to compute or synthesize. The gold value is not literally present.

| qid | qtype | Reasoning required | Gold |
|---|---|---|---|
| `7024f17c` | MS | Sum jogging (30 min) + yoga (?) last week, filtered by date | "0.5 hours" |
| `37f165cf` | MS | Sum page counts of two books finished Jan + Mar | "856" (440 + 416) |
| `b46e15ed` | temporal | Months between consecutive-day charity events and now | "2" |
| `gpt4_a1b77f9c` | temporal | Sum reading durations across 3 books in 6 sessions | "8 weeks" |
| `gpt4_468eb063` | temporal | Days between Emma lunch (Apr 11) and question_date | "9 days ago" |
| `0bc8ad93` | temporal | Resolve "2 months ago" anchor + scan for "friend" | "No, you did not" |
| `6e984302` | temporal | "4 weeks ago" anchor → session #40 (Mar 4) → buy event | "sculpting tools" |
| `d01c6aa8` | temporal | age = 32 − 5 (years in US) | "27" |

### Category 3 — Coincidental (2): not retrieval, model just answered wrong

| qid | qtype | Why | Gold |
|---|---|---|---|
| `7405e8b1` | MS | "Yes." matches everywhere; real failure is yes/no reasoning | "Yes." |
| `1cea1afa` | KU | "600" appears in unrelated sessions; covered in judge-strict FN list | "600" |

## Implications for Tier 2 retrieval fixes

The original estimate ("+8-10 questions from --min-sessions-in-topk + higher k") was overcounted. Refined:

| Fix | Recovers | Mechanism |
|---|---:|---|
| Sub-session chunking (chunk inside sessions, not whole sessions) | +3 to +5 | Off-topic side mentions get their own embedding; gold sentence isn't averaged out |
| `--min-sessions-in-topk 5` | +1 to +2 (overlap with above) | Forces session diversity; even if one session dominates, others appear |
| Higher k = 30-50 | +1 to +3 (overlap) | Brute-force more candidates |
| Hybrid BM25 + vector | +1 to +3 | Catches literal "Target", "1300", "500" strings vector misses |
| Temporal-anchor pre-resolution ("last Tuesday" → date) | +2 to +4 | Helps Category 2 multi-hop temporal |

**Realistic Tier 2 total (accounting for overlap): +4 to +7**

## Revised path-to-90 forecast

| Tier | Recovery (after overlap) | Cumulative |
|---|---:|---:|
| Baseline | 433 / 498 = 86.95% | 86.95% |
| Tier 1 (judge re-pass + 2 bug fixes) | +9 to +13 | **88.96 — 89.56%** |
| Tier 2 (sub-session chunking + min-sessions-in-topk) | +4 to +7 | **89.76 — 90.96%** |
| Tier 3 (temporal qtype hint + verification step + KU specificity) | +3 to +6 | **90.36 — 92.17%** |

**90% on accuracy_correct (449/498 = 90.16%) is achievable with Tier 1 + half of Tier 2.**

The pure retrieval bucket is smaller than initially estimated, but a substantial bottleneck remains in **multi-hop temporal reasoning** (Category 2 — 7 of 16) — these need both retrieval improvements AND prompt/tool support.

## Highest-leverage next step

**Sub-session chunking** is the single highest-leverage retrieval change because:
- It directly addresses Category 1 (the 6 buried-mention failures)
- It also indirectly helps Category 2 (more chunks → more relevant context)
- It doesn't require qtype-conditional logic (the worst-case scenario for protocol purity)
- It composes well with reranking (BGE reranker is already in pipeline)

Recommended chunk size: ~3-5 turns or ~512 tokens, whichever is smaller. Current is whole-session.
