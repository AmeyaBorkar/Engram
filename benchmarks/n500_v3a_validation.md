# Manifest comparison report

- **Baseline**: `20260516T224729_100252+0000-e503e185-dirty-longmemeval.json`
- **New**: `20260518T033410_441206+0000-bb7c8412-dirty-longmemeval.json`
- **Baseline commit**: `e503e18`
- **New commit**: `bb7c841`

## Headline

- **Overall accuracy: 68.5% -> 86.9%** (+18.5pp)
- Questions: base 500 / new 500

## Per-qtype accuracy

| qtype | base | new | Δ |
|---|---:|---:|---:|
| knowledge-update | 72.7% | 89.6% | +16.9pp |
| multi-session | 60.9% | 78.9% | +18.0pp |
| single-session-assistant | 96.4% | 100.0% | +3.6pp |
| single-session-preference | 41.4% | 86.2% | +44.8pp |
| single-session-user | 80.0% | 94.3% | +14.3pp |
| temporal-reasoning | 61.7% | 84.2% | +22.6pp |

## Question-level flip analysis

- PP (still pass): 337
- FF (still fail): 63
- PF (lost wins, base passed but new failed): **4**
- FP (recovered failures, base failed but new passed): **96**
- Net flips: **+92**
- Only in base: 0
- Only in new: 0

### Per-qtype flips

| qtype | PP | FF | **PF (lost)** | **FP (gained)** | net |
|---|---:|---:|---:|---:|---:|
| knowledge-update | 54 | 7 | **2** | **15** | +13 |
| multi-session | 79 | 26 | **2** | **26** | +24 |
| single-session-assistant | 54 | 0 | **0** | **2** | +2 |
| single-session-preference | 12 | 5 | **0** | **13** | +13 |
| single-session-user | 56 | 4 | **0** | **10** | +10 |
| temporal-reasoning | 82 | 21 | **0** | **30** | +30 |
| **total** | 337 | 63 | **4** | **96** | **+92** |

## Response-shape distribution

| metric | base | new | Δ |
|---|---:|---:|---:|
| p50 length | 13 | 67 | +54 |
| p90 length | 4041 | 222 | -3819 |
| p99 length | 4625 | 346 | -4279 |
| max length | 4997 | 430 | -4567 |
| mean length | 739 | 90 | -649 |
| cliff hits (3500-5000ch) | 82 | 0 | -82 |

### Failure-mode classification

| class | base | new | Δ |
|---|---:|---:|---:|
| concrete | 308 | 467 | +159 |
| cot_preamble | 82 | 29 | -53 |
| empty | 50 | 2 | -48 |
| refusal | 48 | 0 | -48 |
| verbose_other | 12 | 2 | -10 |

## Sample flipped questions

### Lost wins (PF): base passed, new failed

| qtype | question | gold | base resp | new resp |
|---|---|---|---|---|
| knowledge-update | Where do I initially keep my old sneakers? | under my bed | Under your bed. | Under the bed. |
| knowledge-update | How many of Emma's recipes have I tried out? | 3 | 3. | You mentioned trying 2 of Emma's recipes on 2023/05/29 and 3 on 2023/05/30 — 3 ... |
| multi-session | How many pages do I have left to read in 'Sapiens'? | The information provided is not enough. You did not mention... | I don't know. | You mentioned reading "Sapiens" at a pace of 10–20 pages a week, but you did no... |
| multi-session | How many different types of food delivery services have I used recently? | 3 | 3: Domino's Pizza, Fresh Fusion, and Uber Eats. | You mentioned Domino's Pizza (May 22), Fresh Fusion (May 30), and Uber Eats (Ma... |


### Recovered (FP): base failed, new passed

| qtype | question | gold | base resp | new resp |
|---|---|---|---|---|
| temporal-reasoning | How many months ago did I book the Airbnb in San Francisco? | Five months ago | We need answer question: "How many months ago did I book the Airbnb in San Fran... | 5 months ago |
| multi-session | How many weeks did it take me to watch all the Marvel Cinematic Universe movies... | 3.5 weeks | About two weeks for the 22 MCU movies, and a week and a half for the main Star ... | You watched all 22 MCU movies in about 2 weeks (May 25), and all the main Star ... |
| knowledge-update | What time do I wake up on Saturday mornings? | 7:30 am | The user wants to know what time they wake up on Saturday mornings based on the... | 7:30 am |
| single-session-preference | I've got some free time tonight, any documentary recommendations? | The user would prefer documentary recommendations that are ... | The user is asking for documentary recommendations on 2023/05/30 (Tue) 22:56, b... | The user would prefer a nature documentary such as Dynasties or Chasing Coral, ... |
| single-session-user | How long was I in Korea for? | You did not mention this information. You mentioned staying... | I don't know. | You did not mention being in Korea. You mentioned planning a trip to Seoul in J... |
| single-session-assistant | I'm looking back at our previous conversation where you created two sad songs f... | C D E F G A B A G F E D C | _(empty)_ | C D E F G A B A G F E D C |


## engram_config delta

| key | base | new |
|---|---|---|
| chat | _unset_ | {"manifest_hash": "openai-chat/kimi-k2.6/base=https://opencode.ai/zen/go/v1/b4e... |
| embedder | _unset_ | {"dim": 1024, "manifest_hash": "local-embed/BAAI/bge-large-en-v1.5/dim=1024/nor... |
| enable_tools | _unset_ | true |
| judge_chat | {"manifest_hash": "openai-chat/anthropic/claude-sonnet-4-5/base=https://openrou... | {"manifest_hash": "openai-chat/openai/gpt-4o/base=https://openrouter.ai/api/v1/... |
| prompt_version | _unset_ | "v3a" |
