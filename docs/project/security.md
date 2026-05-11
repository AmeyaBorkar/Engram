# Security

Threat model + how to report.

## Summary

Engram is a memory layer for LLM systems. The threats it explicitly cares about:

| Threat | Mitigation |
|---|---|
| **Prompt injection via stored events** | Hardened prompts (`abstract_v1.txt`, `judge_v1.txt`, `merge_v1.txt`) treat event content as DATA. Parse-time filters reject injection-like outputs. Stage 5 regression corpus runs on every PR. |
| **SQL injection** | Parameterized queries everywhere. `ruff S608` enforced in CI. |
| **PII leakage in logs** | Configurable `Redactor` scrubs request/response payloads before structured logging. Default patterns cover OpenAI / Anthropic / AWS keys, emails, phones, SSNs, credit-card-shaped digits. |
| **Cross-tenant data leak** | Tenant-scoped writes today; full read-side enforcement via Postgres RLS in v0.4.0. SQLite is single-process and not a security boundary for multi-tenant deployments. |
| **Replay-based attacks on temporal queries** | Invalidation timestamps are write-once (first wins via COALESCE). `as_of` queries return historically-correct state but can't be tricked into re-validating an invalidated item. |
| **Provider key exfiltration** | The `Redactor` ships with patterns for the major providers. Adapters use them on logs by default. |

## Out of scope

- Network-layer security (TLS, firewall rules) — the deployer's responsibility.
- LLM model weights / supply chain — out of scope for the library.
- DoS via expensive queries — the library has perf budgets; the deployer should impose rate limits at the API gateway.

## Reporting

If you find a security issue, please email `ameyaborkar17@gmail.com` rather than filing a public issue. We aim to acknowledge within 48 hours and ship a patch within two weeks for High / Critical severity.

See [`SECURITY.md`](https://github.com/AmeyaBorkar/Engram/blob/main/SECURITY.md) at the root of the repo for the canonical version of this policy.

## Auditability

Every release has:

- A reproducibility manifest in `benchmarks/runs/` for headline benchmarks.
- `pip-audit` passing in CI (no known CVEs in transitive deps).
- A pinned set of dependency versions.

Future (v0.4.0+):

- Sigstore-signed releases.
- CI pinned by digest where it touches secrets.
- Refreshed threat model per minor release.
