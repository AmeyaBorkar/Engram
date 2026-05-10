# Security Policy

## Reporting a vulnerability

Please email **ameyaborkar17@gmail.com** with subject prefix `[engram-security]`. Do not open public issues for security-sensitive bugs.

We aim to acknowledge reports within 72 hours and to coordinate a fix and disclosure with the reporter.

## What's in scope

- The Engram library: storage, retrieval, consolidation, decay, provider adapters bundled in this repository.
- Documentation that could mislead implementers into insecure usage.

What's **out** of scope:

- Vulnerabilities in upstream provider APIs (report to the provider).
- Application-level misuse (e.g. observing raw API keys without redaction — the host application owns secret hygiene).

## Threats Engram explicitly cares about

These are the failure modes we test against, and reports here are always in scope:

- **Prompt injection during consolidation.** Event content is user-controlled and is fed to an LLM. The consolidation prompt is hardened to treat content as data, not instructions; bypasses are tracked in a corpus and regression-tested.
- **Provenance integrity.** Every memory item links to the events that support it. Reports of dropped or forged provenance links are critical.
- **SQL injection.** Engram only constructs queries with parameterized statements. Any code path that interpolates untrusted input into SQL is a bug.
- **Stored-secret leakage.** Engram redacts configured PII patterns from request/response logs and telemetry. A path that emits redactable content unredacted is in scope.
- **Cross-tenant leakage** *(once Stage 9 lands)*. Multi-tenant deployments must isolate at the storage layer; any read or write that crosses tenant boundaries is critical.

## Supported versions

While the library is pre-1.0, only the latest minor release receives security fixes. Once v1.0 ships, we'll publish a longer support window here.

## Hardening commitments

- Dependencies are scanned (`pip-audit`) on every CI run.
- Releases will be Sigstore-signed once we publish to PyPI.
- The threat model is reviewed each minor release; changes are noted in `CHANGELOG.md`.
