# Deprecation policy

Pre-`v1.0.0`:

- **Within a minor version** (e.g. `v0.3.0` → `v0.3.x`): no breaking API changes. New names are additive only.
- **Across a minor version bump** (e.g. `v0.3.x` → `v0.4.0`): breaking changes are allowed, but must follow the deprecation cycle below.
- **Across a major version bump** pre-1.0 (e.g. `v0.x` → `v0.(x+1)`): breaking changes allowed; documented in the CHANGELOG.

Post-`v1.0.0`:

- We follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
- Breaking changes only in major version bumps.

## Deprecation cycle

A symbol that's deprecated for removal goes through this cycle:

1. **Marked as deprecated** in version `vN.M.P`. The symbol still works exactly as before, but:
   - A `DeprecationWarning` fires on use.
   - The CHANGELOG entry calls it out.
   - The API stability doc moves it to the "experimental / may change" section.
   - A migration note in the docs explains what to use instead.
2. **One minor version later** (`vN.(M+1).0`), the symbol may be removed.
3. **Before removal**, the migration must be clear and the replacement must be in place.

## Example

A hypothetical deprecation of `Memory.observe(content)` in favor of `Memory.observe_event(event)`:

```python
# v0.4.0: deprecated
import warnings

def observe(self, content: str | Event) -> Event:
    warnings.warn(
        "Memory.observe is deprecated; use Memory.observe_event(event). "
        "Will be removed in v0.5.0.",
        DeprecationWarning,
        stacklevel=2,
    )
    return self.observe_event(...)
```

```python
# v0.5.0: removed
# (no `observe` attribute on Memory)
```

## What is NOT a breaking change

- Adding a new optional keyword argument with a sensible default.
- Adding new methods or attributes.
- Adding new enum values (e.g. a new `Resolution` policy).
- Loosening a constraint (e.g. allowing `None` where only `str` was accepted).
- Improving error messages.

## What IS a breaking change

- Removing or renaming a public symbol.
- Changing a method signature in a way existing callers would notice.
- Removing an enum value.
- Tightening a constraint (e.g. now rejecting input that was previously accepted).
- Changing default behavior in a way that produces different output for the same input.

The cross-cutting [SECURITY.md](https://github.com/AmeyaBorkar/Engram/blob/main/SECURITY.md) covers any security-driven exceptions.
