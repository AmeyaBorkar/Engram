"""Versioned LLM prompts for the consolidation engine.

Each prompt is a `<name>_v<n>.txt` file in this package. Version bumps
land alongside a CHANGELOG entry; downstream code that needs an exact
version pins it via the public `PROMPT_VERSIONS` registry in
`engram.consolidation._abstraction`.
"""
