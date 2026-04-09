# Type Hints and Strong Typing

- **Prefer Strong Typing**: Use type hints for all function arguments and return types. Avoid leaving types unspecified.
- **Minimize `Any`**: Use specific types or `Generic` whenever possible. Avoid `Any` as it bypasses type checking.
- **No double-quoted type hints**: When `from __future__ import annotations` is present, use bare type names (e.g., `list[str]` instead of `"list[str]"`).
