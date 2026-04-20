# File Organization

- One class per file in `workflow/`.
- Private modules prefixed with `_` (e.g., `_base_node.py`).
- Public API exported through `__init__.py`.

## File Headers

Every source file must have:
1. Apache 2.0 license header.
2. `from __future__ import annotations`.
3. Standard library imports, then third-party, then relative.
