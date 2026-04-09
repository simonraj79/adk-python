# Imports Style Guide

## General Rules

- **Source code** (`src/`): Use relative imports.
  `from ..agents.llm_agent import LlmAgent`
- **Tests** (`tests/`): Use absolute imports.
  `from google.adk.agents.llm_agent import LlmAgent`
- **Import from module**: Import from the module file, not from `__init__.py`.
  `from ..agents.llm_agent import LlmAgent` (not `from ..agents import LlmAgent`)

## TYPE_CHECKING Imports

Use `TYPE_CHECKING` for imports needed only by type hints to avoid circular imports at runtime:

```python
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..agents.invocation_context import InvocationContext
```

This works because `from __future__ import annotations` makes all annotations strings (deferred evaluation), so the import is never needed at runtime.
