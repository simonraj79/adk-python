# Pydantic Patterns

ADK models use Pydantic v2. Key patterns:

```python
from pydantic import BaseModel, Field, PrivateAttr

class MyModel(BaseModel):
    # Public fields with defaults
    name: str
    timeout: int = Field(default=30, ge=1)

    # Private attributes (not serialized, not in schema)
    _cache: dict = PrivateAttr(default_factory=dict)

    # Post-init logic
    def model_post_init(self, __context):
        self._cache = {}
```

- Use `Field()` for validation, defaults, and descriptions.
- Use `PrivateAttr()` for internal state that shouldn't be serialized.
- Use `model_post_init()` instead of `__init__` for setup logic.
- Prefer `model_dump()` over `dict()` (Pydantic v2).
