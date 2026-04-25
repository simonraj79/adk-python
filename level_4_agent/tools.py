"""Custom function tools for Level 4.

These are *function tools* (plain Python functions auto-wrapped by ADK
when assigned to `tools=[...]`). They live here rather than inline in
`agent.py` so the safety allowlist (`safety.py`) can import them
without dragging the whole agent module.

ADK 2.0 has no built-in calculator, so we ship one here. The other
tools in this file (`load_web_page`) are imported directly from
`google.adk.tools.*` — included by reference in the allowlist, not
re-implemented.
"""

from __future__ import annotations

import ast
import math
import operator
from typing import Any


# Whitelisted binary and unary operators. Anything not in this map
# raises a ValueError during evaluation.
_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPS = {
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

# Whitelisted callables. Includes a handful of math functions that
# cover most BI / data analysis arithmetic without giving the LLM a
# code-execution surface.
_FUNCS: dict[str, Any] = {
    "abs": abs,
    "min": min,
    "max": max,
    "round": round,
    "sum": sum,
    "len": len,
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "log2": math.log2,
    "exp": math.exp,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "floor": math.floor,
    "ceil": math.ceil,
}

# Whitelisted name lookups (constants).
_NAMES: dict[str, Any] = {
    "pi": math.pi,
    "e": math.e,
    "inf": math.inf,
}


def _eval_node(node: ast.AST) -> Any:
  """Recursively evaluate a whitelisted AST node tree.

  Anything not explicitly handled raises ValueError — this is the
  defensive default that keeps the calculator from becoming a code-
  execution surface (no attribute access, no imports, no comprehensions,
  no arbitrary function calls).
  """
  if isinstance(node, ast.Constant):
    if isinstance(node.value, (int, float)):
      return node.value
    raise ValueError(
        f"Only numeric constants are allowed (got {type(node.value).__name__})."
    )
  if isinstance(node, ast.Name):
    if node.id in _NAMES:
      return _NAMES[node.id]
    raise ValueError(f"Unknown name: {node.id!r}")
  if isinstance(node, ast.BinOp):
    op = _BIN_OPS.get(type(node.op))
    if op is None:
      raise ValueError(
          f"Unsupported binary operator: {type(node.op).__name__}"
      )
    return op(_eval_node(node.left), _eval_node(node.right))
  if isinstance(node, ast.UnaryOp):
    op = _UNARY_OPS.get(type(node.op))
    if op is None:
      raise ValueError(
          f"Unsupported unary operator: {type(node.op).__name__}"
      )
    return op(_eval_node(node.operand))
  if isinstance(node, ast.Call):
    if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCS:
      func_name = (
          node.func.id if isinstance(node.func, ast.Name) else "<expr>"
      )
      raise ValueError(
          f"Unsupported function: {func_name!r}. Allowed:"
          f" {sorted(_FUNCS)}."
      )
    args = [_eval_node(a) for a in node.args]
    return _FUNCS[node.func.id](*args)
  if isinstance(node, ast.List):
    return [_eval_node(e) for e in node.elts]
  if isinstance(node, ast.Tuple):
    return tuple(_eval_node(e) for e in node.elts)
  raise ValueError(f"Unsupported syntax: {type(node).__name__}")


def calculator(expression: str) -> str:
  """Safely evaluate a math expression and return the result as a string.

  Use this for arithmetic that doesn't justify spinning up the full
  Python code-execution sandbox. Compared to free-form Python, this
  tool:
    - cannot import modules
    - cannot read or write files
    - cannot access attributes (no `obj.attr` syntax)
    - cannot define functions or comprehensions
    - only the operators and functions documented below

  Supported operations:
    - Arithmetic: + - * / // % **
    - Functions: abs, min, max, round, sum, len, sqrt, log, log10,
      log2, exp, sin, cos, tan, floor, ceil
    - Constants: pi, e, inf
    - Lists / tuples for multi-arg functions, e.g. min([1, 2, 3])

  Args:
    expression: A math expression string.
      Examples: "2 + 2", "sqrt(144)", "round(391.04 - 383.29, 2)",
      "min([5, 3, 9])", "(391.04 - 383.29) / 383.29 * 100".

  Returns:
    The numeric result formatted as a string, or an "Error: ..."
    message if the expression is invalid or uses disallowed syntax.
  """
  try:
    tree = ast.parse(expression, mode="eval")
    result = _eval_node(tree.body)
  except SyntaxError as exc:
    return f"Error: invalid expression syntax — {exc.msg}"
  except ValueError as exc:
    return f"Error: {exc}"
  except Exception as exc:  # pylint: disable=broad-except
    return f"Error: {type(exc).__name__}: {exc}"
  return str(result)
