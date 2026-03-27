# ADK Workflow use_as_output Sample

## Overview

This sample demonstrates how to use `ctx.run_node(node, use_as_output=True)` to delegate a node's output to a dynamically executed child node.

When `use_as_output=True` is set, the child node's output replaces the parent's output. The parent's own output event is suppressed to avoid duplication, and the child's output flows downstream through the graph as if the parent produced it.

## Sample Inputs

- Any text input (e.g. `hello world`)

## Graph

```text
      [ START ]
          |
          v
    [orchestrate]
          |  (delegates output to [transform] via ctx.run_node)
          v
      [finalize]
```

## How To

1. **Mark the orchestrator as rerun_on_resume**: The parent node that calls `ctx.run_node` must use `@node(rerun_on_resume=True)`.

   ```python
   from google.adk.workflow import node

   @node(rerun_on_resume=True)
   async def orchestrate(ctx: Context, node_input: str) -> str:
       return await ctx.run_node(
           transform, node_input=node_input, use_as_output=True
       )
   ```

2. **Define the child node**: The child is a plain function. Its output becomes the parent's output and flows to downstream nodes.

   ```python
   def transform(node_input: str) -> str:
       return node_input.upper()
   ```

3. **Downstream receives delegated output**: The `finalize` node receives the child's output (`"HELLO WORLD"`) as its `node_input`, not the parent's.

   ```python
   def finalize(node_input: str) -> str:
       return f'final: {node_input}'
   ```
