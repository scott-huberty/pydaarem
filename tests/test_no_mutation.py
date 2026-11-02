"""Check that daarem.py never writes into an existing array.

The array API standard advises against in-place writes, because a library
can return a view or a copy, and some libraries (for example JAX) have
immutable arrays. See "Copy-view behavior and mutability" in the standard.
"""

import ast
import inspect

import pydaarem

SOURCE = inspect.getsource(inspect.getmodule(pydaarem.daarem))


def _subscript_targets(node):
    """Return the item and slice assignment targets of an assignment node."""
    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
    found = []
    for target in targets:
        found.extend(n for n in ast.walk(target) if isinstance(n, ast.Subscript))
    return found


def test_no_item_or_slice_assignment():
    tree = ast.parse(SOURCE)
    writes = [
        f"line {target.lineno}: {ast.unparse(target)}"
        for node in ast.walk(tree)
        if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign))
        for target in _subscript_targets(node)
    ]
    assert writes == []
