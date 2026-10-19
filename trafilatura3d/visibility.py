""""Visible if" expressions: whether a graph's input parameter, Input node or Output node
shows on its instances. UI only, never part of the cooking.

Syntax: inputs by identifier, `input.id` or `input["id"]`, components `.x .y
.z .w`, literals `true` / `false` and numbers, `&& || !`, `< > <= >= == !=` and parentheses. The result
must be a boolean; numbers are truthy when not 0. An unknown identifier or a malformed expression is
invalid: the item stays visible and the Properties show a warning."""

from __future__ import annotations

import ast
import re

_TOKEN = re.compile(r"""\s*(?:
    (?P<num>\d+\.\d*|\.\d+|\d+)|
    (?P<str>"[^"]*"|'[^']*')|
    (?P<op>&&|\|\||==|!=|<=|>=|[<>!()\[\].])|
    (?P<name>[A-Za-z_][A-Za-z0-9_]*))""", re.X)
_COMPONENTS = {"x": 0, "y": 1, "z": 2, "w": 3, "r": 0, "g": 1, "b": 2, "a": 3}


class VisibleIfError(ValueError):
    pass


def _translate(expr: str) -> str:
    """The expression as Python source: input.id -> _in["id"], && -> and, ! -> not ..."""
    out, pos = [], 0
    tokens = []
    while pos < len(expr):
        m = _TOKEN.match(expr, pos)
        if not m or m.end() == pos:
            if expr[pos:].strip() == "":
                break
            raise VisibleIfError(f"unexpected {expr[pos:pos + 8]!r}")
        tokens.append((m.lastgroup, m.group(m.lastgroup)))
        pos = m.end()
    i = 0
    while i < len(tokens):
        kind, text = tokens[i]
        if kind == "name" and text == "input":
            if i + 2 < len(tokens) and tokens[i + 1][1] == "." and tokens[i + 2][0] == "name":
                out.append(f'_in[{tokens[i + 2][1]!r}]')
                i += 3
            elif i + 3 < len(tokens) and tokens[i + 1][1] == "[" and tokens[i + 2][0] == "str" and tokens[i + 3][1] == "]":
                out.append(f'_in[{tokens[i + 2][1][1:-1]!r}]')
                i += 4
            else:
                raise VisibleIfError("expected input.identifier or input[\"identifier\"]")
            while i + 1 < len(tokens) and tokens[i][1] == "." and tokens[i + 1][1] in _COMPONENTS:
                out[-1] += f"[{_COMPONENTS[tokens[i + 1][1]]}]"
                i += 2
            continue
        if kind == "name":
            if text not in ("true", "false"):
                raise VisibleIfError(f"unknown name {text!r}")
            out.append("True" if text == "true" else "False")
        elif kind == "op":
            out.append({"&&": " and ", "||": " or ", "!": " not "}.get(text, f" {text} "))
        elif kind == "num":
            out.append(text)
        else:
            raise VisibleIfError(f"unexpected {text!r}")
        i += 1
    if not any(t[1] == "input" for t in tokens):
        raise VisibleIfError("the expression must refer to an input (input.identifier)")
    return "".join(out)


_ALLOWED = (ast.Expression, ast.BoolOp, ast.And, ast.Or, ast.UnaryOp, ast.Not, ast.Compare, ast.Eq, ast.NotEq,
            ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Subscript, ast.Name, ast.Load, ast.Constant)


def compile_visible_if(expr: str):
    """A function values -> bool for a non empty expression, or raise VisibleIfError."""
    tree = ast.parse(_translate(expr).strip(), mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED) or (isinstance(node, ast.Name) and node.id != "_in"):
            raise VisibleIfError("not a Visible if expression")
    code = compile(tree, "<visible if>", "eval")

    def evaluate(values: dict) -> bool:
        try:
            return bool(eval(code, {"__builtins__": {}}, {"_in": _Inputs(values)}))  # noqa: S307 - checked tree
        except (KeyError, IndexError, TypeError) as e:
            raise VisibleIfError(str(e)) from None
    return evaluate


class _Inputs(dict):
    def __missing__(self, key):
        raise KeyError(f"unknown input {key!r}")


def visible(expr: str, values: dict) -> bool:
    """Evaluate a Visible if expression; empty or invalid means visible."""
    if not expr or not expr.strip():
        return True
    try:
        return compile_visible_if(expr)(values)
    except (VisibleIfError, SyntaxError):
        return True


def check(expr: str, values: dict) -> str | None:
    """A warning for the Visible if field (`values`: the graph's inputs with their defaults), or None
    when the expression is fine or empty."""
    if not expr or not expr.strip():
        return None
    try:
        compile_visible_if(expr)(values)
    except (VisibleIfError, SyntaxError) as e:
        return f"Visible if: {e}"
    return None

