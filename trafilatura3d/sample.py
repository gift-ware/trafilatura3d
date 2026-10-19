"""A one-node graph with the standard test pattern on every input: what tests, node sheets and
library thumbnails cook."""

from __future__ import annotations

from trafilatura3d.compiler import compile_graph
from trafilatura3d.evaluator import Job
from trafilatura3d.nodes import NODES
from trafilatura3d.nodes.base import PortType
from trafilatura3d.package import Package


def sample_job(node_type: str, color: bool = False, size_log2: int = 6, params: dict | None = None) -> Job:
    """The job of `node_type` fed by test patterns (color ones on `any` inputs when `color`)."""
    pkg = Package()
    g = pkg.new_graph("sample", size_log2)
    node = g.add_node(node_type)
    for name, value in (params or {}).items():
        g.set_param(node, name, value)
    d = g.definition(node)
    shown = set(d.visible_inputs(d.values(g.nodes[node])))  # numbered inputs beyond the count don't exist
    for port, spec in NODES[node_type].inputs.items():
        if port not in shown:
            continue
        use_color = spec.type is PortType.COLOR or (spec.type is PortType.ANY and color)
        src = g.add_node("test_pattern_color" if use_color else "test_pattern")
        g.connect(src, "output", node, port)
    return compile_graph(pkg, "sample").jobs[node]


def has_any_input(node_type: str) -> bool:
    return any(s.type is PortType.ANY for s in NODES[node_type].inputs.values())
