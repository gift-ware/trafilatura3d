from pathlib import Path

from trafilatura3d.nodes.base import Node, Output, ParamKind, PortType


class TestPattern(Node):
    """Standard source for tests, node sheets and library thumbnails (hidden from the library): soft
    discs, a hard square, a gradient band and fine noise, so filters meet edges, slopes and detail."""

    type = "test_pattern"
    label = "Test Pattern"
    category = ""
    outputs = {"output": Output(PortType.GRAY)}
    internal_uniforms = {"color": ParamKind.INT}

    def encode(self, ctx, job):
        ctx.dispatch(self, job, {**job.values, "color": int(self.outputs["output"].type is PortType.COLOR)})

    def wgsl(self):
        return (Path(__file__).parent / "test_pattern.wgsl").read_text(encoding="utf-8")


class TestPatternColor(TestPattern):
    type = "test_pattern_color"
    label = "Test Pattern Color"
    outputs = {"output": Output(PortType.COLOR)}
