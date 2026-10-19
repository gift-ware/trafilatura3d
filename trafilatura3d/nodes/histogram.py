from pathlib import Path

from trafilatura3d.nodes.base import Input, Node, Output, PortType

BINS = 256
GROUPS = 256  # rows of partial counts (workgroups), summed on the CPU


class Histogram(Node):
    """Helper: the histogram of a result (BINS bins of 0-1 per channel, values outside counted in the end bins) and
    its exact per channel minimum and maximum, over every texel, for the Levels editor. Workgroup g counts the rows
    y = g (mod GROUPS) in workgroup memory and writes row g of a (BINS + 2) x GROUPS rgba32float texture: texel x <
    BINS = the (R, G, B, A) counts of bin x, then the minimum and the maximum (`Evaluator.histogram` sums the rows)."""

    type = "histogram"
    registered = False
    inputs = {"image": Input()}
    outputs = {"output": Output(PortType.COLOR)}

    def wgsl(self):
        return f"const BINS = {BINS}u;\n" + (Path(__file__).parent / "histogram.wgsl").read_text(encoding="utf-8")


HISTOGRAM = Histogram()
