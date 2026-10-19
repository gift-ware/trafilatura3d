from trafilatura3d.nodes.base import Input, Output, Param, ParamKind, PortType

from ._lib import BasicNode, helper, source
from .blur_hq import blur_hq


class HeightMapFrequenciesMapper(BasicNode):
    """Split a heightmap into large-scale differences and small-scale ones: Displacement is the
    Blur HQ of Intensity Relief pulled towards mid gray by Relief / 32 (at 32 all the relief goes to the other
    output), Relief (parallax) is input - Displacement + 0.5."""

    type = "height_map_frequencies_mapper"
    label = "Height Map Frequencies Mapper"
    category = "Filters/Adjustments"
    catalog = "height_map_frequencies_mapper"
    inputs = {"input": Input(PortType.GRAY, "Input")}
    outputs = {"displacement": Output(PortType.GRAY, "Displacement"), "relief": Output(PortType.GRAY, "Relief")}
    params = {"relief": Param(ParamKind.FLOAT, 16.0, "Relief", 0.0, 32.0)}

    def encode(self, ctx, job):
        src = source(ctx, job, "input")
        blurred = blur_hq(ctx, job, src, True, job.values["relief"], 0, ctx.scratch(job, True))
        ctx.dispatch(FINAL, job, job.values, srcs={"blurred": (blurred, True)})


FINAL = helper("height_map_frequencies_mapper_final", "height_map_frequencies_mapper.wgsl",
               {"input": Input(PortType.GRAY), "blurred": Input(PortType.GRAY)},
               {"displacement": Output(PortType.GRAY), "relief": Output(PortType.GRAY)},
               params=HeightMapFrequenciesMapper.params)
