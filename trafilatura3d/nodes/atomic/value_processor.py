"""Value processor: a function graph evaluated once (no $pos), compiled to WGSL; its result is a value
(a 1x1 texture) for value connectors. It can read the graph's parameters and sample its inputs."""

from __future__ import annotations

from trafilatura3d.functions import BUILTINS, FunctionGraph, WGSLCodegen
from trafilatura3d.nodes.atomic.pixel_processor import MAX_INPUTS, SYSTEM_NAMES, PixelProcessor, _shader, graph_uniforms
from trafilatura3d.nodes.base import Output, Param, ParamKind, PortType

DEFAULT_FUNCTION = {"nodes": [{"id": "value", "type": "const_float4", "params": {"value": [0.0, 0.0, 0.0, 1.0]},
                               "position": [0, 0]}], "output": "value"}


class ValueProcessor(PixelProcessor):
    type = "value_processor"
    label = "Value processor"
    catalog = "value_processor"
    color_mode = None
    outputs = {"output": Output(PortType.VALUE)}
    params = {
        "value_processor_function": Param(ParamKind.FUNCTION, DEFAULT_FUNCTION, "Value processor function"),
        "input_count": Param(ParamKind.INT, 0, "Input count", 0, MAX_INPUTS),
    }
    inputs = {}

    def encode(self, ctx, job):
        variables, names, uniforms, values = graph_uniforms(job, BUILTINS, SYSTEM_NAMES)
        fg = FunctionGraph(job.values["value_processor_function"], variables)
        ports = list(job.node.inputs)
        stmts, result = WGSLCodegen(fg, names, ports).emit()
        out_t = fg.types[fg.output]
        to_vec4 = {"float": f"vec4f({result}, 0.0, 0.0, 0.0)", "float2": f"vec4f({result}, 0.0, 0.0)",
                   "float3": f"vec4f({result}, 0.0)", "float4": result, "int": f"vec4f(f32({result}), 0.0, 0.0, 0.0)",
                   "int2": f"vec4f(vec2f({result}), 0.0, 0.0)", "int3": f"vec4f(vec3f({result}), 0.0)",
                   "int4": f"vec4f({result})", "bool": f"vec4f(select(0.0, 1.0, {result}), 0.0, 0.0, 0.0)"
                   }.get(out_t, "vec4f(0.0)")
        touch = "".join(f"    if (d.x < 0) {{ o = o + texel_{port}(vec2i(0)); }}\n" for port in ports)
        code = (f"@compute @workgroup_size(WG, WG)\nfn main(@builtin(global_invocation_id) id: vec3u) {{\n"
                f"    let d = size();\n    let c = vec2i(id.xy);\n    if (c.x != 0 || c.y != 0) {{ return; }}\n"
                f"{stmts}\n    var o = {to_vec4};\n{touch}    store(c, o);\n}}\n")
        ctx.dispatch(_shader(code, job.node.inputs, uniforms, job.node.outputs), job, values)
