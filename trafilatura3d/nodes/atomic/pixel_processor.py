"""Pixel processor: a function graph evaluated for every output pixel, compiled to WGSL. It reads
$pos (the pixel's normalized position), $size and the other system variables, the graph's
parameters (Get by identifier) and its inputs through Sample Gray / Sample Color."""

from __future__ import annotations

import hashlib

from trafilatura3d.functions import PARAM_TYPES, PIXEL_BUILTINS, FunctionGraph, WGSLCodegen, _ident
from trafilatura3d.nodes.base import Input, Node, Output, Param, ParamKind

DEFAULT_FUNCTION = {"nodes": [
    {"id": "pos", "type": "get_float2", "params": {"variable": "$pos"}, "position": [0, 0]},
    {"id": "sample", "type": "sample_color", "inputs": {"position": "pos"}, "position": [200, 0]}],
    "output": "sample"}

SYSTEM_NAMES = {"$pos": "pos", "$size": "p.size_", "$sizelog2": "log2(p.size_)", "$tiling": "p.tiling_",
                "$pixelratio": "1", "$time": "0.0", "$normalformat": "1", "$physicalsize": "vec3f(0.0)"}
MAX_INPUTS = 8


class PixelProcessor(Node):
    type = "pixel_processor"
    label = "Pixel processor"
    category = "Atomic nodes"
    catalog = "pixel_processor"
    color_mode = "color_mode"
    outputs = {"output": Output()}
    params = {
        "color_mode": Param(ParamKind.ENUM, "Color", "Color mode", choices=("Grayscale", "Color")),
        "per_pixel_function": Param(ParamKind.FUNCTION, DEFAULT_FUNCTION, "Per pixel function"),
        "input_count": Param(ParamKind.INT, 1, "Input count", 0, MAX_INPUTS),  # inputs by count, not added on demand
    }
    inputs = {"input_image_0": Input(label="Input image 0")}

    def instantiate(self, values):
        n = max(0, min(MAX_INPUTS, int(values.get("input_count", 1)) if isinstance(values.get("input_count"), int) else 1))
        if n == len(self.inputs):
            return self
        clone = object.__new__(type(self))
        clone.__dict__.update(self.__dict__)
        clone.inputs = {f"input_image_{i}": Input(label=f"Input image {i}") for i in range(n)}
        return clone

    def encode(self, ctx, job):
        variables, names, uniforms, values = graph_uniforms(job, PIXEL_BUILTINS, SYSTEM_NAMES)
        fg = FunctionGraph(job.values["per_pixel_function"], variables)
        ports = list(job.node.inputs)
        stmts, result = WGSLCodegen(fg, names, ports).emit()
        out_t = fg.types[fg.output]
        to_vec4 = {"float": f"vec4f({result}, {result}, {result}, 1.0)", "float2": f"vec4f({result}, 0.0, 1.0)",
                   "float3": f"vec4f({result}, 1.0)", "float4": result, "int": f"vec4f(vec3f(f32({result})), 1.0)",
                   "bool": f"vec4f(vec3f(select(0.0, 1.0, {result})), 1.0)"}.get(out_t, "vec4f(0.0)")
        # Inputs no sampler reads would be dropped by layout="auto": a branch never taken uses them.
        touch = "".join(f"    if (d.x < 0) {{ o = o + texel_{port}(vec2i(0)); }}\n" for port in ports)
        code = (f"@compute @workgroup_size(WG, WG)\nfn main(@builtin(global_invocation_id) id: vec3u) {{\n"
                f"    let d = size();\n    let c = vec2i(id.xy);\n    if (c.x >= d.x || c.y >= d.y) {{ return; }}\n"
                f"    let pos = (vec2f(c) + 0.5) / vec2f(d);\n{stmts}\n    var o = {to_vec4};\n{touch}"
                f"    if (OUT_OUTPUT_GRAY) {{ o = vec4f(o.r, o.r, o.r, 1.0); }}\n    store(c, o);\n}}\n")
        ctx.dispatch(_shader(code, job.node.inputs, uniforms, job.node.outputs), job, values)


def graph_uniforms(job, builtins: dict, system_names: dict):
    """Variables a processor's function may Get (system variables and the graph's parameters), the
    WGSL expression of each, and the graph parameters as uniform fields with their values."""
    variables, names = dict(builtins), {k: v for k, v in system_names.items() if k in builtins}
    uniforms, values = {}, {}
    for pid, (kind_name, value) in job.extra.get("graph_params", {}).items():
        kind = ParamKind(kind_name)
        if kind not in PARAM_TYPES or kind is ParamKind.STRING:
            continue
        variables[pid] = PARAM_TYPES[kind]
        field = f"gp_{_ident(pid)}"
        names[pid] = f"p.{field}"
        uniforms[field] = ParamKind.INT if kind is ParamKind.ENUM else kind  # enums arrive as their index
        values[field] = value
    return variables, names, uniforms, values


_SHADERS: dict[str, Node] = {}


def _shader(code: str, inputs: dict, uniforms: dict, outputs: dict) -> Node:
    key = hashlib.sha1((code + repr(sorted(inputs)) + repr(sorted(uniforms.items()))
                        + repr([(k, o.type) for k, o in outputs.items()])).encode()).hexdigest()
    if key not in _SHADERS:
        _SHADERS[key] = _ProcessorShader(code, inputs, uniforms, outputs, key)
    return _SHADERS[key]


class _ProcessorShader(Node):
    """A compiled function: a helper node with the job's inputs and outputs and the graph parameters
    it reads as uniforms."""
    type = "processor_shader"
    registered = False

    def __init__(self, code, inputs, uniforms, outputs, key):
        self.code, self.inputs, self.internal_uniforms, self.pipeline_key = code, dict(inputs), dict(uniforms), key
        self.outputs, self.params = dict(outputs), {}

    def wgsl(self):
        return self.code
