"""Graph boundary nodes: Input color / Input grayscale (the graph's inputs; their Default value when
nothing comes in) and Output (the graph's results; at the top level, the exported files)."""

from trafilatura3d.nodes.base import Input, Node, Output, Param, ParamKind, PortType

# The usages: the legacy PBR ones, ASM's (its specification) and OpenPBR's (the 3D view's shaders read them,
# shaders.py).
USAGES = ("", "baseColor", "normal", "roughness", "metallic", "height", "ambientOcclusion", "opacity",
          "emissive", "specular", "glossiness", "diffuse", "specularLevel", "anisotropyLevel",
          "anisotropyAngle", "transmissive", "scattering", "mask", "any",
          "specularEdgeColor", "IOR", "translucency", "absorptionColor", "absorptionDistance", "scatteringColor",
          "scatteringDistance", "scatteringDistanceScale", "sheenOpacity", "sheenColor", "sheenRoughness",
          "coatOpacity", "coatColor", "coatRoughness", "coatNormal",
          "metalness", "specularWeight", "specularColor", "specularRoughness", "specularRoughnessAnisotropy",
          "specularIOR", "transmissionWeight", "transmissionColor", "transmissionDepth", "subsurfaceWeight",
          "subsurfaceColor", "subsurfaceRadius", "subsurfaceRadiusScale", "coatWeight", "fuzzWeight", "fuzzColor",
          "fuzzRoughness", "emissionColor")


def _attributes(stem: str) -> dict:
    """The attributes of graph inputs and outputs."""
    return {"identifier": Param(ParamKind.STRING, stem, "Identifier"),
            "description": Param(ParamKind.STRING, "", "Description"),
            "label": Param(ParamKind.STRING, "", "Label"),
            "user_data": Param(ParamKind.STRING, "", "User data"),
            "group": Param(ParamKind.STRING, "", "Group"),
            "component": Param(ParamKind.STRING, "", "Component"),
            "usage": Param(ParamKind.ENUM, "", "Usage", choices=USAGES),
            "color_space": Param(ParamKind.STRING, "", "Color space"),
            "visible_if": Param(ParamKind.STRING, "", "Visible if")}


class InputColor(Node):
    type = "input_color"
    label = "Input color"
    category = "Atomic nodes"
    catalog = "input_color"
    outputs = {"output": Output(PortType.COLOR)}
    params = {"default_value": Param(ParamKind.COLOR, [0.0, 0.0, 0.0, 1.0], "Default value"),
              "resource": Param(ParamKind.STRING, "", "PKG resource path"), **_attributes("input")}
    unsupported = {}

    def wgsl(self):
        return UNIFORM_WGSL


class InputGrayscale(Node):
    type = "input_grayscale"
    label = "Input grayscale"
    category = "Atomic nodes"
    catalog = "input_grayscale"
    outputs = {"output": Output(PortType.GRAY)}
    params = {"default_value": Param(ParamKind.FLOAT, 0.0, "Default value", 0.0, 1.0),
              "resource": Param(ParamKind.STRING, "", "PKG resource path"), **_attributes("input")}

    def wgsl(self):
        return UNIFORM_GRAY_WGSL


class InputValue(Node):
    """A value input of the graph: on the graph's instances it is a value connector (fed by a Value
    processor, say); inside, functions read it with Get <identifier> (a float4). Unconnected: its
    Default value."""
    type = "input_value"
    label = "Input value"
    category = "Atomic nodes"
    catalog = "input_value"
    outputs = {"output": Output(PortType.VALUE)}
    params = {"default_value": Param(ParamKind.FLOAT4, [0.0, 0.0, 0.0, 0.0], "Default value"),
              **{k: v for k, v in _attributes("input").items() if k in ("identifier", "description", "label",
                                                                      "user_data", "group", "visible_if")}}

    def wgsl(self):
        return UNIFORM_WGSL


class OutputNode(Node):
    type = "output"
    label = "Output"
    category = "Atomic nodes"
    catalog = "output"
    inputs = {"input": Input(label="Input")}
    params = _attributes("output")  # the file format is the Export outputs dialog's, for all outputs

    def wgsl(self):
        return COPY_WGSL


UNIFORM_WGSL = """
@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    store(c, p.default_value);
}
"""

UNIFORM_GRAY_WGSL = """
@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let g = p.default_value;
    store(c, vec4f(g, g, g, 1.0));
}
"""

COPY_WGSL = """
@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    store(c, load_input(c));
}
"""
