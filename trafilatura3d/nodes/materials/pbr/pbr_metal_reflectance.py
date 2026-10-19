from trafilatura3d.nodes.base import Output
from trafilatura3d.nodes.materials._base import COLOR, E, MatNode
from trafilatura3d.nodes.materials.pbr._pbr import CATEGORY, METALS


class PbrMetalReflectance(MatNode):
    """Uniform reflection color (F0, sRGB-encoded) of a pure metal, for the Base Color of the
    metallic workflow or the Specular of the specular/glossiness one."""

    type = "pbr_metal_reflectance"
    label = "PBR Metal Reflectance"
    category = CATEGORY
    catalog = "pbr_metal_reflectance"
    includes = ("_pbr.wgsl",)
    outputs = {"output": Output(COLOR)}
    params = {"metal_color": E("Gold", "Metal Color", METALS)}
