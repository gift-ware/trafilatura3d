"""Light scenes the editor tests open, built at test time (no example files): a small material package of our own
nodes, and a Paint project on the shader ball shipped with the editor."""

from pathlib import Path

from trafilatura3d.package import Package

SHADERBALL = (Path(__file__).resolve().parent.parent / "trafilatura3d" / "editor" / "paint" / "shaderball" /
              "Shader-Ball.fbx")
MATERIAL_GRAPH = "test_material"


def material_package(path: Path, size_log2: int = 8) -> Path:
    """A package with one material graph (MATERIAL_GRAPH): a noise as base color, its normal, a uniform roughness;
    outputs with their usages. Saved at `path`; returns it."""
    pkg = Package()
    g = pkg.new_graph(MATERIAL_GRAPH, size_log2)
    noise = g.add_node("perlin_noise")
    normal = g.add_node("normal")
    g.connect(noise, "output", normal, "input")
    rough = g.add_node("uniform_color")
    g.set_param(rough, "color_mode", "Grayscale")
    g.set_param(rough, "output_color", [0.6, 0.6, 0.6, 1.0])
    for src, ident, usage in ((noise, "basecolor", "baseColor"), (normal, "normal", "normal"),
                              (rough, "roughness", "roughness")):
        out = g.add_node("output")
        g.set_param(out, "identifier", ident)
        g.set_param(out, "usage", usage)
        g.connect(src, "output", out, "input")
    pkg.save(path)
    return path


def shaderball_session(evaluator, size: int = 128):
    """A Paint session (one Texture Set) on the shader ball at `size`."""
    from trafilatura3d.paint.session import Session
    return Session.new(str(SHADERBALL), size, evaluator=evaluator)
