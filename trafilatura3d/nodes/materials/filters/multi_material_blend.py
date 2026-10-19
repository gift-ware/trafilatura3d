from trafilatura3d.nodes.base import Input
from trafilatura3d.nodes.materials._base import (CHANNELS, GRAY, MatNode, Pass, channel_passes, channel_toggles, color_in,
                                           material_inputs, material_outputs, C, E, F)
from trafilatura3d.nodes.materials.filters._common import BLEND_ON

N = 16


def _params():
    """The defaults: 2 materials, every color green, exact matches (Fuzziness 0.01, no Padding)."""
    ps = {**channel_toggles(BLEND_ON), "materials": E("2", "Materials", tuple(str(i) for i in range(2, N + 1)))}
    for k in range(2, N + 1):
        g = f"Material {k}"
        ps[f"color_{k}"] = C([0.0, 1.0, 0.0], "Color", group=g)
        ps[f"fuzziness_{k}"] = F(0.01, "Fuzziness", 0.01, 1.0, group=g)
        ps[f"padding_{k}"] = F(0.0, "Padding", 0.0, 1.0, group=g)
    return ps


def _body(ch: str, first: int, last: int) -> str:
    lines = []
    for k in range(first, last + 1):
        lines.append(f"    if ({k} <= n) {{ o = put(o, load_{ch}_{k}(c), "
                     f"id_mask(idc, p.color_{k}.rgb, p.fuzziness_{k}, p.padding_{k})); }}")
    return "\n".join(lines)


class MultiMaterialBlend(MatNode):
    """Combines up to 16 materials through a Color ID map: material k (2..16) shows where the ID
    map matches its Color (Fuzziness: tolerance, Padding: hardness); Material 1 is the default
    everywhere else. Per channel, in two passes of eight materials (texture bindings stay low)."""

    type = "multi_material_blend"
    label = "Multi-Material Blend"
    category = "Material filters/Blending (Material)"
    catalog = "multi_material_blend"
    inputs = {**{k: v for i in range(1, N + 1) for k, v in material_inputs(suffix=f"_{i}", label=f"Material {i}").items()},
              "color_id": color_in("Color ID", (0.0, 0.0, 0.0, 1.0))}
    main_input = "diffuse_1"  # the library graph's primary input: Relative to input follows it
    outputs = material_outputs()
    params = _params()

    def encode(self, ctx, job):
        def make(n, ch):
            ins = lambda r: {f"{ch}_{i}": n.inputs[f"{ch}_{i}"] for i in r}
            a = Pass(n, f"{ch}:a", "multi_material_blend.wgsl", {**ins(range(1, 9)), "color_id": n.inputs["color_id"]},
                     {ch: n.outputs[ch]}, channel=ch, subst={"$START": f"load_{ch}_1(c)", "$BODY": _body(ch, 2, 8)})
            b = Pass(n, f"{ch}:b", "multi_material_blend.wgsl",
                     {"acc": Input(CHANNELS[ch][1]), **ins(range(9, N + 1)), "color_id": n.inputs["color_id"]},
                     {ch: n.outputs[ch]}, channel=ch, subst={"$START": "load_acc(c)", "$BODY": _body(ch, 9, N)})
            return a, b
        for ch, (a, b) in channel_passes(self, make).items():
            gray = CHANNELS[ch][1] is GRAY
            t = ctx.scratch(job, gray)
            ctx.dispatch(a, job, job.values, dsts={ch: t})
            ctx.dispatch(b, job, job.values, srcs={"acc": (t, gray)})
