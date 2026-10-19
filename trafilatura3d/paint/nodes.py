"""The GPU operations of the layer stack, as helper nodes of the shared evaluator (not in the node library):
the stack compiles to jobs of these, so the evaluator's content-addressed cache recooks only the layers above
a change."""

from __future__ import annotations

from pathlib import Path

from trafilatura3d.nodes.base import Input, Node, Output, Param, ParamKind, PortType

WGSL = Path(__file__).parent / "wgsl"


class _Helper(Node):
    registered = False
    shader = ""

    def wgsl(self):
        return (WGSL / self.shader).read_text(encoding="utf-8")


class Blend(_Helper):
    """A layer over the stack below for one channel (blend.wgsl). The layers of a channel one over the other are
    computed together where the stacks between them are not kept (`BlendChain`): an edit recomputes every layer
    above, a 4096 blend reads three textures and writes one."""
    pointwise = True
    type = "paint_blend"
    shader = "blend.wgsl"
    inputs = {"below": Input(PortType.COLOR), "top": Input(PortType.COLOR), "mask": Input(PortType.GRAY, default=(1, 1, 1, 1))}
    outputs = {"output": Output(PortType.COLOR)}
    params = {"mode": Param(ParamKind.INT, 0), "opacity": Param(ParamKind.FLOAT, 1.0),
              "linear": Param(ParamKind.BOOL, False), "clamp01": Param(ParamKind.BOOL, True),
              "normal_": Param(ParamKind.BOOL, False), "keep_alpha": Param(ParamKind.BOOL, False)}
    fuse_input = "below"
    max_fused = 6  # below + a top and a mask per layer: 13 textures, the 16 a shader reads by default

    def wgsl(self):
        return (WGSL / "blend_lib.wgsl").read_text(encoding="utf-8") + (WGSL / self.shader).read_text(encoding="utf-8")

    def encode_fused(self, ctx, chain):
        from trafilatura3d.evaluator import DEPTHS
        ev, job = ctx.ev, chain[-1]
        srcs, values = {}, {"steps": len(chain), "truncate": ev.half_store_truncates()}
        below = chain[0].inputs.get("below")
        if below:
            srcs["below"] = (ev.texture(*below), below[0].is_gray(below[1]))
        for i, j in enumerate(chain):
            for port in ("top", "mask"):
                s = j.inputs.get(port)
                if s:
                    srcs[f"{port}{i}"] = (ev.texture(*s), s[0].is_gray(s[1]))
            values.update({f"{k}{i}": v for k, v in j.values.items()})
            q, kind = 0.0, 0
            if j is not job:  # what storing the result and reading it back does to it (its format, its depth)
                fmt = ev.storage_format(j.depths["output"], j.is_gray("output"),
                                        j.output_types["output"] is PortType.GRAY_ALPHA)
                q = DEPTHS[j.depths["output"]][0]
                kind = (2 if "8unorm" in fmt else 1 if "16float" in fmt else 0) + (
                    4 if fmt.startswith("rg") and not fmt.startswith("rgba") else 8 if fmt.startswith("r") and
                    not fmt.startswith("rg") else 0)
            values[f"q{i}"], values[f"kind{i}"] = q, kind
        for i in range(len(chain), self.max_fused):
            values.update({f"{k}{i}": p.default for k, p in self.params.items()})
            values[f"q{i}"], values[f"kind{i}"] = 0.0, 0
        ctx.dispatch(BLEND_CHAIN, job, values, srcs=srcs)


class BlendChain(_Helper):
    """Up to Blend.max_fused layers of a channel over the stack below in one dispatch (Blend.encode_fused): each
    step's result goes through what storing and reading it back would do (quantization to its depth, rounding to
    its storage format, the channels it keeps), so the chain gives the blends' texels exactly."""
    pointwise = True
    type = "paint_blend_chain"
    inputs = {"below": Input(PortType.COLOR),
              **{f"{port}{i}": Input(PortType.GRAY, default=(1, 1, 1, 1)) if port == "mask" else Input(PortType.COLOR)
                 for i in range(Blend.max_fused) for port in ("top", "mask")}}
    outputs = {"output": Output(PortType.COLOR)}
    params = {**{f"{k}{i}": p for i in range(Blend.max_fused) for k, p in Blend.params.items()},
              **{f"q{i}": Param(ParamKind.FLOAT, 0.0) for i in range(Blend.max_fused)},
              **{f"kind{i}": Param(ParamKind.INT, 0) for i in range(Blend.max_fused)},
              "steps": Param(ParamKind.INT, 1),
              "truncate": Param(ParamKind.BOOL, False)}  # half floats are stored rounded toward zero

    def wgsl(self):
        steps = "".join(f"""
    if (p.steps > {i}) {{{f" v = stored(v, p.q{i - 1}, p.kind{i - 1});" if i else ""}
        var m{i} = 1.0;
        if (MASK{i}_CONNECTED) {{ m{i} = clamp(load_mask{i}(c).r, 0.0, 1.0); }}
        v = lay(v, load_top{i}(c), m{i}, p.mode{i}, p.opacity{i}, p.linear{i} != 0, p.clamp01{i} != 0,
                p.normal_{i} != 0, p.keep_alpha{i} != 0);
    }}""" for i in range(Blend.max_fused))
        return (WGSL / "blend_lib.wgsl").read_text(encoding="utf-8") + """
// A float as a half rounded toward zero, from its bits: comparing unpack2x16float(pack2x16float(v)) with v does not
// work everywhere (NVIDIA's compiler, Vulkan: the comparison never holds). Too large: the largest half; NaN stays NaN.
fn half_toward_zero(x: f32) -> u32 {
    let b = bitcast<u32>(x);
    let s = (b >> 16u) & 0x8000u;
    let e = i32((b >> 23u) & 0xffu) - 112;  // the half's exponent field
    let m = b & 0x7fffffu;
    if (e >= 31) { return s | select(0x7bffu, select(0x7c00u, 0x7e00u, m != 0u), e == 143); }
    if (e <= 0) { return s | select(0u, (m | 0x800000u) >> u32(14 - e), e > -11); }  // a subnormal half, or 0
    return s | (u32(e) << 10u) | (m >> 13u);
}

// Two values as half floats: pack2x16float rounds to the nearest, a texture write may round toward zero
// (p.truncate, Evaluator.half_store_truncates).
fn half2(v: vec2f) -> vec2f {
    if (p.truncate != 0) { return unpack2x16float(half_toward_zero(v.x) | (half_toward_zero(v.y) << 16u)); }
    return unpack2x16float(pack2x16float(v));
}

// A result stored and read back: quantized to its depth (q: levels - 1, 0 for float), rounded to its format (kind
// & 3: 0 32-bit float, 1 half float, 2 8-bit unorm), its channels (kind & 12: 4 gray and alpha, 8 gray: alpha 1).
fn stored(v: vec4f, q: f32, kind: i32) -> vec4f {
    var o = v;
    if (q > 0.0) { o = quantize_depth(v, q); }
    switch kind & 3 {
        case 1: { o = vec4f(half2(o.xy), half2(o.zw)); }
        case 2: { o = unpack4x8unorm(pack4x8unorm(o)); }
        default: {}
    }
    if ((kind & 4) != 0) { return vec4f(o.r, o.r, o.r, o.a); }
    if ((kind & 8) != 0) { return vec4f(o.r, o.r, o.r, 1.0); }
    return o;
}

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let c = vec2i(id.xy) + origin();
    if (any(c >= size())) { return; }
    var v = load_below(c);""" + steps + """
    store(c, v);
}
"""


class Fill(_Helper):
    """A uniform value or an image through the fill's projection (fill.wgsl): UV, Tri-planar or Planar, the 3D
    ones reading the mesh in texture space (`position`, `geonormal`: the Texture Set's uvmaps)."""
    pointwise = True
    type = "paint_fill"
    shader = "fill.wgsl"
    inputs = {"image": Input(PortType.COLOR), "position": Input(PortType.COLOR), "geonormal": Input(PortType.COLOR),
              "warp": Input(PortType.COLOR),  # warp: the Warp projection's surface sampled (paint/warp.py)
              "sym": Input(PortType.COLOR)}  # sym: the symmetry's copies (stack.py `_projection`)
    outputs = {"output": Output(PortType.COLOR)}
    params = {"color": Param(ParamKind.COLOR, [0.5, 0.5, 0.5, 1.0]), "alpha": Param(ParamKind.FLOAT, 1.0),
              "tiling": Param(ParamKind.FLOAT, 1.0), "rotation": Param(ParamKind.FLOAT, 0.0),
              "offset": Param(ParamKind.FLOAT2, [0.0, 0.0]), "wrap": Param(ParamKind.INT, 3),
              "filtering": Param(ParamKind.INT, 0), "gray": Param(ParamKind.BOOL, False),
              "flip_green": Param(ParamKind.BOOL, False), "signed_": Param(ParamKind.BOOL, False),
              # projection: 0 UV, 1 Tri-planar, 2 Planar; m0..m2 rows of world -> projection space
              "mode": Param(ParamKind.INT, 0), "m0": Param(ParamKind.FLOAT4, [1.0, 0.0, 0.0, 0.0]),
              "m1": Param(ParamKind.FLOAT4, [0.0, 1.0, 0.0, 0.0]), "m2": Param(ParamKind.FLOAT4, [0.0, 0.0, 1.0, 0.0]),
              "crop": Param(ParamKind.BOOL, False), "hardness": Param(ParamKind.FLOAT, 0.5),
              "depth_culling": Param(ParamKind.BOOL, False), "depth_hardness": Param(ParamKind.FLOAT, 0.5),
              "backface_culling": Param(ParamKind.BOOL, False), "backface_cos": Param(ParamKind.FLOAT, 0.0),
              "backface_hardness": Param(ParamKind.FLOAT, 0.5), "angle": Param(ParamKind.FLOAT, 6.2831853),
              "warp_depth": Param(ParamKind.FLOAT, 0.25)}

    def encode(self, ctx, job):
        srcs = {}
        if job.extra.get("image_texture"):  # a texture outside the cache (a material's output)
            srcs["image"] = job.extra["image_texture"]
        if job.extra.get("position_texture"):  # the mesh in texture space (3D projections)
            srcs["position"] = job.extra["position_texture"]
            srcs["geonormal"] = job.extra["normal_texture"]
        if job.extra.get("warp_texture"):
            srcs["warp"] = job.extra["warp_texture"]
        if job.extra.get("sym_texture"):
            srcs["sym"] = job.extra["sym_texture"]
        ctx.dispatch(self, job, job.values, srcs=srcs or None)


class Copy(_Helper):
    """Painted pixels (paint.Pages, expanded in `encode` into a scratch over the region cooked), a texture or a
    project resource (its pixels, uploaded for this cook) into the stack (copy.wgsl)."""
    pointwise = True
    type = "paint_copy"
    shader = "copy.wgsl"
    inputs = {"image": Input(PortType.COLOR)}
    outputs = {"output": Output(PortType.COLOR)}
    params = {"gray": Param(ParamKind.BOOL, False)}

    def encode(self, ctx, job):
        from trafilatura3d.evaluator import one_channel
        pages = job.extra.get("pages")
        if "texture" in job.extra:  # a whole texture (a computed mesh map, the UV mask)
            tex = job.extra["texture"]
        elif pages is None:
            tex = ctx.upload(job.extra["image"])
        else:
            tex = ctx.scratch(job, False, "32f" if pages.format == "rgba32float" else "16f", size=pages.size)
            region = ctx.region if ctx.region is not None and pages.size == tuple(job.size) else None
            ctx.render(lambda enc: pages.expand(enc, tex, region), uses=[tex])
        ctx.dispatch(self, job, job.values, srcs={"image": (tex, one_channel(tex))})


class LevelsNode(_Helper):
    """The Levels effect (levels.wgsl)."""
    pointwise = True
    type = "paint_levels"
    shader = "levels.wgsl"
    inputs = {"image": Input(PortType.COLOR)}
    outputs = {"output": Output(PortType.COLOR)}
    params = {"in_low": Param(ParamKind.FLOAT, 0.0), "in_mid": Param(ParamKind.FLOAT, 0.5),
              "in_high": Param(ParamKind.FLOAT, 1.0), "out_low": Param(ParamKind.FLOAT, 0.0),
              "out_high": Param(ParamKind.FLOAT, 1.0), "component": Param(ParamKind.INT, 0),
              "clamp01": Param(ParamKind.BOOL, True)}


class NormalMix(_Helper):
    """Baked normal + Normal channel + Height channel (normal.wgsl)."""
    pointwise = True
    type = "paint_normal"
    shader = "normal.wgsl"
    inputs = {"baked": Input(PortType.COLOR), "normal": Input(PortType.COLOR), "height": Input(PortType.COLOR)}
    outputs = {"output": Output(PortType.COLOR)}
    params = {"mixing": Param(ParamKind.INT, 1), "sobel": Param(ParamKind.BOOL, True),
              "strength": Param(ParamKind.FLOAT, 1.0), "dx_in": Param(ParamKind.BOOL, False),
              "dx_out": Param(ParamKind.BOOL, False)}


class Dilate(_Helper):
    """Padding (dilate.wgsl, stage 2): texels outside the UV islands take the value of the nearest texel inside,
    read from the Nearest job's map (port "seeds")."""
    type = "paint_dilate"
    shader = "dilate.wgsl"
    inputs = {"image": Input(PortType.COLOR), "mask": Input(PortType.GRAY), "seeds": Input(PortType.COLOR)}
    outputs = {"output": Output(PortType.COLOR)}
    params = {"distance": Param(ParamKind.FLOAT, -1.0), "outside": Param(ParamKind.INT, 0),
              "color": Param(ParamKind.COLOR, [0.0, 0.0, 0.0, 0.0])}
    internal_uniforms = {"stage": ParamKind.INT, "step": ParamKind.INT}

    def encode(self, ctx, job):
        ctx.dispatch(self, job, {**job.values, "stage": 2, "step": 0})


class Nearest(Dilate):
    """The nearest texel inside the UV islands for every texel (jump flooding: seed, flood passes): depends on
    the UV mask only, so the evaluator keeps it while the stack changes and padding is one pass per channel."""
    type = "paint_nearest"

    def encode(self, ctx, job):
        w, h = job.size
        a, b = ctx.scratch(job, True, alpha=True), ctx.scratch(job, True, alpha=True)  # seeds: two channels
        values = {**job.values, "stage": 0, "step": 0}
        ctx.dispatch(self, job, values, srcs={"seeds": (b, True)}, dsts={"output": a})  # never read the target
        step = 1 << max(max(w, h) - 1, 1).bit_length() - 1
        cur, nxt = a, b
        while step >= 1:
            ctx.dispatch(self, job, {**job.values, "stage": 1, "step": step}, srcs={"seeds": (cur, True)},
                         dsts={"output": nxt})
            cur, nxt = nxt, cur
            step //= 2
        ctx.dispatch(self, job, {**job.values, "stage": 3, "step": 0}, srcs={"seeds": (cur, True)})


class Convert(_Helper):
    """A channel between the Paint and Graph material conventions (convert.wgsl): RGB * scale + offset
    (the signed height <-> 0.5 = flat), green flipped (DirectX <-> OpenGL normals); alpha kept."""
    pointwise = True
    type = "paint_convert"
    shader = "convert.wgsl"
    inputs = {"image": Input(PortType.COLOR)}
    outputs = {"output": Output(PortType.COLOR)}
    params = {"scale": Param(ParamKind.FLOAT, 1.0), "offset": Param(ParamKind.FLOAT, 0.0),
              "flip_green": Param(ParamKind.BOOL, False)}


class Erase(_Helper):
    """Erase along path: the below's alpha times 1 - the top's (erase.wgsl)."""
    pointwise = True
    type = "paint_erase"
    shader = "erase.wgsl"
    inputs = {"below": Input(PortType.COLOR), "top": Input(PortType.COLOR)}
    outputs = {"output": Output(PortType.COLOR)}


BLEND, FILL, COPY, NORMAL, DILATE, NEAREST, LEVELS, CONVERT, ERASE = (
    Blend(), Fill(), Copy(), NormalMix(), Dilate(), Nearest(), LevelsNode(), Convert(), Erase())
BLEND_CHAIN = BlendChain()
