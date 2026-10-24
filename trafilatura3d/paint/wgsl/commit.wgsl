// The stroke into a painted entry: k = min(stroke coverage, Stroke Opacity); painting lays the value over what
// the entry had before the stroke ("over" with straight alpha), erasing lowers its alpha (the Eraser page: the
// eraser sets the layer alpha back toward zero). The value is the tool's uniform one, or per texel (`src`, in
// texture space: the Projection tool's image, the Clone tool's source…) whose alpha scales k. FORMAT is
// replaced by the entry's storage format.
//
// The entry is sparse (paint.Pages): `dst` holds its pages, a layer each. Over the area `u.area` of this pass,
// `table` gives per page (x) what it had before the stroke: 0 not changed by the stroke (left as it is), 1
// nothing painted, else its copy saved the first time the stroke changed it (`saved`, layer x - 2); (y) its
// layer in `dst` + 1 (0: no page, the Eraser over nothing). A Smudge reads what it had from `before` instead
// (whole: the state before each stamp).

// Blended (`mode.y` >= 0, Polygon Fill's Blending): the value is first combined with what the texel held in that
// blending mode (blend_lib.wgsl, prepended: the layer stack's rules for the channel), then laid as above.
struct Uniforms {
    value: vec4f,   // the value painted (RGB; grayscale channels repeat it), A: the stencil applies (1)
    params: vec4f,  // stroke opacity (blended: the blending's), erase (1), per texel value (1), protected by `limit` (1)
    area: vec4u,    // the pass's corner (x, y), page side, pages per row
    mode: vec4f,    // x: what the pages had is in `before` (Smudge); y: blending mode or -1; z: its flags (1 linear,
                    // 2 clamped, 4 normal)
};
@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var before: texture_2d<f32>;
@group(0) @binding(2) var stroke: texture_2d<f32>;
@group(0) @binding(3) var dst: texture_storage_2d_array<FORMAT, write>;
@group(0) @binding(4) var src: texture_2d<f32>;
// The quick mask (Quick Mask page): R = how much a texel is protected from painting.
@group(0) @binding(5) var limit: texture_2d<f32>;
// The brush's stencil seen from the stroke's view, in texture space (R: its grayscale, A: inside the image).
@group(0) @binding(6) var stencil: texture_2d<f32>;
@group(0) @binding(7) var saved: texture_2d_array<f32>;
@group(0) @binding(8) var<storage, read> table: array<vec2u>;
const IS_TWO = TWO_ENTRY;

fn commit_value(b: vec4f, c: vec2i) -> vec4f {
    var k = min(textureLoad(stroke, c, 0).r, u.params.x);
    var value = u.value.rgb;
    if (u.params.z > 0.5) {
        let s = textureLoad(src, min(c, vec2i(textureDimensions(src)) - 1), 0);
        value = s.rgb;
        k *= s.a;
    }
    if (u.mode.y >= 0.0 && b.a > 1e-6) {  // over nothing painted the value goes in as it is
        let f = u32(u.mode.z);
        value = lay(vec4f(b.rgb, 1.0), vec4f(value, 1.0), 1.0, i32(u.mode.y), 1.0, (f & 1u) != 0u, (f & 2u) != 0u,
                    (f & 4u) != 0u, false).rgb;
    }
    if (u.params.w > 0.5) {
        k *= 1.0 - clamp(textureLoad(limit, min(c, vec2i(textureDimensions(limit)) - 1), 0).r, 0.0, 1.0);
    }
    if (u.value.w > 0.5) {  // outside the stencil's image nothing is masked (ours: the page does not say)
        let st = textureLoad(stencil, min(c, vec2i(textureDimensions(stencil)) - 1), 0);
        k *= mix(1.0, clamp(st.r, 0.0, 1.0), st.a);
    }
    if (u.params.y > 0.5) {
        return vec4f(b.rgb, b.a * (1.0 - k));
    }
    let a = b.a + (1.0 - b.a) * k;
    let rgb = select(b.rgb, (value * k + b.rgb * b.a * (1.0 - k)) / a, a > 1e-6);
    return vec4f(rgb, a);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let c = vec2i(id.xy) + vec2i(u.area.xy);
    if (any(c >= vec2i(textureDimensions(stroke)))) { return; }
    let size = i32(u.area.z);
    let page = table[(c.y / size) * i32(u.area.w) + c.x / size];
    if (page.x == 0u || page.y == 0u) { return; }
    var b = vec4f(0.0);
    if (u.mode.x > 0.5) {
        b = textureLoad(before, c, 0);
    } else if (page.x > 1u) {
        b = textureLoad(saved, c % size, i32(page.x) - 2, 0);
        if (IS_TWO) { b = b.rrrg; }
    }
    var o = commit_value(b, c);
    if (IS_TWO) { o = vec4f(o.r, o.a, 0.0, 0.0); }
    textureStore(dst, c % size, i32(page.y) - 1, o);
}
