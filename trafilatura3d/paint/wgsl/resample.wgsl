// The Clone and Smudge tools' values in texture space: every texel seen from the stroke's view (its screen point
// s), the content at s + delta on the screen (the texel under that pixel, from the camera buffer's UVs, on the
// same Texture Set). Content = the painted entry over the stack below when the layer is in Passthrough (`below`
// bound), else the entry alone. Alpha 0 where nothing can be taken (hidden texel, no surface at the source, a
// source in another UDIM tile: each tile is its own texture, `tile` moves the UVs into this one's square).

struct Uniforms {
    view_proj: mat4x4f,
    params: vec4f,   // delta x, delta y (camera buffer pixels), material + 1, has below (1)
    viewport: vec4f, // camera buffer width, height, depth tolerance, texture space mode (1: delta in UV)
    eye: vec4f,
    tile: vec4f,     // the UV offset into the tile's square (-U, +V of the tile), `below` a grayscale value and alpha (1)
};
@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var pos_map: texture_2d<f32>;
@group(0) @binding(2) var cam_pos: texture_2d<f32>;
@group(0) @binding(3) var cam_uv: texture_2d<f32>;
@group(0) @binding(4) var entry: texture_2d<f32>;
@group(0) @binding(5) var below: texture_2d<f32>;
@group(0) @binding(6) var dst: texture_storage_2d<rgba32float, write>;

fn bilinear(t: texture_2d<f32>, uv: vec2f) -> vec4f {
    let sz = vec2i(textureDimensions(t));
    let x = uv * vec2f(sz) - 0.5;
    let i = vec2i(floor(x));
    let f = fract(x);
    let a = textureLoad(t, clamp(i, vec2i(0), sz - 1), 0);
    let b = textureLoad(t, clamp(i + vec2i(1, 0), vec2i(0), sz - 1), 0);
    let c = textureLoad(t, clamp(i + vec2i(0, 1), vec2i(0), sz - 1), 0);
    let d = textureLoad(t, clamp(i + vec2i(1, 1), vec2i(0), sz - 1), 0);
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

fn content(uv: vec2f) -> vec4f {
    let e = bilinear(entry, uv);
    if (u.params.w < 0.5) { return e; }
    var b = bilinear(below, uv);
    if (u.tile.z > 0.5) { b = vec4f(b.rrr, b.g); }  // two channels: (v, a)
    return vec4f(mix(b.rgb, e.rgb, e.a), 1.0);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let c = vec2i(id.xy);
    if (any(c >= vec2i(textureDimensions(pos_map)))) { return; }
    var v = vec4f(0.0);
    let pw = textureLoad(pos_map, c, 0);
    if (u.viewport.w > 0.5) {  // in texture space (Smudge along path: no view), delta in UV
        if (pw.a > 0.5) {
            v = content((vec2f(c) + 0.5) / vec2f(textureDimensions(pos_map)) + u.params.xy);
        }
        textureStore(dst, c, v);
        return;
    }
    let clip = u.view_proj * vec4f(pw.xyz, 1.0);
    if (pw.a > 0.5 && clip.w > 0.0) {
        let ndc = clip.xy / clip.w;
        let s = vec2f(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5) * u.viewport.xy;
        let q = vec2i(floor(s));
        let qs = vec2i(floor(s + u.params.xy));
        let vp = vec2i(u.viewport.xy);
        if (all(q >= vec2i(0)) && all(q < vp) && all(qs >= vec2i(0)) && all(qs < vp)) {
            let seen = textureLoad(cam_pos, q, 0);  // the texel is the visible surface at its pixel
            let src = textureLoad(cam_pos, qs, 0);
            let visible = distance(seen.xyz, u.eye.xyz) + u.viewport.z >= distance(pw.xyz, u.eye.xyz);
            let uv = textureLoad(cam_uv, qs, 0).xy + u.tile.xy;
            let inside = all(uv >= vec2f(0.0)) && all(uv <= vec2f(1.0));
            if (seen.a > 0.0 && visible && abs(src.a - u.params.z) < 0.5 && inside) {
                v = content(uv);
            }
        }
    }
    textureStore(dst, c, v);
}
