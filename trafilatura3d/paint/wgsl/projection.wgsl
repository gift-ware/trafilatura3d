// The Projection tool's (and the stencil's) image in screen space, shared by the paint pass (project.wgsl) and
// the viewport preview (render3d overlay). `s`: viewport coordinates 0-1 (y down); `aspect`: width / height;
// `xf`: offset x, offset y (fraction of the viewport), rotation (radians), scale (1 = the image's height fills
// the viewport's); `tiling`: 0 none, 1 horizontal, 2 vertical, 3 both (the stencil's Tiling Mode), + 4 flipped
// horizontally, + 8 flipped vertically. Returns the
// image coordinates and 1 inside the image (0 where it does not repeat). Ours: the documentation gives the
// controls (S + drag), not the mapping.
// Bilinear sampling with wrap around by loads (float32 textures cannot be filtered by a sampler).
fn bilinear_wrap(t: texture_2d<f32>, uv: vec2f) -> vec4f {
    let sz = vec2i(textureDimensions(t));
    let x = uv * vec2f(sz) - 0.5;
    let i = vec2i(floor(x));
    let f = fract(x);
    let a = textureLoad(t, wrap(i, sz), 0);  // wrap: nodes/int_math.wgsl, prepended by the loaders
    let b = textureLoad(t, wrap(i + vec2i(1, 0), sz), 0);
    let c = textureLoad(t, wrap(i + vec2i(0, 1), sz), 0);
    let d = textureLoad(t, wrap(i + vec2i(1, 1), sz), 0);
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

fn projection_uv(s: vec2f, aspect: f32, xf: vec4f, tiling: i32) -> vec3f {
    var d = (s - 0.5 - xf.xy) * vec2f(aspect, 1.0);
    let c = cos(-xf.z);
    let sn = sin(-xf.z);
    d = vec2f(c * d.x - sn * d.y, sn * d.x + c * d.y) / max(xf.w, 1e-4);
    var q = d + 0.5;
    if ((tiling & 4) != 0) { q.x = 1.0 - q.x; }  // flipped horizontally
    if ((tiling & 8) != 0) { q.y = 1.0 - q.y; }  // flipped vertically
    let out_u = (q.x < 0.0 || q.x > 1.0) && (tiling & 1) == 0;
    let out_v = (q.y < 0.0 || q.y > 1.0) && (tiling & 2) == 0;
    return vec3f(fract(q), select(1.0, 0.0, out_u || out_v));
}

// The projection's weight at a surface point seen from `eye` (Mari-like falloff): 1 facing the view, fading from
// the facing ratio `fade.x` (cos of the start angle) to `fade.y` (cos of the end angle), 0 on back faces.
fn projection_weight(p: vec3f, n: vec3f, eye: vec3f, fade: vec2f) -> f32 {
    let facing = dot(normalize(n), normalize(eye - p));
    return smoothstep(fade.y, fade.x, facing);
}
