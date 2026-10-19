// ---- hdri/_light.wgsl: shared by the HDRI light nodes ----

// Emitted color: RGB, blackbody temperature, an image (sRGB-decoded when asked) or the background
// image `bg` behind the light (linear; clamped to 1 in mode 4).
fn light_rgb(mode: i32, rgb: vec3f, kelvin: f32, img: vec3f, img_srgb: bool, bg: vec3f) -> vec3f {
    if (mode == 1) { return blackbody(kelvin); }
    if (mode == 2) { return select(img, srgb2lin(img), img_srgb); }
    if (mode == 3) { return bg; }
    if (mode == 4) { return min(bg, vec3f(1.0)); }
    return rgb;
}
// The Background Image Input linearized (what Sample Background reads).
fn background_image(c: vec2i) -> vec3f {
    let v = load_background_image_input(c).rgb;
    return select(v, srgb2lin(v), p.background_gamma == 0);
}
// What the light is composited over: the background input (linearized when sRGB) or a color.
fn backdrop(enabled: bool, img: vec3f, img_srgb: bool, color: vec3f) -> vec3f {
    if (enabled) { return select(img, srgb2lin(img), img_srgb); }
    return color;
}
// Coordinates on a light card, q in [-1, 1]^2 -> pattern mask.
fn pattern_mask(kind: i32, q: vec2f, hard: f32, img: f32) -> f32 {
    let soft = mix(0.6, 0.01, sat(hard));
    let r = length(q);
    switch kind {
        case 1: { return step(max(abs(q.x), abs(q.y)), 1.0); }                        // Sharp Square
        case 2: { return pow(sat(1.0 - r), mix(1.5, 0.25, sat(hard))); }               // Cone
        case 3: { return pow(sqrt(max(1.0 - r * r, 0.0)), mix(1.5, 0.25, sat(hard))); } // Hemisphere
        case 4: { return img * step(max(abs(q.x), abs(q.y)), 1.0); }                  // Image Input
        default: {                                                                     // Smooth Square
            let e = max(abs(q.x), abs(q.y));
            return 1.0 - smoothstep(1.0 - soft, 1.0, e);
        }
    }
}
// Position along a card of length/width ratio `aspect` (s in 0..1) -> texture u: Stretch,
// Stretch Middle only (square ends), Repeat + Spacing (square tiles separated by `spacing`; -1 in
// the gaps).
fn along_uv(mode: i32, s: f32, aspect: f32, spacing: f32) -> f32 {
    let a = max(aspect, 1.0);
    if (mode == 1) {
        let x = s * a;
        if (x < 0.5) { return x; }
        if (x > a - 0.5) { return 1.0 - (a - x); }
        return 0.5;
    }
    if (mode == 2) {
        let period = 1.0 + spacing * 2.0;
        let x = s * a;
        let f = x - floor(x / period) * period;
        return select(-1.0, f, f <= 1.0);
    }
    return s;
}
// Debug grid on the ground plane y = h (lines every 0.5 units), seen along d.
fn ground_grid(d: vec3f, h: f32) -> f32 {
    let t = ray_plane_y(d, h);
    if (t < 0.0 || d.y > -0.01) { return 0.0; }
    let q = (d * t).xz * 2.0;
    let w = fwidth_free(t);
    let g = abs(fract(q + 0.5) - 0.5);
    return (1.0 - smoothstep(0.0, 0.03 * w, min(g.x, g.y))) * exp(-t * 0.15);
}
fn fwidth_free(t: f32) -> f32 { return 1.0 + t * 0.6; }
// World coordinates of the parameters (Z up or Y up) -> ours (Y up).
fn to_yup(v: vec3f, zup: bool) -> vec3f { return select(v, vec3f(v.x, v.z, -v.y), zup); }
// A point placed from the panorama: on the ceiling / ground plane, or at a distance.
fn place(mode: i32, uv: vec2f, height: f32, dist: f32, world: vec3f, zup: bool) -> vec3f {
    let d = ll_dir(uv);
    if (mode == 0) {
        let h = select(-1.0, 1.0, d.y >= 0.0) * mix(0.1, 2.0, height);
        let t = ray_plane_y(d, h);
        return d * select(4.0, min(t, 50.0), t > 0.0);
    }
    if (mode == 1) { return d * mix(0.2, 10.0, dist); }
    return to_yup(world, zup);
}
