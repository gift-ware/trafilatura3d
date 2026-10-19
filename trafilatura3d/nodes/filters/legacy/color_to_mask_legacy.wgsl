const LUMA = vec3f(0.299, 0.587, 0.114);  // Rec. 601
const EDGE = 5e-5;  // a hard cut keeps the colours this close past the range (the reference at range 0)

// Distance from the pixel's colour to Color by Keying type (RGB, Chrominance, Luminance).
fn key_distance(a: vec3f, b: vec3f) -> f32 {
    let la = dot(a, LUMA);
    let lb = dot(b, LUMA);
    switch p.keying_type {
        case 1: { let d = abs((a - la) - (b - lb)); return d.x + d.y + d.z; }
        case 2: { return abs(la - lb); }
        default: { let d = abs(a - b); return d.x + d.y + d.z; }
    }
}

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let dist = key_distance(load_input(c).rgb, p.color);  // alpha ignored: Flatten alpha has nothing to change
    let t = p.mask_range * 0.25;
    let fade = p.mask_softness * t;
    var m = select(0.0, 1.0, dist <= t + EDGE);
    if (fade > EDGE) { m = clamp((t - dist) / fade, 0.0, 1.0); }
    store(c, gray4(m));
}
