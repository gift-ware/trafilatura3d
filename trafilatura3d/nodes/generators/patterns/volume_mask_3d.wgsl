fn rx(a: f32) -> mat3x3f { let c = cos(a * TAU); let s = sin(a * TAU); return mat3x3f(1.0, 0.0, 0.0, 0.0, c, s, 0.0, -s, c); }
fn ry(a: f32) -> mat3x3f { let c = cos(a * TAU); let s = sin(a * TAU); return mat3x3f(c, 0.0, -s, 0.0, 1.0, 0.0, s, 0.0, c); }
fn rz(a: f32) -> mat3x3f { let c = cos(a * TAU); let s = sin(a * TAU); return mat3x3f(c, s, 0.0, -s, c, 0.0, 0.0, 0.0, 1.0); }

fn vm_position(uv: vec2f) -> vec3f {
    if (POSITION_CONNECTED) { return sample_position(uv).rgb; }
    return vec3f(uv, 0.0);
}

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let uv = (vec2f(c) + 0.5) / p.size_;
    var pos = vec3f(uv, 0.0);
    if (POSITION_CONNECTED) { pos = load_position(c).rgb; }
    let ctr = select(p.position_world_space, vm_position(p.position_uv), p.position_input == 0);
    // Into the primitive's frame (inverse of Rz * Ry * Rx, angles in half-turns), normalised by the half extents.
    let a = p.rotation * 0.5;
    let m = rz(a.z) * ry(a.y) * rx(a.x);
    let h = max(p.scale * p.size, vec3f(1e-5));
    let n = (transpose(m) * (pos - ctr)) / h;
    var dist: f32;
    switch p.shape {
        case 0: { dist = max(abs(n.x), max(abs(n.y), abs(n.z))); }
        case 1: { dist = max(length(n.xy), abs(n.z)); }
        default: { dist = length(n); }
    }
    // Feather Width 0: a hard edge, antialiased over a pixel.
    let px = 1.0 / (min(p.size_.x, p.size_.y) * min(h.x, h.y));
    var v = clamp(0.5 + (1.0 - dist) / px, 0.0, 1.0);
    if (p.feather_width > px) { v = clamp((1.0 - dist) / p.feather_width, 0.0, 1.0); }
    store(c, vec4f(v, v, v, 1.0));
}
