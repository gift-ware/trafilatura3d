
// --- caustics: shared optics (UV units of a square image; z up) -------------------------------

const DEV_MAX = 0.05;
const HEIGHT_MAX = 0.15;  // surface relief at Surface Height Scale 1, in UV

fn incident(light: vec2f) -> vec3f { return normalize(vec3f(0.5 - light.x, 0.5 - light.y, -1.0)); }
// Distance from the refracting surface to the projection plane.
fn plane_depth(pos: f32) -> f32 { return 0.03 + clamp(pos, 0.0, 1.0) * 0.4; }
// Landing offset on the plane of a ray refracted by a surface of normal nrm.
fn landing(i: vec3f, nrm: vec3f, ior: f32, depth: f32) -> vec2f {
    let r = refract(i, nrm, 1.0 / max(ior, 1.0));
    return r.xy * depth / max(-r.z, 0.05);
}
fn landing_flat(light: vec2f, ior: f32, depth: f32) -> vec2f {
    return landing(incident(light), vec3f(0.0, 0.0, 1.0), ior, depth);
}
fn channel_ior(ior: f32, disp: f32, ch: i32) -> f32 { return ior + disp * 4.0 * f32(ch - 1); }
// Photon source position (UV) of grid cell q, jittered.
fn photon_pos(q: vec2i, n: i32, jit: f32) -> vec2f {
    let w = wrap(q, vec2i(n));
    return (vec2f(q) + 0.5 + (hash22(w, p.seed_ + 3.0) - 0.5) * jit) / f32(n);
}
