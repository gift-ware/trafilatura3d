// ---- scan/_clone.wgsl: target / source geometry of the clone patches ----
// A stamp is the area that T(x) = 0.5 + Matrix (x - 0.5) + Offset maps onto the unit square (the Shape
// drawn in it); a target point x shows the source point with the same stamp coordinates.
fn stamp_uv(m: vec4f, o: vec2f, x: vec2f) -> vec2f { return 0.5 + mat2x2f(m.x, m.y, m.z, m.w) * (x - 0.5) + o; }
fn tgt_local(uv: vec2f) -> vec2f { return (stamp_uv(p.target_matrix, p.target_offset, uv) - 0.5) * 2.0; }
fn src_uv(uv: vec2f) -> vec2f {
    return 0.5 + inv2(p.source_matrix) * (stamp_uv(p.target_matrix, p.target_offset, uv) - 0.5 - p.source_offset);
}
// Signed distance to the stamp in its local units (-1..1, < 0 inside); Smoothness rounds the square.
fn stamp_sd(q: vec2f) -> f32 {
    if (p.shape == 1) { return length(q) - 1.0; }
    let r = clamp(p.smoothness * 0.5, 0.0, 1.0);
    let a = abs(q) - vec2f(1.0 - r);
    return length(max(a, vec2f(0.0))) + min(max(a.x, a.y), 0.0) - r;
}
// The Areas outputs: the image lightened inside the target and the source stamps (Show Patch Areas).
fn patch_areas(v: vec4f, uv: vec2f) -> vec4f {
    let t = stamp_sd(tgt_local(uv)) < 0.0;
    let s = stamp_sd((stamp_uv(p.source_matrix, p.source_offset, uv) - 0.5) * 2.0) < 0.0;
    if (p.show_patch_areas == 0 || !(t || s)) { return v; }
    return vec4f(mix(v.rgb, vec3f(1.0), 0.5), v.a);
}
