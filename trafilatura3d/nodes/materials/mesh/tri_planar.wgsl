fn plane_uv(q: vec2f, rot: f32, off: vec2f, roff: f32, axis: i32) -> vec2f {
    let r = hash22(vec2i(axis, 7), p.seed_) * roff;
    return fract(rot2(q * p.texture_tiling, (p.global_rotation + rot) * TAU) + off + r);
}

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let pos = load_position(c).rgb;
    let n = normalize(load_world_space_normal(c).rgb * 2.0 - 1.0 + vec3f(0.0, 1e-5, 0.0));
    var qx = vec2f(pos.z, 1.0 - pos.y);
    var qy = vec2f(pos.x, pos.z);
    var qz = vec2f(pos.x, 1.0 - pos.y);
    if (p.fix_mirrored_projection != 0) {
        if (n.x < 0.0) { qx.x = 1.0 - qx.x; }
        if (n.y < 0.0) { qy.x = 1.0 - qy.x; }
        if (n.z < 0.0) { qz.x = 1.0 - qz.x; }
    }
    let ux = plane_uv(qx, p.rotation_x, p.offset_x, p.random_offset_x, 0);
    let uy = plane_uv(qy, p.rotation_y, p.offset_y, p.random_offset_y, 1);
    let uz = plane_uv(qz, p.rotation_z, p.offset_z, p.random_offset_z, 2);
    let vx = sample_input_x(ux);
    var vy = sample_input_x(uy);
    var vz = sample_input_x(uz);
    if (p.image_inputs == 1) { vy = sample_input_y(uy); vz = sample_input_z(uz); }
    var w = tri_weights(n, p.blending_contrast);
    if (p.blending_mode == 1) {
        w *= vec3f(luma(vx.rgb), luma(vy.rgb), luma(vz.rgb)) + 0.25;
        w = pow(w, vec3f(1.0 + p.blending_contrast * 4.0));
        w /= max(w.x + w.y + w.z, 1e-5);
    }
    if (p.projection == 1) { w = vec3f(1.0, 0.0, 0.0); }
    if (p.projection == 2) { w = vec3f(0.0, 1.0, 0.0); }
    if (p.projection == 3) { w = vec3f(0.0, 0.0, 1.0); }
    var v = vx * w.x + vy * w.y + vz * w.z;
    // Mixing projections pulls values to their mean: scale the deviation back.
    let k = 1.0 / sqrt(max(dot(w, w), 1e-3));
    v = vec4f(mix(v.rgb, (v.rgb - 0.5) * k + 0.5, p.normalization_factor), v.a);
    store(c, v);
}
