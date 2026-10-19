// Square coordinates of `dir` in the tangent plane at `ctr` (rotated, divided by the half size).
fn local_q(dir: vec3f, ctr: vec3f, rot: f32, half: vec2f) -> vec3f {
    let f = frame(ctr);
    let z = dot(dir, ctr);
    let q = rot2(vec2f(dot(dir, f[0]), dot(dir, f[1])) / max(z, 1e-4), -rot) / half;
    return vec3f(q, z);
}

fn frame_line(q: vec3f, w: f32) -> f32 {
    if (q.z <= 0.0) { return 0.0; }
    let e = max(abs(q.x), abs(q.y));
    return step(abs(e - 1.0), w);
}

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let dir = ll_dir(uv_of(c, d));
    let base = load_input(c);
    let half = max(p.patch_scale * 2.0 * p.patch_size, vec2f(1e-3));
    let tgt = ll_dir(p.patch_position);
    let src = ll_dir(p.source_position);
    let rot = p.patch_rotation * TAU;
    let q = local_q(dir, tgt, rot, half);
    var col = base.rgb;
    if (p.enable_ != 0 && q.z > 0.0 && max(abs(q.x), abs(q.y)) < 1.5) {
        var a = 0.0;
        let e = max(abs(q.x), abs(q.y));
        let soft = mix(0.7, 0.02, sat(p.patch_hardness));
        if (p.patch_alpha == 0) { a = 1.0 - smoothstep(1.0 - soft, 1.0, e); }
        else if (p.patch_alpha == 1) { a = exp(-dot(q.xy, q.xy) * mix(1.5, 12.0, sat(p.patch_hardness))); }
        else { a = sample_mask_input(q.xy * 0.5 + 0.5).r * step(e, 1.0); }
        // Same square coordinates in the source frame.
        let fs = frame(src);
        let qs = rot2(q.xy * half, rot + p.source_rotation_offset * TAU);
        let sd = normalize(src + fs[0] * qs.x + fs[1] * qs.y);
        col = mix(col, sample_input(dir_ll(sd)).rgb, a);
    }
    if (p.show_frames_helper != 0) {
        let w = 0.01 + p.frame_thickness * 0.08;
        let qs = local_q(dir, src, rot + p.source_rotation_offset * TAU, half);
        col = mix(col, vec3f(0.0, 1.0, 0.0), frame_line(q, w));
        col = mix(col, vec3f(1.0, 0.0, 0.0), frame_line(qs, w));
    }
    store(c, vec4f(col, base.a));
}
