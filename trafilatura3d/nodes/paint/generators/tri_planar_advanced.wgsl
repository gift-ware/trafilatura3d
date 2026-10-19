fn wrapped_mask(uv: vec2f) -> f32 {
    let d = size_mask();
    let x = uv * vec2f(d) - 0.5;
    let i = vec2i(floor(x));
    let f = fract(x);
    return mix(mix(texel_mask(wrap(i, d)).r, texel_mask(wrap(i + vec2i(1, 0), d)).r, f.x),
               mix(texel_mask(wrap(i + vec2i(0, 1), d)).r, texel_mask(wrap(i + vec2i(1, 1), d)).r, f.x), f.y);
}

// One axis' projection: the plane coordinates rotated (degrees) about the tile centre, then offset.
fn plane(q: vec2f, rotation: f32, offset: vec2f) -> f32 {
    return wrapped_mask(g_rot(q - 0.5, rotation / 360.0) + 0.5 + offset);
}

fn effect(c: vec2i) -> vec4f {
    if (!MASK_CONNECTED) { return vec4f(0.0, 0.0, 0.0, 1.0); }
    let pos = load_position(c).rgb * p.texture_tiling;
    let v = vec3f(plane(pos.zy, p.rotation_x, vec2f(p.offset_x_x, p.offset_x_y)),
                  plane(pos.xz, p.rotation_y, vec2f(p.offset_y_x, p.offset_y_y)),
                  plane(pos.xy, p.rotation_z, vec2f(p.offset_z_x, p.offset_z_y)));
    if (p.projection > 0) { return vec4f(vec3f(v[p.projection - 1]), 1.0); }
    let n = abs(g_normal(load_world_space_normals(c)));
    var w: vec3f;
    if (p.blending_mode == 0) {
        w = pow(n, vec3f(1.0 + p.blending_contrast * 16.0));
    } else {
        let score = n + v * 0.5;
        let band = mix(0.5, 0.01, p.blending_contrast);
        w = max(score - (max(score.x, max(score.y, score.z)) - band), vec3f(0.0));
    }
    w /= max(w.x + w.y + w.z, 1e-6);
    return vec4f(vec3f(dot(w, v)), 1.0);
}
