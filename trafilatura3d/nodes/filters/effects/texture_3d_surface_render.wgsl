fn vol_grid() -> vec2i { return vec2i(16); }
fn vol_res() -> vec2i { return fx_slice_res(size_distance_field_3d()); }
fn vol_texel(t: vec2i) -> f32 { return texel_distance_field_3d(t).r; }

// Distance to the shape: the field inside the cube, the distance to the cube plus a margin outside.
fn sdf(q: vec3f) -> f32 {
    let out = max(abs(q - 0.5) - 0.5, vec3f(0.0));
    let o = length(out);
    if (o > 0.0) { return o + 0.01; }
    return vol_sample(q);
}

fn env(d: vec3f) -> vec3f { return sample_environment(fx_latlong(d, 0.0)).rgb; }

// Soft visibility along d from q (1 = open) by marching the distance field.
fn visibility(q: vec3f, d: vec3f) -> f32 {
    var t = 0.02;
    var v = 1.0;
    for (var i = 0; i < 12; i++) {
        let s = sdf(q + d * t);
        v = min(v, clamp(8.0 * s / t, 0.0, 1.0));
        if (v < 0.01) { break; }
        t += max(s, 0.02);
        if (t > 1.8) { break; }
    }
    return v;
}

fn irradiance(q: vec3f, n: vec3f, r: vec2f, shadows: bool) -> vec3f {
    var acc = vec3f(0.0);
    for (var i = 0; i < 16; i++) {
        let d = ray_dir_cos(i, 16, r, n);
        var v = 1.0;
        if (shadows) { v = visibility(q, d); }
        acc += env(d) * v;
    }
    return acc / 16.0;
}

fn ray_dir_cos(i: i32, n: i32, r: vec2f, axis: vec3f) -> vec3f {
    let f = fx_fib(i, n, r);
    let ct = sqrt(1.0 - f.x);
    let st = sqrt(f.x);
    let ph = f.y * 2.0 * PI;
    return fx_basis(axis) * vec3f(st * cos(ph), st * sin(ph), ct);
}

fn sdf_normal(q: vec3f) -> vec3f {
    let e = 1.5 / f32(max(vol_res().x, 8));
    return normalize(vec3f(sdf(q + vec3f(e, 0.0, 0.0)) - sdf(q - vec3f(e, 0.0, 0.0)),
                           sdf(q + vec3f(0.0, e, 0.0)) - sdf(q - vec3f(0.0, e, 0.0)),
                           sdf(q + vec3f(0.0, 0.0, e)) - sdf(q - vec3f(0.0, 0.0, e))) + vec3f(1e-6));
}

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    var eye = vec3f(0.0);
    let dir = vol_camera(p.camera_position, p.camera_distance, p.camera_fov, fx_uv(c), &eye);
    let r = hash22(c, p.seed_ + 31.0);
    // Background.
    var col = p.background_color.rgb;
    if (p.background_mode == 0) { col = irradiance(vec3f(0.0), vec3f(0.0, 0.0, 1.0), vec2f(0.0), false); }
    if (p.background_mode == 1) { col = irradiance(vec3f(0.0), dir, vec2f(0.0), false); }
    if (p.background_mode == 3) { col = env(dir); }
    var t_end = 1e9;
    // Ground plane under the cube (z = 0).
    if (p.enable_ground_plane == 1 && dir.z < -1e-5) {
        let tg = -eye.z / dir.z;
        let g = eye + dir * tg;
        let half_size = p.ground_plane_size * 0.5;
        if (p.infinite_plane == 1 || all(abs(g.xy - 0.5) <= half_size)) {
            t_end = tg;
            col = 0.8 * irradiance(g, vec3f(0.0, 0.0, 1.0), r, true);
        }
    }
    // The shape.
    let b = vol_box(eye, dir);
    if (b.y > b.x && b.x < t_end) {
        var t = b.x;
        for (var i = 0; i < 128; i++) {
            let q = eye + dir * t;
            let s = vol_sample(q);
            if (s < 0.001) {
                let n = sdf_normal(q);
                col = p.albedo * irradiance(q + n * 0.01, n, r, true);
                break;
            }
            t += max(s, 0.002);
            if (t > b.y) { break; }
        }
    }
    store(c, vec4f(col, 1.0));
}
