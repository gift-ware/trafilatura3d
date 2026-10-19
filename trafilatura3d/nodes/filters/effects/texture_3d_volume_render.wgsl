fn vol_grid() -> vec2i { return vec2i(16); }
fn vol_res() -> vec2i { return fx_slice_res(size_sdf_3d()); }
fn vol_texel(t: vec2i) -> f32 { return texel_sdf_3d(t).r; }

fn density_at(q: vec3f) -> f32 {
    if (any(q < vec3f(0.0)) || any(q > vec3f(1.0))) { return 0.0; }
    let s = vol_sample(q);
    // Density shares the atlas layout: nearest voxel of its own resolution.
    let dr = fx_slice_res(size_density());
    let v = vol_nearest(q, dr);
    let rho = texel_density(fx_voxel_texel(clamp(v, vec3i(0), vec3i(dr.x - 1, dr.y - 1, fx_slices() - 1)), dr)).r;
    return max(rho, 0.0) * clamp(-s / max(p.feather, 1e-4), 0.0, 1.0);
}

// Transmittance from q along d through the cube (12 steps).
fn transmittance(q: vec3f, d: vec3f) -> f32 {
    let b = vol_box(q, d);
    let len = max(b.y, 0.0);
    let dt = len / 12.0;
    var tau = 0.0;
    for (var i = 0; i < 12; i++) { tau += density_at(q + d * (f32(i) + 0.5) * dt) * dt; }
    return exp(-tau * p.absorption * 256.0);
}

// Blackbody color (Tanner Helland's fit), normalized.
fn kelvin(k: f32) -> vec3f {
    let t = clamp(k, 1000.0, 40000.0) / 100.0;
    var c: vec3f;
    if (t <= 66.0) {
        c.r = 1.0;
        c.g = clamp(0.3900815788 * log(t) - 0.6318414438, 0.0, 1.0);
        c.b = select(clamp(0.5432067891 * log(t - 10.0) - 1.1962540891, 0.0, 1.0), 0.0, t <= 19.0);
    } else {
        c.r = clamp(1.2929361861 * pow(t - 60.0, -0.1332047592), 0.0, 1.0);
        c.g = clamp(1.1298908609 * pow(t - 60.0, -0.0755148492), 0.0, 1.0);
        c.b = 1.0;
    }
    return c;
}

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    var eye = vec3f(0.0);
    let dir = vol_camera(p.camera_position, p.camera_distance, p.camera_fov, fx_uv(c), &eye);
    let l = -vol_dir_from_xy(p.light_position);  // towards the light, as the shadow rays
    var light = p.light_color * p.light_intensity;
    if (p.light_color_mode == 0) { light = kelvin(p.light_temperature_kelvin) * p.light_intensity; }
    if (p.enable_ground_plane == 1 && l.z < 0.0) { light = vec3f(0.0); }  // under the ground: off
    let ambient = p.ambient_color * p.ambient_intensity;

    var bg = p.background_color.rgb;
    if (p.background_mode == 0) { bg = bg * light; }
    var t_end = 1e9;
    if (p.enable_ground_plane == 1 && dir.z < -1e-5) {
        let tg = -eye.z / dir.z;
        let g = eye + dir * tg;
        if (p.infinite_plane == 1 || all(abs(g.xy - 0.5) <= p.ground_plane_size * 0.5)) {
            t_end = tg;
            bg = 0.8 * (light * max(l.z, 0.0) * transmittance(g, l) + ambient);
        }
    }
    var col = vec3f(0.0);
    var tr = 1.0;
    let b = vol_box(eye, dir);
    if (b.y > b.x && b.x < t_end) {
        let steps = 96;
        let dt = (min(b.y, t_end) - b.x) / f32(steps);
        // Interleaved gradient noise: a cheap blue-ish noise for the start offset.
        let ign = fract(52.9829189 * fract(dot(vec2f(c), vec2f(0.06711056, 0.00583715))));
        var t = b.x + dt * ign * p.dithering;
        for (var i = 0; i < steps; i++) {
            let q = eye + dir * t;
            let rho = density_at(q);
            if (rho > 1e-4) {
                let sigma = rho * p.absorption * 256.0;  // unit: 1/256 of the cube
                let lit = light * transmittance(q, l) + ambient * transmittance(q, vec3f(0.0, 0.0, 1.0));
                let a = 1.0 - exp(-sigma * dt);
                col += tr * a * p.albedo * lit;
                tr *= 1.0 - a;
                if (tr < 0.003) { break; }
            }
            t += dt;
        }
    }
    store(c, vec4f(col + tr * bg, 1.0));
}
