// The view from the other side (Camera Angle y in (1, 2) modulo 2).
fn he_behind() -> bool { return fract(p.camera_angle.y * 0.5) * 2.0 > 1.0; }

fn he_height(w: vec2f, yaw: mat2x2f) -> f32 {
    var r = yaw * w;
    if (he_behind()) { r.y = -r.y; }
    let uv = vec2f(r.x + 0.5, 0.5 - r.y);
    if (any(uv < vec2f(0.0)) || any(uv > vec2f(1.0))) { return 0.0; }
    return sample_input(uv).r * p.height_scale;
}

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    // Horizontal angle (x - 0.5) half-turns, tilt |1 - y mod 2| half-turns from the top view.
    let yaw_turns = (p.camera_angle.x - 0.5) * 0.5;
    let tilt = abs(1.0 - fract(p.camera_angle.y * 0.5) * 2.0) * 0.5;
    var s = screen_point(c, 0);
    s.y -= select(p.vertical_offset, -p.vertical_offset, he_behind());
    s = s / max(p.camera_scale, 1e-3);
    let pitch = clamp(tilt, 0.0, 0.245) * TAU;
    let v = vec3f(0.0, -sin(pitch), cos(pitch));       // towards the camera
    let up = vec3f(0.0, cos(pitch), sin(pitch));
    let o = vec3f(s.x, 0.0, 0.0) + s.y * up + 4.0 * v;
    let yaw = rot(-yaw_turns);
    let zmax = max(p.height_scale, 1e-4);
    let t0 = (o.z - zmax) / v.z;
    let t1 = o.z / v.z;
    let n = 160;
    var lo = t0;
    var hi = t1;
    var found = false;
    for (var i = 1; i <= n; i++) {
        let t = mix(t0, t1, f32(i) / f32(n));
        let q = o - v * t;
        if (q.z <= he_height(q.xy, yaw)) { hi = t; found = true; break; }
        lo = t;
    }
    for (var i = 0; i < 8 && found; i++) {
        let t = 0.5 * (lo + hi);
        let q = o - v * t;
        if (q.z <= he_height(q.xy, yaw)) { hi = t; } else { lo = t; }
    }
    let q = o - v * hi;
    var out = dot(q, v) + 0.5 * sin(pitch);
    if (p.ground == 0 && he_height(q.xy, yaw) <= 1e-4) { out = 0.0; }
    store(c, vec4f(out, out, out, 1.0));
}
