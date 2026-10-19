// Lat-long panoramas, the convention of the library's HDRI nodes: u = longitude from -Z through +X, v = 0 at the
// zenith, +Y up.
fn bl_ll_dir(uv: vec2f) -> vec3f {
    let lon = (uv.x - 0.5) * 2.0 * PI;
    let lat = (0.5 - uv.y) * PI;
    return vec3f(cos(lat) * sin(lon), sin(lat), -cos(lat) * cos(lon));
}
fn bl_dir_ll(d: vec3f) -> vec2f {
    let n = normalize(d);
    // At the poles atan2(0, 0) is undefined (NaN on some GPUs): any longitude will do.
    let xz = select(vec2f(n.x, -n.z), vec2f(0.0, 1.0), abs(n.x) + abs(n.z) < 1e-7);
    return vec2f(atan2(xz.x, xz.y) / (2.0 * PI) + 0.5, 0.5 - asin(clamp(n.y, -1.0, 1.0)) / PI);
}
