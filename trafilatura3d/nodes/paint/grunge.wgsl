// Grunge: fbm in texture space, or projected from the three axes (Use Triplanar) with the position and world space
// normal maps, blended by the normal raised to the Triplanar Blending Contrast. Needs the inputs `position` and
// `world_space_normals`.
fn grunge(c: vec2i, scale: f32, seed: f32, triplanar: bool, contrast: f32) -> f32 {
    if (!triplanar || !POSITION_CONNECTED || !WORLD_SPACE_NORMALS_CONNECTED) {
        return fbm(vec3f(uv_of(c) * 8.0 * scale, 0.5), seed);
    }
    let pos = load_position(c).rgb * 8.0 * scale;
    let n = abs(load_world_space_normals(c).rgb * 2.0 - 1.0);
    var w = pow(n, vec3f(1.0 + contrast * 16.0));
    w /= max(w.x + w.y + w.z, 1e-6);
    return w.x * fbm(vec3f(pos.yz, 1.3), seed) + w.y * fbm(vec3f(pos.xz, 2.7), seed) + w.z * fbm(vec3f(pos.xy, 4.1), seed);
}
