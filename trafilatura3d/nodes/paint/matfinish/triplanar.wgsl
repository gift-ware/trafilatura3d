// MatFinish mapping with Triplanar Mapping: the pattern projected from the three axes on the position map
// (the mesh's bounding box mapped to 0..1, so a unit of pattern spans the box like it spans the UV square),
// blended by the world space normal; in texture space when the toggle is off or a map is missing. Needs the
// inputs `position` and `world_space_normals` and the parameter `triplanar_mapping`.
fn mf_triplanar() -> bool { return p.triplanar_mapping != 0 && POSITION_CONNECTED && WORLD_SPACE_NORMALS_CONNECTED; }
fn mf_sample(c: vec2i) -> vec4f {
    if (!mf_triplanar()) { return pattern(uv_of(c)); }
    let pos = load_position(c).rgb;
    let n = abs(load_world_space_normals(c).rgb * 2.0 - 1.0);
    var w = pow(n, vec3f(6.0));
    w /= max(w.x + w.y + w.z, 1e-6);
    // Each plane seen from its axis, with a different offset so the three projections are not alike.
    return w.x * pattern(pos.zy) + w.y * pattern(pos.xz + vec2f(0.37, 0.61)) + w.z * pattern(pos.xy + vec2f(0.71, 0.13));
}
// Across UV seams the position of a neighbour belongs to another part of the mesh.
fn mf_robust() -> bool { return mf_triplanar(); }
