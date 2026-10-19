// Seeds of the UV island borders: pixels inside an island (A of the UV islands map) next to a pixel outside or of
// another island (its RG differ) are seeds (xy); zw unused.
fn island(q: vec2i) -> vec3f {
    let t = texel_uv_islands(clamp(q, vec2i(0), size_uv_islands() - 1));
    return vec3f(t.rg, step(0.5, t.a));
}
fn effect(c: vec2i) -> vec4f {
    let q = vec2i(floor((vec2f(c) + 0.5) * vec2f(size_uv_islands()) / p.size_));
    let me = island(q);
    var border = false;
    var offsets = array<vec2i, 4>(vec2i(1, 0), vec2i(-1, 0), vec2i(0, 1), vec2i(0, -1));
    if (me.z > 0.0) {
        for (var k = 0; k < 4; k++) {
            let qq = q + offsets[k];
            let n = island(qq);
            let outside_image = any(qq < vec2i(0)) || any(qq >= size_uv_islands());
            if (outside_image || n.z < 0.5 || any(abs(n.xy - me.xy) > vec2f(1e-4))) { border = true; }
        }
    }
    return select(vec4f(-1.0), vec4f(vec2f(c), -1.0, -1.0), border);
}
