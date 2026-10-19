// The copy of each stage (a reflect_uv kind), -1 = no stage.
fn stage(i: i32) -> i32 {
    var s = vec3i(-1);
    switch p.symmetry_mode {
        case 0: { s = vec3i(1, -1, -1); }
        case 1: { s = vec3i(0, -1, -1); }
        case 2: { s = vec3i(3, -1, -1); }
        case 3: { s = vec3i(2, -1, -1); }
        case 4: { s = vec3i(4, -1, -1); }
        case 5: { s = vec3i(1, 0, -1); }
        case 6: { s = vec3i(3, 2, -1); }
        case 7: { s = vec3i(4, 5, -1); }
        default: { s = vec3i(3, 2, 6); }
    }
    return s[i];
}

fn v0(q: vec2f) -> vec4f { return texel_input(vec2i(floor(q * vec2f(size_input())))); }

fn v1(q: vec2f) -> vec4f { return transfer(p.transfer_mode, v0(q), v0(reflect_uv(stage(0), q))); }
fn v2(q: vec2f) -> vec4f {
    let k = stage(1);
    if (k < 0) { return v1(q); }
    return transfer(p.transfer_mode, v1(q), v1(reflect_uv(k, q)));
}
fn v3(q: vec2f) -> vec4f {
    let k = stage(2);
    if (k < 0) { return v2(q); }
    return transfer(p.transfer_mode, v2(q), v2(reflect_uv(k, q)));
}

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    var o = v3(uv_at(c));
    if (INPUT_GRAY) { o = gray4(o.r); }
    store(c, o);
}
