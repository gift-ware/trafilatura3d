// Gaussian noise (soft signed blobs) warped twice by its own two decorrelated copies.
fn gauss(uv: vec2f, per: i32, s: f32) -> f32 {
    let dis = vec4f(p.disorder, 1.0, 0.0, 0.0);
    return blobs(uv, per, 2, 0.35, s, dis) + 0.5 * blobs(uv, per * 2, 2, 0.35, s + 11.0, dis);
}

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let c = vec2i(id.xy);
    if (c.x >= size().x || c.y >= size().y) { return; }
    var uv = expand_uv(pix_uv(c), p.non_square_expansion);
    let per = max(p.scale, 1);
    let k = p.warp_intensity * 5.0 / f32(per);
    for (var it = 0; it < 2; it++) {
        uv += k * vec2f(gauss(uv, per, p.seed_ + 50.0), gauss(uv, per, p.seed_ + 90.0));
    }
    let v = sat(0.5 + 0.5 * gauss(uv, per, p.seed_));
    store(c, vec4f(v, v, v, 1.0));
}
