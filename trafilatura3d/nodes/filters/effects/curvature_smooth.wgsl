// Normal XY (+Y up) of mip level j (the image box-averaged over 2^j x 2^j pixels), bilinear.
fn nxy(j: i32, uv: vec2f) -> vec2f {
    var t: vec4f;
    switch j {
        case 0: { t = sample_normal(uv); }
        case 1: { t = sample_m1(uv); }
        case 2: { t = sample_m2(uv); }
        case 3: { t = sample_m3(uv); }
        case 4: { t = sample_m4(uv); }
        case 5: { t = sample_m5(uv); }
        case 6: { t = sample_m6(uv); }
        case 7: { t = sample_m7(uv); }
        case 8: { t = sample_m8(uv); }
        case 9: { t = sample_m9(uv); }
        case 10: { t = sample_m10(uv); }
        default: { t = sample_m11(uv); }
    }
    return fx_decode_normal(t, p.normal_format == 0).xy;
}

// Divergence of the normals' XY at level j, per texel of the level: four bilinear samples on the diagonals half a
// texel away (each the mean of a 2x2 block), the differences of their sides (+Y is up, rows grow downwards).
fn div_at(j: i32, uv: vec2f) -> f32 {
    let h = 0.5 * exp2(f32(j)) / p.size_;  // half a texel of the level
    let rd = nxy(j, uv + vec2f(h.x, h.y));
    let ru = nxy(j, uv + vec2f(h.x, -h.y));
    let ld = nxy(j, uv + vec2f(-h.x, h.y));
    let lu = nxy(j, uv + vec2f(-h.x, -h.y));
    return 0.5 * ((rd.x + ru.x) - (ld.x + lu.x) + (ru.y + lu.y) - (rd.y + ld.y));
}

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let uv = fx_uv(c);
    var acc = 0.0;
    let reach = p.log2_size * p.radius;  // Radius: how many octaves count (1 = all, 0 = only the finest)
    for (var j = 0; j < p.levels; j++) {
        acc += div_at(j, uv) * p.gain * clamp(reach - f32(j), 0.0, 1.0);
    }
    let curv = clamp(0.5 + acc, 0.0, 1.0);
    store_curvature(c, fx_gray(curv));
    store_convexity(c, fx_gray(clamp((curv - 0.5) * 2.0, 0.0, 1.0)));
    store_concavity(c, fx_gray(clamp((0.5 - curv) * 2.0, 0.0, 1.0)));
}
