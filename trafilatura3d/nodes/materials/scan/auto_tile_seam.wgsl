fn a_uv(uv: vec2f) -> vec2f { return crop_uv(uv); }
fn b_uv(uv: vec2f) -> vec2f { return crop_uv(fract(uv + 0.5)); }

// Weighted difference between the two copies at a point (per-channel thresholds).
fn diff(uv: vec2f) -> f32 {
    var t = vec3f(p.threshold);
    if (p.use_threshold_per_channel != 0) { t = vec3f(p.threshold_base_color, p.threshold_normal, p.threshold_height); }
    var e = 0.0;
    if (p.use_base_color != 0) { e += t.x * distance(sample_basecolor(a_uv(uv)).rgb, sample_basecolor(b_uv(uv)).rgb); }
    if (p.use_normal != 0) { e += t.y * distance(sample_normal(a_uv(uv)).rg, sample_normal(b_uv(uv)).rg); }
    if (p.use_height != 0) { e += t.z * abs(sample_height(a_uv(uv)).r - sample_height(b_uv(uv)).r) * 2.0; }
    return e;
}

fn edges(uv: vec2f, k: f32) -> f32 {
    let e = vec2f(k, 0.0);
    let hx = sample_height(a_uv(uv + e.xy)).r - sample_height(a_uv(uv - e.xy)).r;
    let hy = sample_height(a_uv(uv + e.yx)).r - sample_height(a_uv(uv - e.yx)).r;
    let cx = luma(sample_basecolor(a_uv(uv + e.xy)).rgb - sample_basecolor(a_uv(uv - e.xy)).rgb);
    return sat(length(vec2f(hx, hy)) * 6.0 + abs(cx) * 4.0);
}

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    if (p.use_mask != 0) { store(c, g4(load_mask(c).r)); return; }
    let n = f32(1 << u32(clamp(p.grid_resolution, 1, 11)));
    let uv = (floor(uv_of(c, d) * n) + 0.5) / n;
    // Distance to the tile border per axis; the cut sits at Cut Offset, moved by the analysis.
    let bd = min(uv, 1.0 - uv);
    let s = 1.0 + p.smoothness;
    var k = diff(uv) * 0.25 * s;
    if (p.detect_edges != 0) { k -= edges(uv, 1.5 / n) * 0.12; }
    let cut = max(p.cut_offset, vec2f(0.02)) * (1.0 - clamp(k, -0.6, 0.6));
    let w = select(0.0, 1.0, bd.x < cut.x || bd.y < cut.y);
    // The tile corners and edges themselves always take the shifted copy.
    let core = select(0.0, 1.0, min(bd.x, bd.y) < 0.02);
    store(c, g4(max(w, core)));
}
