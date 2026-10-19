@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let s = vec2f(size_input());
    let lo = texel_lo(vec2i(0)).xy;
    let hi = texel_hi(vec2i(0)).xy + 1.0;   // exclusive, in input pixels
    let blank = select(vec4f(0.0), gray4(0.0), INPUT_GRAY);
    if (lo.x >= hi.x) { store(c, blank); return; }   // no content
    let ext = hi - lo;
    let uv = uv_at(c);
    var q: vec2f;   // input pixel coordinates
    var keep = vec2f(0.0);  // half-size of the area kept around the centre (crop modes), in pixels
    if (p.mode <= 1) {
        let shift = floor((s - (lo + hi)) * 0.5);
        q = uv * s - shift;
        keep = select(vec2f(max(ext.x, ext.y)), ext, p.mode == 1) * 0.5;
        if (any(abs(uv * s - s * 0.5) > keep + 0.5)) { store(c, blank); return; }
    } else if (p.mode == 2) {
        let k = min(s.x / ext.x, s.y / ext.y);
        q = (lo + hi) * 0.5 + (uv * s - s * 0.5) / k;
    } else {
        q = lo + uv * ext;
    }
    if (any(q < lo) || any(q > hi)) { store(c, blank); return; }
    var nearest = p.filtering_mode == 0;
    if (p.filtering_mode == 2) { nearest = p.mode <= 1; }
    if (nearest) { store(c, texel_input(vec2i(floor(q)))); }
    else { store(c, sample_input(q / s)); }
}
