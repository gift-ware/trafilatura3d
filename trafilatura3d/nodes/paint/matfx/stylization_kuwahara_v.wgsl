// The quadrants from 4 rows of the horizontal pass (above: towards -y, below: towards +y): the least varied
// one's mean, mixed by the global Smoothness, into the channel's output.
fn fx(c: vec2i) {
    let uv = mx_uv(c);
    let center = fetch(uv);
    var res = center.rgb;
    if (p.radius_ > 0.0 && p.smoothness > 0.0) {
        let d = grid_step().y;
        // quadrants in order: (lo, above), (hi, above), (lo, below), (hi, below); the center row is shared
        var quads = array<vec4f, 4>(sample_lo(uv), sample_hi(uv), vec4f(0.0), vec4f(0.0));
        quads[2] = quads[0];
        quads[3] = quads[1];
        for (var j = 1; j < 4; j++) {
            let above = uv - vec2f(0.0, f32(j) * d);
            let below = uv + vec2f(0.0, f32(j) * d);
            quads[0] += sample_lo(above);
            quads[1] += sample_hi(above);
            quads[2] += sample_lo(below);
            quads[3] += sample_hi(below);
        }
        var best = center.rgb;
        var best_var = 1e9;
        for (var qd = 0; qd < 4; qd++) {
            let q = quads[qd] / 4.0;
            let s = q.w - dot(q.rgb, q.rgb);
            if (s < best_var) { best_var = s; best = q.rgb; }
        }
        res = mix(center.rgb, best, p.smoothness);
        if (p.channel_ == 3) { res = mx_nenc(mx_ndec(res)); }
    }
    let out = vec4f(res, center.a);
    switch p.channel_ {
        case 0: { store_k_basecolor(c, out); }
        case 1: { store_k_roughness(c, out); }
        case 2: { store_k_metallic(c, out); }
        case 3: { store_k_normal(c, out); }
        case 4: { store_k_height(c, out); }
        default: { store_k_ambient_occlusion(c, out); }
    }
}
