@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let n = clamp(p.samples_amount, 2, 8);
    var s: array<vec3f, 8>;
    var l: array<f32, 8>;
//@each
    s[$i] = srgb2lin(load_$ch(c).rgb);
//@end
    // Rank by luminance (insertion sort, at most 8).
    for (var i = 0; i < n; i++) { l[i] = luma(s[i]); }
    for (var i = 1; i < n; i++) {
        var j = i;
        while (j > 0 && l[j - 1] > l[j]) {
            let tl = l[j]; l[j] = l[j - 1]; l[j - 1] = tl;
            let ts = s[j]; s[j] = s[j - 1]; s[j - 1] = ts;
            j--;
        }
    }
    // Drop the darkest and brightest quarter (at least one each from 4 samples on).
    var lo = n / 4;
    var hi = n - n / 4;
    if (n == 2) { lo = 0; hi = 2; }
    if (n == 3) { lo = 1; hi = 2; }
    var acc = vec3f(0.0);
    for (var i = lo; i < hi; i++) { acc += s[i]; }
    store(c, vec4f(lin2srgb(acc / f32(hi - lo)), 1.0));
}
