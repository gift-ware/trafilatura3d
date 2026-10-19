@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let uv = uv_of(c, d);
    let mt = p.transformation;
    let mi = inv2(mt);
    let q = mi * (uv - 0.5 - p.offset) + 0.5;
    // Image space has y down; the normal's xy turn with the matrix (renormalized: rotation + mirror),
    // or with the whole matrix: the slopes of h(M^-1 uv) are M^-T times the input's.
    let mm = mat2x2f(mt.x, mt.y, mt.z, mt.w);
    let dx = p.normal_format == 0;
//@each
    {
        var v = sample_$ch(q);
        if ($K == 1) {
            var n = n_dec(v.rgb, dx);
            if (p.normal_affected_by_scale_and_skew_transformations != 0) {
                let t = transpose(mi) * vec2f(n.x, -n.y);
                n = normalize(vec3f(t.x, -t.y, n.z));
            } else {
                let t = mm * vec2f(n.x, -n.y);
                let l = length(n.xy);
                let txy = select(vec2f(0.0), normalize(t) * l, length(t) > 1e-6);
                n = vec3f(txy.x, -txy.y, n.z * sign(determinant(mm) + 1e-9));
                n.z = abs(n.z);
            }
            v = vec4f(n_enc(n, dx), 1.0);
        }
        if ($K == 2) { v = g4(v.r); }
        store_$ch(c, out_$ch(c, v));
    }
//@end
}
