@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let uv = fx_uv(c);
    let m = p.matrix2x2;
    let sampling = mat2x2f(m.x, m.y, m.z, m.w);
    // Transformation 2D's convention: input point = M * (output point - centre) + centre + offset.
    let q = sampling * (uv - 0.5) + 0.5 + p.offset;
    let dx = p.normal_format == 0;
    let t = sample_input(q);
    let n = fx_decode_normal(t, dx);
    // On XY in image space (y down): the vectors follow the content, by the transpose
    // of the sampling matrix with Handle Scale and Skew, else only by its rotation (the angle of its
    // first row) and its mirroring.
    var g = vec2f(n.x, -n.y);
    if (p.handle_scale_and_skew_transformation == 1) {
        g = transpose(sampling) * g;
    } else {
        if (determinant(sampling) < 0.0) { g.y = -g.y; }
        let a = atan2(m.z, m.x);
        g = mat2x2f(cos(a), sin(a), -sin(a), cos(a)) * g;
    }
    let v = vec3f(g.x, -g.y, n.z);
    let l = length(v);
    store(c, vec4f(fx_encode_normal(select(vec3f(0.0, 0.0, 1.0), v / l, l > 1e-6), dx), t.a));
}
