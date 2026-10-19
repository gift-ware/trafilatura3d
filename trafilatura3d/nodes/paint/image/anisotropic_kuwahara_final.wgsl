// Anisotropic Kuwahara with polynomial sector weights (Kyprianidis, Semmo, Kang, Doellner 2010).
fn effect(c: vec2i) -> vec4f {
    let center = load_input(c);
    let T = load_tensor(c);
    let disc = sqrt((T.x - T.z) * (T.x - T.z) + 4.0 * T.y * T.y);
    let l1 = 0.5 * (T.x + T.z + disc);
    let l2 = 0.5 * (T.x + T.z - disc);
    var t = vec2f(l1 - T.x, -T.y);  // the edge's direction (minor eigenvector)
    t = select(vec2f(0.0, 1.0), normalize(t), length(t) > 1e-12);
    let A = select(0.0, (l1 - l2) / (l1 + l2), l1 + l2 > 1e-12);
    let dir = rot2(t, p.anisotropy_angle);
    let r = max(p.radius * px256() * load_radius_map(c).r, 0.0);
    if (r < 0.5) { return center; }
    let stretch = 1.0 + p.anisotropy * A;
    let ax = r * min(stretch, 4.0);
    let ay = r / min(stretch, 4.0);
    // Sector shape: overlap of the 8 sectors and the weight of the centre.
    let zeta = mix(0.05, 0.35, p.smoothness);
    let zc = mix(0.25, 0.5, p.smoothness) * PI;
    let eta = (zeta + cos(zc)) / (sin(zc) * sin(zc));
    let q = 1.0 + 17.0 * p.sharpness;
    var m: array<vec4f, 8>;
    var s: array<vec3f, 8>;
    for (var k = 0; k < 8; k++) { m[k] = vec4f(0.0); s[k] = vec3f(0.0); }
    let n = clamp(i32(ceil(ax)), 2, 10);
    let uv = uv_of(c);
    for (var j = -n; j <= n; j++) {
        for (var i = -n; i <= n; i++) {
            var v = vec2f(f32(i), f32(j)) / f32(n);  // unit disk, ellipse frame
            if (dot(v, v) > 1.0) { continue; }
            let off = dir * v.x * ax + vec2f(-dir.y, dir.x) * v.y * ay;
            let col = sample_input(uv + off / p.size_).rgb;
            var w: array<f32, 8>;
            var vxx = zeta - eta * v.x * v.x;
            var vyy = zeta - eta * v.y * v.y;
            var z = max(0.0, v.y + vxx); w[0] = z * z;
            z = max(0.0, -v.x + vyy); w[2] = z * z;
            z = max(0.0, -v.y + vxx); w[4] = z * z;
            z = max(0.0, v.x + vyy); w[6] = z * z;
            let u = 0.70710678 * vec2f(v.x - v.y, v.x + v.y);
            vxx = zeta - eta * u.x * u.x;
            vyy = zeta - eta * u.y * u.y;
            z = max(0.0, u.y + vxx); w[1] = z * z;
            z = max(0.0, -u.x + vyy); w[3] = z * z;
            z = max(0.0, -u.y + vxx); w[5] = z * z;
            z = max(0.0, u.x + vyy); w[7] = z * z;
            var sum = 0.0;
            for (var k = 0; k < 8; k++) { sum += w[k]; }
            let g = exp(-3.125 * dot(v, v)) / max(sum, 1e-8);
            for (var k = 0; k < 8; k++) {
                let wk = w[k] * g;
                m[k] += vec4f(col * wk, wk);
                s[k] += col * col * wk;
            }
        }
    }
    var acc = vec3f(0.0);
    var wsum = 0.0;
    for (var k = 0; k < 8; k++) {
        if (m[k].w <= 1e-8) { continue; }
        let mean = m[k].rgb / m[k].w;
        let var3 = abs(s[k] / m[k].w - mean * mean);
        let a = 1.0 / (1.0 + pow(max(255.0 * (var3.r + var3.g + var3.b), 1e-8), 0.5 * q));
        acc += mean * a;
        wsum += a;
    }
    if (wsum <= 1e-20) { return center; }
    return vec4f(acc / wsum, center.a);
}
