fn inv3(m: mat3x3f) -> mat3x3f {
    let a = m[0]; let b = m[1]; let cc = m[2];
    let r0 = cross(b, cc); let r1 = cross(cc, a); let r2 = cross(a, b);
    let det = dot(a, r0);
    if (abs(det) < 1e-8) { return mat3x3f(vec3f(0.0), vec3f(0.0), vec3f(0.0)); }
    return transpose(mat3x3f(r0, r1, r2)) * (1.0 / det);
}

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let n = clamp(p.samples_amount, 2, 8);
    let dirsign = select(1.0, -1.0, p.next_sample_light_angle == 1);
    let el = 0.70710678;
    var ata = mat3x3f(vec3f(0.0), vec3f(0.0), vec3f(0.0));
    var atb = vec3f(0.0);
    var v: f32;
//@each
    v = luma(srgb2lin(load_$ch(c).rgb));
    if ($i < n) {
        let a = (p.first_sample_light_angle + dirsign * f32($i) * 360.0 / f32(n)) * PI / 180.0;
        let l = vec3f(cos(a) * el, sin(a) * el, el);
        ata += mat3x3f(l * l.x, l * l.y, l * l.z);
        atb += l * v;
    }
//@end
    // Two samples leave z undetermined: regularize it towards the mean brightness.
    ata += mat3x3f(vec3f(0.0), vec3f(0.0), vec3f(0.0, 0.0, 0.05));
    var g = inv3(ata) * atb;
    if (length(g) < 1e-6) { g = vec3f(0.0, 0.0, 1.0); }
    var nn = normalize(g);
    nn = normalize(vec3f(nn.xy * p.intensity * 2.0, max(nn.z, 0.05)));
    store(c, vec4f(n_enc(nn, p.normal_format == 0), 1.0));
}
