// The normal the shader reads (Texture Set settings: Normal mixing, Height to normal method): the baked
// normal map, the Normal channel over it (Combine: detail oriented blending; Replace: the channel alone where
// it is painted) and the Height channel turned into normals on top. Inputs in the project's normal format,
// output in the format asked.

fn unit(v: vec3f) -> vec3f { let l = length(v); return select(vec3f(0.0, 0.0, 1.0), v / l, l > 1e-6); }
fn decode(v: vec3f, dx: bool) -> vec3f {
    var n = v * 2.0 - 1.0;
    if (dx) { n.y = -n.y; }
    return unit(n);
}
fn rnm(base: vec3f, detail: vec3f) -> vec3f {
    let n1 = base + vec3f(0.0, 0.0, 1.0);
    let n2 = detail * vec3f(-1.0, -1.0, 1.0);
    return unit(n1 * dot(n1, n2) / max(n1.z, 1e-4) - n2);
}
fn h(q: vec2i) -> f32 { return texel_height(q).r; }

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let c = vec2i(id.xy) + origin();
    if (any(c >= size())) { return; }
    let dx_in = p.dx_in != 0;
    var n = vec3f(0.0, 0.0, 1.0);
    if (BAKED_CONNECTED) { n = decode(load_baked(c).rgb, dx_in); }
    if (NORMAL_CONNECTED) {
        let ch = load_normal(c);
        let nc = decode(ch.rgb, dx_in);
        if (p.mixing == 1) {  // Combine
            n = unit(mix(n, rnm(n, nc), ch.a));
        } else {  // Replace (the baked normal where the channel is empty)
            n = unit(mix(n, nc, ch.a));
        }
    }
    if (HEIGHT_CONNECTED) {
        // Height slope per UV unit, scaled by Strength / 64 (ours). OpenGL: +Y up the image (rows go down).
        var gx: f32;
        var gy: f32;
        if (p.sobel != 0) {
            gx = (h(c + vec2i(1, -1)) + 2.0 * h(c + vec2i(1, 0)) + h(c + vec2i(1, 1))
                - h(c + vec2i(-1, -1)) - 2.0 * h(c + vec2i(-1, 0)) - h(c + vec2i(-1, 1))) / 8.0;
            gy = (h(c + vec2i(-1, 1)) + 2.0 * h(c + vec2i(0, 1)) + h(c + vec2i(1, 1))
                - h(c + vec2i(-1, -1)) - 2.0 * h(c + vec2i(0, -1)) - h(c + vec2i(1, -1))) / 8.0;
        } else {
            gx = (h(c + vec2i(1, 0)) - h(c + vec2i(-1, 0))) * 0.5;
            gy = (h(c + vec2i(0, 1)) - h(c + vec2i(0, -1))) * 0.5;
        }
        let k = p.strength / 64.0 * f32(size_height().x);
        let hn = unit(vec3f(-gx * k, gy * k, 1.0));
        n = rnm(n, hn);
    }
    if (p.dx_out != 0) { n.y = -n.y; }
    store(c, vec4f(n * 0.5 + 0.5, 1.0));
}
