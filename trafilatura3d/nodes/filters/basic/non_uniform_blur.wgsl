// One pass: the mean of the pixel and Blades directional warps of it, the warps on an ellipse of radius
// p.radius (pixels at 256) turned by p.inner, each scaled by the Blur Map.
@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let uv = uv_at(c);
    let blades = clamp(p.blades, 1, 9);
    let amount = load_blur_map(c).r * p.radius / 256.0;
    let fold = clamp(p.asymmetry * 0.5, 0.0, 0.5);
    var acc = sample_input(uv);
    for (var j = 1; j <= blades; j++) {
        let t = 2.0 * PI * (p.inner + f32(j) / f32(blades));
        // Ellipse: Anisotropy squeezes it across Angle, Asymmetry folds its back half forward.
        let x = mix(cos(t), abs(cos(t)), fold);
        let e = rot(vec2f(x, (1.0 - p.anisotropy) * sin(t)), p.angle);
        acc += sample_input(uv + e * amount);
    }
    store(c, acc / f32(blades + 1));
}
