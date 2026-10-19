// The stamped noises' last pass: their layers (each the stamps' sum around 0.5) combined: a layer adds weight x
// (value - 0.5), or folded, weight x (|value - 0.5| / sigma) ^ fold (dark creases where it crosses 0.5); the pixel
// level (one Square stamp per cell of `grain_px` pixels, its luminosity random: `grain` x a uniform value in
// -0.5..0.5, hashed here rather than stamped); a vertical ramp (`ramp` x (y - 0.5), y down); then offset + gain x the sum, in 0-1.
@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let c = vec2i(id.xy);
    if (c.x >= size().x || c.y >= size().y) { return; }
    let x = vec3f(load_layer0(c).r, load_layer1(c).r, load_layer2(c).r) - 0.5;
    let w = vec3f(p.w0, p.w1, p.w2);
    let fold = vec3f(p.f0, p.f1, p.f2);
    let sigma = max(vec3f(p.s0, p.s1, p.s2), vec3f(1e-6));
    let folded = pow(abs(x) / sigma, max(fold, vec3f(1e-6)));
    let t = select(x, folded, fold > vec3f(0.0)) * w;
    let gp = max(i32(p.grain_px), 1);
    let grain = p.grain * (hash21(c / gp, p.seed_ + 7.0) - 0.5);
    let y = (f32(c.y) + 0.5) / f32(size().y);
    let v = clamp(p.offset + p.gain * (t.x + t.y + t.z + grain + p.ramp * (y - 0.5)), 0.0, 1.0);
    store(c, vec4f(v, v, v, 1.0));
}
