@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let uv = uv_of(c, d);
    let e = 1.0 / vec2f(d);
    let hc = sample_position(uv).r;
    let up = up_dir(hc, sample_position(uv + vec2f(e.x * 2.0, 0.0)).r, sample_position(uv + vec2f(0.0, e.y * 2.0)).r);
    let specks = sample_blurred(uv).r;
    // Drips: look uphill for rust that ran down to here, fading with distance.
    let n = p.drips_samples_amount;
    let len = p.drips_intensity * 0.25;
    var drip = 0.0;
    let side = vec2f(-up.y, up.x);
    let wob = (vnoise(uv, vec2i(64), p.seed_ + 9.0) - 0.5) * 0.004 * (1.0 + p.drips_smoothness * 4.0);
    for (var k = 1; k <= n; k++) {
        let t = f32(k) / f32(n);
        let s = sample_blurred(uv + up * t * len + side * wob).r;
        drip = max(drip, s * (1.0 - t) * mix(1.0, 0.6, p.drips_smoothness));
    }
    // Streaks thinner than the specks: modulate across the run direction.
    let streak = vnoise(vec2f(dot(uv, side), dot(uv, up) * 0.05), vec2i(96, 4), p.seed_ + 3.0);
    drip *= smoothstep(0.35 - p.drips_smoothness * 0.3, 0.75, streak);
    store(c, g4(max(specks, drip) * load_mask(c).r));
}
