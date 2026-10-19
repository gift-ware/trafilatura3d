fn ray_height(uv: vec2f) -> f32 { return sample_input(uv).r * p.height_scale; }

fn to_srgb(x: f32) -> f32 { return select(1.055 * pow(x, 1.0 / 2.4) - 0.055, x * 12.92, x <= 0.0031308); }

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let uv = fx_uv(c);
    // The light direction (not normalized): towards the light position from the centre, its
    // height falling from 1 at the centre to the horizon 0.5 away.
    let lp = floor(p.light_position * p.size_) / p.size_;
    let soft = p.light_size * p.light_scale_soft_shadows;
    let floor_z = select(0.01, max(soft.y, 0.01), p.keep_light_above_horizon == 1);
    let l = vec3f(lp.x - 0.5, 0.5 - lp.y, max(1.0 - 2.0 * distance(lp, vec2f(0.5)), floor_z));
    let ax = cross(vec3f(0.0, 0.0, 1.0), l);
    let ay = cross(ax, l);
    let count = select(clamp(p.samples, 1, 4096), 1, p.light_scale_soft_shadows == 0.0);
    let n = ray_normal(uv);
    let r = hash22(c, p.seed_ + 9.0);
    let maxlen = max(p.max_shadows_length, 1e-4);
    var acc = 0.0;
    for (var i = 0; i < count; i++) {
        let f = (fx_fib(i, count, r) - 0.5) * 2.0 * soft;
        let dir = l + ax * f.x + ay * f.y;
        let t = ray_march(uv, normalize(dir), maxlen, 32, fract(r.y + f32(i) * 0.618));
        let s = select(0.0, 1.0, t >= 0.0);
        let k = max(t, 0.0) / maxlen;  // distance to the caster, share of the length
        if (p.mode == 0) {
            let shadow = mix(s, s * (1.0 - k), p.shadows_attenuation) * p.shadows_opacity;
            acc += to_srgb(max(p.light_intensity * dot(dir, n), 0.0) * (1.0 - shadow));
        } else {
            var lit = 1.0 - s;
            if (p.mode == 2) { lit = 1.0 - s * (1.0 - k); }
            if (p.mode == 3) { lit = 1.0 - s * k; }
            acc += select(lit, 1.0 - lit, p.invert == 1);
        }
    }
    store(c, fx_gray(acc / f32(count)));
}
