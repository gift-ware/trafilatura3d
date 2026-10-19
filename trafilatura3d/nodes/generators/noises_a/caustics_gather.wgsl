// Intensity of one channel at uv: photons of channel ch landing around it (flat surface = 1).
fn gather(uv: vec2f, ch: i32, asp: vec2f) -> f32 {
    let n = p.grid;
    let ior = channel_ior(p.surface_ior, p.dispersion, ch);
    let shift = landing_flat(p.light_position, ior, plane_depth(p.surface_height_position)) / asp;
    let centre = vec2i(floor((uv - shift) * f32(n)));
    let st = max(p.stride, 1);
    let cell = f32(st) / f32(n);
    let sigma = max(p.photon_size / 1024.0, cell);
    var acc = 0.0;
    for (var j = -p.steps; j <= p.steps; j++) {
        for (var i = -p.steps; i <= p.steps; i++) {
            // Strided photons stay on a fixed lattice so the result does not swim with the pixel.
            let q = (centre / st + vec2i(i, j)) * st;
            let w = wrap(q, vec2i(n));
            var dv = vec2f(0.0);
            if (ch == 0) { dv = textureLoad(src_photons_a, w, 0).xy; }
            else if (ch == 1) { dv = textureLoad(src_photons_a, w, 0).zw; }
            else { dv = textureLoad(src_photons_b, w, 0).xy; }
            var diff = photon_pos(q, n, p.jittering) + shift + dv - uv;
            diff -= round(diff);
            acc += exp(-dot(diff * asp, diff * asp) / (2.0 * sigma * sigma));
        }
    }
    return acc * cell * cell / (2.0 * PI * sigma * sigma);
}

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    var asp = vec2f(1.0);
    if (p.non_square_expansion != 0) { asp = p.size_ / min(p.size_.x, p.size_.y); }
    let uv = uv_of(c);
    // Flat surface: 0.4; focused light rises above it.
    let gain = 0.4;
    var o = vec3f(0.0);
    if (OUT_OUTPUT_GRAY) {
        o = vec3f(gather(uv, 1, asp) * gain);
    } else {
        let light = vec3f(gather(uv, 0, asp), gather(uv, 1, asp), gather(uv, 2, asp)) * gain;
        o = p.background_color.rgb + light * (1.0 - p.background_color.rgb);
    }
    if (p.output_color_space == 1) { o = to_srgb(o); }
    store(c, vec4f(o, 1.0));
}
