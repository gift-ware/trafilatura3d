fn deviation(uv: vec2f, asp: vec2f, ior: f32) -> vec2f {
    // Slope over +-2 texels, averaged across: 8-bit height maps are terraced, and a 1-texel slope
    // turns the terraces into rings in the caustics.
    let px = 2.0 / vec2f(size_height_map());
    let k = clamp(p.surface_height_scale, 0.0, 1.0) * HEIGHT_MAX;
    var gx = 0.0;
    var gy = 0.0;
    for (var o = -1; o <= 1; o++) {
        let a = f32(o) * 0.5;
        gx += sample_height_map(uv + vec2f(px.x, a * px.y)).r - sample_height_map(uv + vec2f(-px.x, a * px.y)).r;
        gy += sample_height_map(uv + vec2f(a * px.x, px.y)).r - sample_height_map(uv + vec2f(a * px.x, -px.y)).r;
    }
    // In square units (the image spans `asp`).
    gx /= 3.0 * 2.0 * px.x * asp.x;
    gy /= 3.0 * 2.0 * px.y * asp.y;
    let nrm = normalize(vec3f(-gx * k, -gy * k, 1.0));
    let depth = plane_depth(p.surface_height_position);
    let i = incident(p.light_position);
    var dv = landing(i, nrm, ior, depth) - landing_flat(p.light_position, ior, depth);
    let l = length(dv);
    if (l > DEV_MAX) { dv *= DEV_MAX / l; }
    return dv / asp;
}

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    var asp = vec2f(1.0);
    if (p.non_square_expansion != 0) { asp = vec2f(size_height_map()) / f32(min(size_height_map().x, size_height_map().y)); }
    let uv = photon_pos(c, d.x, p.jittering);
    let ior = p.surface_ior;
    if (p.pair == 0) {
        store_photons(c, vec4f(deviation(uv, asp, channel_ior(ior, p.dispersion, 0)),
                               deviation(uv, asp, channel_ior(ior, p.dispersion, 1))));
    } else {
        store_photons(c, vec4f(deviation(uv, asp, channel_ior(ior, p.dispersion, 2)), 0.0, 0.0));
    }
}
