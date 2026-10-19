// The 3D Perlin spans about +-0.33 around the Baseline at Contrast 1.
const PERLIN_AMP = 0.65;
@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    var u = unit_slice(c);
    if (POSITION_MAP_CONNECTED) { u = load_position_map(c).rgb; }
    let tiled = p.enable_tiling != 0;
    let f = freq3(p.scale, p.size, tiled);
    let per = per3(f, tiled);
    var x = u * f + p.offset;
    x = distort3(x, u, f, tiled, p.distortion_intensity, p.distortion_scale_multiplier, p.seed_ + 50.0);
    let n = PERLIN_AMP * perlin3(x, per, p.seed_);
    var v = 0.5 + 0.5 * n;
    if (p.absolute != 0) { v = abs(n); }
    v = clamp(p.baseline + (v - 0.5) * contrast3(p.contrast), 0.0, 1.0);
    if (p.invert != 0) { v = 1.0 - v; }
    store(c, vec4f(v, v, v, 1.0));
}
