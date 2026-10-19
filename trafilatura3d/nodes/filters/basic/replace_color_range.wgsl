// Replace Color Range, ours: the colours near Source Color (Euclidean RGB distance, a smooth falloff to 0 at the
// radius) take the HSL offset from Source to Target Color. The radius is RADIUS x sqrt(Threshold), times
// (2 Source Range)^3: Source Range 0 replaces nothing, 0.5 the radius as is, 1 eight times it.
const RADIUS = 1.1;  // fitted on the output

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let v = load_input(c);
    let src = p.source_color.rgb;
    let r = RADIUS * sqrt(max(p.threshold, 0.0)) * pow(2.0 * clamp(p.source_range, 0.0, 1.0), 3.0);
    var w = 0.0;
    if (r > 1e-5) { w = 1.0 - smoothstep(0.0, r, length(v.rgb - src)); }
    store(c, vec4f(mix(v.rgb, hsl_shift(v.rgb, rgb_to_hsl(p.target_color.rgb) - rgb_to_hsl(src)), w), v.a));
}
