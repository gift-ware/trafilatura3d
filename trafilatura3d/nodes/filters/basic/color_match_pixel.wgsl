// Shifted colour and the selection (pixels near the source in every HCY axis) of Color Match.
@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let si = load_source_color(c).rgb;
    let ti = load_target_color(c).rgb;
    var src = texel_mean(vec2i(0)).rgb;
    if (p.source_color_mode == 1) { src = p.source_color.rgb; }
    if (p.source_color_mode == 2) { src = si; }
    let tgt = select(p.target_color.rgb, ti, p.target_color_mode == 1);
    let x = rgb_to_hcl(load_input(c).rgb);
    let s = rgb_to_hcl(src);
    let t = rgb_to_hcl(tgt);
    // The pixel keeps its offset from the source, scaled per axis, around the target (default: the target's hue,
    // the pixel's own chroma and luma spread).
    let k = select(vec3f(0.0, 1.0, 1.0), vec3f(p.variation_hue / 360.0, p.variation_chroma, p.variation_luma),
                   p.custom_color_variation != 0);
    let o = t + (x - s) * k;
    store_color(c, vec4f(hcl_to_rgb(vec3f(fract(o.x), max(o.yz, vec2f(0.0)))), 1.0));
    // Selected: within half of each range from the source (hue measured the short way round the circle).
    let dh = abs(x.x - s.x);
    let off = vec3f(min(dh, 1.0 - dh), abs(x.yz - s.yz));
    let half_range = 0.5 * vec3f(p.mask_hue_range / 360.0, p.mask_chroma_range, p.mask_luma_range);
    store_selection(c, gray4(select(0.0, 1.0, all(off <= half_range))));
}
