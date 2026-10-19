// Final pass (format and balance were applied while integrating): the relief scaled by Global Opacity,
// The default 0.36 = as integrated (the level opacities are 100 x Global Opacity x ...).
@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    store(c, fx_gray(mix(0.5, texel_relief(c).r, max(p.global_opacity, 0.0) / 0.36)));
}
