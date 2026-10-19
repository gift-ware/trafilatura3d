// Emboss With Gloss: the part of the light that faces it (0-1), smoothed next for the gloss bands.
@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    store_out(c, fx_gray(clamp(ewg_slope(c), 0.0, 1.0)));
}
