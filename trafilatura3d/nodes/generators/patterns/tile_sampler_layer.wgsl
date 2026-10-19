// The Tile Sampler's layer: final_ 0 writes the background (Background Input, else Background Color) into the
// layer the tiles are drawn over (tile_sampler_draw.wgsl); 1 writes the layer, clamped to 0-1, as the output.

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let c = vec2i(id.xy) + origin();
    if (any(c >= size())) { return; }
    if (p.final_ == 0) {
        var bg = p.background_color;
        if (BACKGROUND_INPUT_CONNECTED) { bg = load_background_input(c); }
        store(c, bg);
        return;
    }
    let r = clamp(load_layer(c), vec4f(0.0), vec4f(1.0));
    if (OUT_OUTPUT_GRAY) { store(c, vec4f(r.r, r.r, r.r, 1.0)); } else { store(c, r); }
}
