// One layer over the stack below, for one channel. Values are what the channel holds (RGB; grayscale
// channels repeat the value), alpha is how much of the texture the stack covers so far. The mask and the
// opacity scale the layer's own alpha; the mode combines the layer's value with the value below
// (blending.py: identifiers in this order).

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let c = vec2i(id.xy) + origin();
    if (any(c >= size())) { return; }
    var m = 1.0;
    if (MASK_CONNECTED) { m = clamp(load_mask(c).r, 0.0, 1.0); }
    store(c, lay(load_below(c), load_top(c), m, p.mode, p.opacity, p.linear != 0, p.clamp01 != 0, p.normal_ != 0,
                 p.keep_alpha != 0));
}
