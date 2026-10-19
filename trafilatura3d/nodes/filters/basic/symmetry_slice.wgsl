fn texel_at(q: vec2f) -> vec4f { return texel_input(vec2i(floor(q * vec2f(size_input())))); }

// The first stage's mask at w (centred UV): the signed distance from the mirror line (positive on the side
// that takes the copy; Flip Side swaps them), divided by the farthest distance in the image and halved (-0.5 ..
// 0.5), through a smoothstep Blend wide (0: a hard cut, 1: across the whole image).
fn mask_a(w0: vec2f) -> f32 {
    let w = select(w0, -w0, p.flip_side != 0);
    var x = w.x;                                           // Vertical: the right side
    switch p.symmetry_mode {
        case 0, 5: { x = w.y; }                            // Horizontal: the bottom
        case 2, 6: { x = 0.5 * (w.x + w.y); }              // across the TR-BL diagonal: the lower right
        case 3: { x = 0.5 * (w.x - w.y); }                 // across the TL-BR diagonal: the upper right
        case 4: { x = -w.x; }                              // Vertical Invert: the left side
        default: {}
    }
    let half = 0.5 * clamp(p.blend, 0.0, 1.0);
    if (half < 1e-5) { return select(0.0, 1.0, x > 0.0); }
    return smoothstep(-half, half, x);
}

// The second stage's: the first one's turned a quarter (Corner modes) and by Flip Side2's half turn.
fn mask_b(w0: vec2f) -> f32 {
    let w = select(w0, -w0, p.flip_side2 != 0);
    switch p.symmetry_mode {
        case 5: { return mask_a(vec2f(w.y, -w.x)); }
        case 6: { return mask_a(vec2f(-w.y, w.x)); }
        default: { return mask_a(w); }
    }
}

fn first_copy() -> i32 {
    switch p.symmetry_mode {
        case 0, 5: { return 1; }
        case 1: { return 0; }
        case 2, 6: { return 3; }
        case 3: { return 2; }
        default: { return 4; }
    }
}

fn s1(q: vec2f) -> vec4f {
    let a = texel_at(q);
    return mix(a, transfer(p.transfer_mode, a, texel_at(reflect_uv(first_copy(), q))), mask_a(q - 0.5));
}

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let q = uv_at(c);
    var o = s1(q);
    // The corner modes mirror the first stage's result once more, across the other line.
    if (p.symmetry_mode == 5 || p.symmetry_mode == 6) {
        let r = reflect_uv(select(2, 0, p.symmetry_mode == 5), q);
        o = mix(o, transfer(p.transfer_mode, o, s1(r)), mask_b(q - 0.5));
    }
    if (INPUT_GRAY) { o = gray4(o.r); }
    store(c, o);
}
