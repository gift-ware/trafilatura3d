// Wrapped (tiled) input UV, flipped by Symmetry.
fn fetch(q: vec2f) -> vec2f {
    var r = fract(q);
    if (p.symmetry == 1 || p.symmetry == 3) { r.x = 1.0 - r.x; }
    if (p.symmetry == 2 || p.symmetry == 3) { r.y = 1.0 - r.y; }
    return r;
}

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let uv = uv_at(c);
    let t = f32(max(p.tile, 1));
    // Offset in input UV, snapped down to whole pixels; an even Tile adds half an image so that the
    // copies' corners land on the image's corners.
    var off = p.offset;
    if (p.offset_mode == 1) { off = hash22(vec2i(17, 31), p.seed_); }
    off = floor(off * p.size_) / p.size_ + f32((max(p.tile, 1) + 1) % 2) * 0.5;
    let s = square_scale();
    let snapped = p.tile_safe_rotation != 0;
    // Tile Safe Rotation: the rotation goes down to a multiple of 45 degrees, and a diagonal one tiles
    // sqrt(2) times more, so that Tile x (cos, sin) is a whole vector.
    var turns = p.rotation;
    var k = 1.0;
    if (snapped) {
        turns = floor(8.0 * p.rotation) / 8.0;
        let m = abs(turns) % 0.25 * 2.0 * PI;
        k = cos(m) + sin(m);
    }
    let ang = -turns * 2.0 * PI;
    var ab = t * k * vec2f(cos(ang), sin(ang));
    if (snapped) { ab = round(ab); }
    let v = (uv - 0.5) * s;
    let q = vec2f(ab.x * v.x - ab.y * v.y, ab.y * v.x + ab.x * v.y) / s + 0.5 - off;
    // One output pixel covers |ab| / size_out of the input image.
    let si = vec2f(size_input());
    var level = 0.0;
    if (p.mipmap_mode == 1) { level = f32(p.mipmap_level); }
    else { level = max(log2(length(ab) * max(si.x / p.size_.x, si.y / p.size_.y)), 0.0); }
    let axis = ab.y == 0.0 && fract(ab.x) == 0.0;
    if (!snapped && !axis && off_image((q - 0.5) / t + 0.5)) { store(c, background(p.background_color, INPUT_GRAY)); return; }
    if (level < 0.01 && axis && p.size_.x == si.x && p.size_.y == si.y) {
        store(c, texel_input(vec2i(floor(fetch(q) * si))));   // pixel exact
        return;
    }
    let taps = clamp(i32(exp2(ceil(level))), 1, 8);
    let step = exp2(level) / si / f32(taps);
    var acc = vec4f(0.0);
    for (var j = 0; j < taps; j++) {
        for (var i = 0; i < taps; i++) {
            let o = (vec2f(f32(i), f32(j)) - f32(taps - 1) * 0.5) * step;
            acc += sample_input(fetch(q) + o);
        }
    }
    store(c, acc / f32(taps * taps));
}
