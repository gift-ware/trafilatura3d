// Position inside a pair of lines split by Middle Size s (0-1): (line index, coordinate in the line, line size).
fn bg_split(g: f32, ms: f32) -> vec3f {
    let s = 2.0 * clamp(ms, 0.05, 0.95);
    let pair = floor(g * 0.5);
    let f = g - 2.0 * pair;
    let second = f >= s;
    return vec3f(2.0 * pair + select(0.0, 1.0, second), select(f, f - s, second), select(s, 2.0 - s, second));
}

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let uv = frame(c, p.non_square_expansion);
    let cells = vec2f(max(p.bricks, vec2i(1)));
    let g = uv * cells;
    // Rows go in pairs, Middle Size y splits each pair; columns the same with Middle Size x.
    let ry = bg_split(g.y, p.middle_size.y);
    let row = ry.x;
    let rshift = p.offset.x * imod2(row) + p.offset.y * rnd(vec2i(0, i32(row)), 1103u);
    let rx = bg_split(g.x + rshift, p.middle_size.x);
    let wrap = vec2i(round(cells * frame_size(p.non_square_expansion)));
    let brick = vec2i(wrapi(i32(rx.x), wrap.x), wrapi(i32(row), wrap.y));
    // Distances to the joints, uv; slope widths per axis.
    let joint = p.gap * 0.5 / cells.yx;
    let dx = min(rx.y, rx.z - rx.y) / cells.x - joint.x;
    let dy = min(ry.y, ry.z - ry.y) / cells.y - joint.y;
    let bw = p.bevel.xy * 0.5 * vec2f(select(rx.z / cells.x, ry.z / cells.y, p.keep_ratio == 1), ry.z / cells.y);
    let px = pixel_uv();
    var t = vec2f(clamp(dx / max(bw.x, px), 0.0, 1.0), clamp(dy / max(bw.y, px), 0.0, 1.0));
    t = mix(t, t * t * (3.0 - 2.0 * t), clamp(p.bevel.z, 0.0, 1.0));                  // falloff
    var v = mix(min(t.x, t.y), t.x * t.y, clamp(p.bevel.w, 0.0, 1.0));                // corner rounding
    // Height: one of two levels per brick, then the luminance variation.
    let hi = max(max(p.height.x, p.height.y), 0.0);
    let lo = max(min(p.height.x, p.height.y), 0.0);
    var h = select(hi, lo, rnd(brick, 1101u) > p.height.z * 0.5 + 0.5);
    let rl = rnd(brick, 1104u);
    h *= mix(1.0, rl, max(p.height.w, 0.0)) * mix(1.0, 1.0 - rl, max(-p.height.w, 0.0));
    v *= h;
    // Slope: a ramp along x and one along y per brick (sign = direction, some flipped), mixed per brick.
    if (any(p.slope != vec4f(0.0))) {
        let flip = rnd(brick, 1105u) > p.slope.w * 0.5 + 0.5;
        var gx = rx.y / rx.z;
        var gy = ry.y / ry.z;
        if ((p.slope.x < 0.0) != flip) { gx = 1.0 - gx; }
        if ((p.slope.y < 0.0) != flip) { gy = 1.0 - gy; }
        let sx = 1.0 - abs(p.slope.x) * (1.0 - gx);
        let sy = 1.0 - abs(p.slope.y) * (1.0 - gy);
        v *= mix(sy, sx, clamp(rnd(brick, 1102u) + p.slope.z, 0.0, 1.0));
    }
    store(c, vec4f(v, v, v, 1.0));
}
