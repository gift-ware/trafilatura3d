fn hsv(h: f32, s: f32, v: f32) -> vec3f {
    let k = clamp(abs(fract(h + vec3f(0.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0 - 3.0) - 1.0, vec3f(0.0), vec3f(1.0));
    return v * mix(vec3f(1.0), k, s);
}

// 1 on lines of `width` (in cells) at the integer coordinates of x, antialiased over `aa` cells.
fn lines(x: vec2f, width: f32, aa: vec2f) -> f32 {
    let d = abs(x - round(x));
    let l = vec2f(1.0) - smoothstep(vec2f(width * 0.5), vec2f(width * 0.5) + aa, d);
    return max(l.x, l.y);
}

fn effect(c: vec2i) -> vec4f {
    let isl = load_uv_islands(c);
    var color = p.checker_color;
    if (p.checker_color_mode == 1) {
        let q = vec2i(round(isl.rg * 1048575.0));
        color = hsv(hash21(q, p.seed_), 0.75, 0.9);
    }
    // Square tiles: as many across the height as across the width per unit of length.
    let g = uv_of(c) * p.checker_grid_tile * vec2f(p.size_.x / p.size_.y, 1.0);
    let cell = vec2i(floor(g));
    let dark = ((cell.x + cell.y) & 1) == 1;
    var rgb = select(color, color * 0.55, dark);
    let px = p.checker_grid_tile / p.size_.y;  // one output pixel, in tiles
    let grid = max(lines(g, 0.03, vec2f(px)), 0.5 * lines(g * 4.0, 0.06, vec2f(px * 4.0)));
    rgb = mix(rgb, vec3f(0.08), grid * p.checker_grid_opacity);
    let s = load_seeds_(c);
    if (s.x >= 0.0 && p.uv_borders_width > 0.0 && isl.a >= 0.5) {
        let d = distance(vec2f(c), s.xy) * 256.0 / p.size_.x;  // px at 256
        let w = max(p.uv_borders_width, 256.0 / p.size_.x);  // at least one output pixel
        rgb = mix(rgb, p.uv_borders_color, select(0.0, 1.0, d < w));
    }
    return vec4f(rgb, 1.0);
}
