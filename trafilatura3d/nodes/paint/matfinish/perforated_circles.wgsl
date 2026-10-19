// Staggered grid: `cols` holes across, an even number of rows (so the stagger tiles) close to the hexagonal
// spacing. .x relief (the pressed rim), .y distance to the nearest hole's centre and .z the hole radius, both in
// column spacings.
fn grid() -> vec2i {
    let cols = mf_cells(10.0 * p.scale);
    return vec2i(cols, max(2, 2 * i32(round(f32(cols) / 0.8660254 / 2.0))));
}
fn pattern(x: vec2f) -> vec4f {
    let g = grid();
    let y = x * vec2f(g);
    let aspect = f32(g.x) / f32(g.y);  // a row spacing in column spacings
    let row = floor(y.y);
    var d = 9.0;
    for (var j = -1; j <= 1; j++) {
        let r = row + f32(j);
        let s = select(0.0, 0.5, wrap1(i32(r), 2) == 1);
        let cx = floor(y.x - s) + 0.5 + s;
        for (var k = -1; k <= 1; k++) {
            d = min(d, length(vec2f(y.x - cx - f32(k), (y.y - r - 0.5) * aspect)));
        }
    }
    let rad = 0.5 * p.holes_size;
    let rim = smoothstep(rad, rad + 0.18, d);
    return vec4f(rim * 0.02 / f32(g.x), d, rad, 0.0);
}
fn effect(c: vec2i) -> vec4f {
    let r = mf_relief(c);
    let k = p.details_intensity;
    let px = f32(grid().x) / p.size_.x;  // a pixel, in column spacings
    let hole = 1.0 - smoothstep(r.v.z - px, r.v.z + px, r.v.y);
    let edge = 1.0 - smoothstep(r.v.z, r.v.z + 0.15, r.v.y);
    store_opacity(c, mf_gray(load_opacity(c).r * (1.0 - hole)));
    store_roughness(c, mf_gray(load_roughness(c).r + 0.15 * k * edge * (1.0 - hole)));
    store_normal(c, mf_normal(load_normal(c), r.g * (1.0 - hole), k * 2.0));
    let bc = load_basecolor(c);
    return vec4f(mix(bc.rgb, vec3f(0.02), hole), bc.a);
}
