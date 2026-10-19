@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let uv = frame(c, p.non_square_expansion);
    let n = f32(max(p.scale, 1)) / 2.0;     // four rows per unit
    let gap = p.interstice_width * 0.5;
    let bev = p.edge_smoothness * 0.5;
    // Even rows: one brick per row. Odd rows: two bricks (brick_simple shifts them by half a brick),
    // moved so that their joints fall at 45/256 and 173/256 of the tile.
    let odd = imod2(floor(uv.y * n * 4.0)) == 1.0;
    let v = select(brick_simple(uv, n, 1.0, 4.0, gap, bev, false),
                   brick_simple(uv + vec2f((0.25 - 45.0 / 256.0) / n, 0.0), n, 2.0, 4.0, gap, bev, false), odd);
    store(c, vec4f(v, v, v, 1.0));
}
