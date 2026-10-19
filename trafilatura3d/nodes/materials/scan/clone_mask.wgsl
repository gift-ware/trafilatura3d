// Patch mask evaluated on a grid of 2^Grid Resolution cells: inside the stamp the source shows; in a
// band at its edge (as wide as Threshold, per channel where the node has them) the border moves
// inwards where source and target differ, so the seam runs where they look alike.
fn band(uv: vec2f, su: vec2f, nz: f32) -> f32 { return $BAND; }

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let n = f32(1 << u32(clamp(p.grid_resolution, 1, 11)));
    let uv = (floor(uv_of(c, d) * n) + 0.5) / n;
    let sd = stamp_sd(tgt_local(uv));
    let edge = sd + 0.8 * band(uv, src_uv(uv), (vnoise(uv, vec2i(i32(n)), p.seed_) - 0.5) * 0.3);
    store(c, g4(select(0.0, 1.0, edge < 0.0 && sd < 0.0)));
}
