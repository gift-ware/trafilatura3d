@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let uv = fx_uv(c);
    let m = mat2x2f(p.matrix.x, p.matrix.y, p.matrix.z, p.matrix.w);
    let det = determinant(m);
    if (abs(det) < 1e-8) { store(c, sample_input(uv)); return; }
    let inv = mat2x2f(m[1][1], -m[0][1], -m[1][0], m[0][0]) * (1.0 / det);
    let centre = 0.5 - inv * p.offset;
    var rel = uv - centre;
    if (tile_h()) { rel.x -= round(rel.x); }
    if (tile_v()) { rel.y -= round(rel.y); }
    let local = m * rel;
    let f = 1.0 - smoothstep(0.0, 0.5, length(local));
    let a = p.amount * 0.5 * PI * f * f;  // Amount / 4 turns at the centre, easing out to nothing at the rim
    let turned = vec2f(cos(a) * local.x + sin(a) * local.y, cos(a) * local.y - sin(a) * local.x);
    store(c, sample_input(centre + inv * turned));
}
