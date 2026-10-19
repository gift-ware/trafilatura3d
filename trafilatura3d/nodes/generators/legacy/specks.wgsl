// Specks (ours): sparse round dots over `input`. The image is p.cells x p.cells cells; a cell holds a dot with
// probability p.density, at a random point (kept inside the cell), radius p.radius cells, a bell (1 - r^2)^2;
// bright with probability p.bright, else dark, its height p.amount x u^2 (u uniform: a few strong, many faint).
@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let n = max(p.cells, 1);
    let x = (vec2f(c) + 0.5) / vec2f(d) * f32(n);
    let q = vec2i(floor(x));
    let s = p.seed_ + 7.31;
    let h = hash22(q, s);
    var v = load_input(c).r;
    if (h.x < p.density) {
        let g = hash22(q, s + 1.7);
        let at = vec2f(q) + 0.5 + (g - 0.5) * (1.0 - 2.0 * p.radius);
        let r2 = dot(x - at, x - at) / (p.radius * p.radius);
        let b = max(1.0 - r2, 0.0);
        let u = h.y;
        let sign = select(-1.0, 1.0, hash21(q, s + 3.9) < p.bright);
        v += sign * p.amount * u * u * b * b;
    }
    let o = clamp(v, 0.0, 1.0);
    store(c, vec4f(o, o, o, 1.0));
}
