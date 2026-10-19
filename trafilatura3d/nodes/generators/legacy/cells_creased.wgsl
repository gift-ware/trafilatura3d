// Creases (ours): the soft shards of `strokes` moved sideways by themselves: the pixel reads them at
// uv + (p.shift + p.fold x s, 0), s the shards at uv + (p.shift, 0) (shifts in images; bright shards fold right, dark
// ones left); clamped to 0-1.
@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let c = vec2i(id.xy);
    if (c.x >= size().x || c.y >= size().y) { return; }
    let uv = (vec2f(c) + 0.5) / p.size_;
    let s = sample_strokes(uv + vec2f(p.shift, 0.0)).r;
    let v = clamp(sample_strokes(uv + vec2f(p.shift + p.fold * s, 0.0)).r, 0.0, 1.0);
    store(c, vec4f(v, v, v, 1.0));
}
