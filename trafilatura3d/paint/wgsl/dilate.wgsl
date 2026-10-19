// Padding: texels outside the UV islands (mask) take the value of the nearest texel inside, by jump flooding.
// Stage 0 seeds (inside texels store their own coordinate), stage 1 floods with step p.step, stage 3 writes the
// nearest map (Nearest node, cached per UV mask), stage 2 writes the result from it: the nearest inside value within p.distance texels (negative: no limit), else the fallback
// (p.outside: 0 transparent, 1 the channel default p.color). A seed is a texel's coordinate in two channels
// (rg32float: read as (x, x, x, y), x < 0 none).

fn seed(c: vec2i) -> vec4f { return vec4f(f32(c.x), f32(c.x), f32(c.x), f32(c.y)); }
fn at(s: vec4f) -> vec2f { return vec2f(s.x, s.w); }

fn inside(q: vec2i) -> bool { return texel_mask(q).r > 0.5; }

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let c = vec2i(id.xy);
    if (any(c >= size())) { return; }
    if (p.stage == 0) {
        store(c, select(vec4f(-1.0), seed(c), inside(c)));
        return;
    }
    if (p.stage == 1) {
        var best = texel_seeds(c);
        var bd = select(1e20, dot(at(best) - vec2f(c), at(best) - vec2f(c)), best.x >= 0.0);
        for (var j = -1; j <= 1; j++) {
            for (var i = -1; i <= 1; i++) {
                let q = c + vec2i(i, j) * p.step;
                if (any(q < vec2i(0)) || any(q >= size())) { continue; }
                let s = texel_seeds(q);
                if (s.x < 0.0) { continue; }
                let d = dot(at(s) - vec2f(c), at(s) - vec2f(c));
                if (d < bd) { bd = d; best = s; }
            }
        }
        store(c, best);
        return;
    }
    if (p.stage == 3) { store(c, texel_seeds(c)); return; }  // the nearest map itself (Nearest node)
    let v = texel_image(c);
    if (inside(c)) { store(c, v); return; }
    let s = texel_seeds(c);
    if (s.x < 0.0) { store(c, select(vec4f(0.0), vec4f(p.color.rgb, 0.0), p.outside == 1)); return; }
    let d = length(at(s) - vec2f(c));
    let near = texel_image(vec2i(at(s)));
    if (p.distance < 0.0 || d <= p.distance) { store(c, near); return; }
    store(c, select(vec4f(0.0), vec4f(p.color.rgb, 0.0), p.outside == 1));
}
