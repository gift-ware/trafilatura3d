// Passes of the Auto Stitcher's arc length along its paths (list ranking by pointer jumping), `p.pass_`:
// 0 successor: each path pixel (path_.r) links to the neighbour path pixel most along its tangent (path_.gb):
//   (x, y, own id, step length), or -1;
// 1 min jump: state (next xy, min id along the next 2^k pixels): next = next of next, min id merged; after
//   enough jumps a pixel still linked is on a loop and knows the loop's smallest id;
// 2 cut: the successor links, without the link into that smallest id on loops: (next xy, step length);
// 3 rank jump: (next xy, length to the end of the chain): length += next's length, next = next of next.
fn pid(q: vec2f) -> f32 { return q.y * p.size_.x + q.x; }
fn at(q: vec2f) -> vec4f { return texel_a_(vec2i(q)); }

fn effect(c: vec2i) -> vec4f {
    let none = vec4f(-1.0, -1.0, 0.0, -1.0);
    let me = texel_a_(c);
    if (p.pass_ == 0) {
        if (me.r < 0.5) { return none; }
        var best = 0.3;
        var found = vec2f(-1.0);
        for (var j = -1; j <= 1; j++) {
            for (var i = -1; i <= 1; i++) {
                let o = vec2i(i, j);
                let q = c + o;
                if ((i == 0 && j == 0) || any(q < vec2i(0)) || any(q >= size())) { continue; }
                if (texel_a_(q).r < 0.5) { continue; }
                let d = dot(normalize(vec2f(o)), me.gb);
                if (d > best) { best = d; found = vec2f(q); }
            }
        }
        if (found.x < 0.0) { return vec4f(-1.0, -1.0, pid(vec2f(c)), 0.0); }
        return vec4f(found, pid(vec2f(c)), length(found - vec2f(c)));
    }
    if (p.pass_ == 1) {
        if (me.x < 0.0) { return me; }
        let n = at(me.xy);
        return vec4f(n.xy, min(me.z, n.z), me.w);
    }
    if (p.pass_ == 2) {
        let st = texel_b_(c);
        if (me.x < 0.0) { return vec4f(-1.0, -1.0, 0.0, 0.0); }
        if (st.x >= 0.0 && pid(me.xy) == st.z) { return vec4f(-1.0, -1.0, 0.0, 0.0); }  // into the loop's start
        return vec4f(me.xy, me.w, 0.0);
    }
    if (me.x < 0.0) { return me; }
    let n = at(me.xy);
    return vec4f(n.xy, me.z + n.z, 0.0);
}
