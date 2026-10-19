// Cube 3D: analytic orthographic ray cast of the rotated box.
struct CubeHit {
    hit: bool,
    depth: f32,      // output value: 1/3 + world z in frame widths
    obj: vec3f,      // hit point in object space, 0-1 over the box
    normal: vec3f,   // world-space normal (x right, y up, z to the viewer)
};

fn cube_hit(c: vec2i) -> CubeHit {
    var r: CubeHit;
    let s = screen_point(c, p.non_square_expansion);
    let h = max(0.5 * p.size * p.scale, vec3f(1e-5));
    // Turned about the vertical axis by Y, then tilted towards the viewer about the screen x axis by X.
    let m = rot3(vec3f(p.orientation_offset.x, 0.0, 0.0)) * rot3(vec3f(0.0, p.orientation_offset.y, 0.0));
    let mt = transpose(m);
    let o = mt * vec3f(s, 4.0);
    let dir = mt * vec3f(0.0, 0.0, -1.0);
    let t = ray_box(o, dir, h);
    r.hit = t.x <= t.y && t.y > 0.0;
    if (!r.hit) { return r; }
    let q = o + dir * t.x;
    r.obj = q / (2.0 * h) + 0.5;
    // The face hit: the axis where q touches the box.
    let f = abs(q) / h;
    var n = vec3f(0.0, 0.0, sign(q.z));
    if (f.x >= f.y && f.x >= f.z) { n = vec3f(sign(q.x), 0.0, 0.0); }
    else if (f.y >= f.z) { n = vec3f(0.0, sign(q.y), 0.0); }
    r.normal = m * n;
    r.depth = clamp(1.0 / 3.0 + (4.0 - t.x), 0.0, 1.0);
    return r;
}
