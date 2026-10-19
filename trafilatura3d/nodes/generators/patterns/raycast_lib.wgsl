// Shared by the 3D-rendering patterns (Cube 3D, Cube 3D GBuffers, Height Extrude, Shape Extrude).
// World frame: x right, y up (= -v on screen), z towards the viewer; orthographic camera looking down -z.

// Rotation about x, then y, then z, angles in turns (object -> world).
fn rot3(a: vec3f) -> mat3x3f {
    let cx = cos(a.x * TAU); let sx = sin(a.x * TAU);
    let cy = cos(a.y * TAU); let sy = sin(a.y * TAU);
    let cz = cos(a.z * TAU); let sz = sin(a.z * TAU);
    let rx = mat3x3f(1.0, 0.0, 0.0, 0.0, cx, sx, 0.0, -sx, cx);
    let ry = mat3x3f(cy, 0.0, -sy, 0.0, 1.0, 0.0, sy, 0.0, cy);
    let rz = mat3x3f(cz, sz, 0.0, -sz, cz, 0.0, 0.0, 0.0, 1.0);
    return rz * ry * rx;
}

// Screen point of an output pixel in the world frame, centred (x right, y up).
fn screen_point(c: vec2i, nse: i32) -> vec2f {
    let uv = frame(c, nse) - frame_size(nse) * 0.5;
    return vec2f(uv.x, -uv.y);
}

// Ray / axis-aligned box (half extents h, centred at 0): (t near, t far), near > far when missed.
fn ray_box(o: vec3f, d: vec3f, h: vec3f) -> vec2f {
    let inv = 1.0 / select(d, vec3f(1e-9), abs(d) < vec3f(1e-9));
    let t0 = (-h - o) * inv;
    let t1 = (h - o) * inv;
    let lo = min(t0, t1);
    let hi = max(t0, t1);
    return vec2f(max(max(lo.x, lo.y), lo.z), min(min(hi.x, hi.y), hi.z));
}
