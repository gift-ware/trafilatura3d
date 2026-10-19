// Trilinear sampling of a 3D texture atlas; the node defines `vol_texel(t: vec2i) -> f32` and
// `vol_res() -> vec2i` (its slice resolution). q in the unit cube.
fn vol_voxel(v: vec3i) -> f32 {
    let r = vol_res();
    let c = clamp(v, vec3i(0), vec3i(r.x - 1, r.y - 1, fx_slices() - 1));
    return vol_texel(fx_voxel_texel(c, r));
}

// Nearest voxel of a point of the unit cube.
fn vol_nearest(q: vec3f, res: vec2i) -> vec3i { return vec3i(floor(fx_cube_voxel(q, res) + vec3f(0.0, 0.0, 0.5))); }

fn vol_sample(q: vec3f) -> f32 {
    let x = fx_cube_voxel(q, vol_res()) - vec3f(0.5, 0.5, 0.0);
    let i = vec3i(floor(x));
    let f = x - floor(x);
    let a = mix(mix(vol_voxel(i), vol_voxel(i + vec3i(1, 0, 0)), f.x),
                mix(vol_voxel(i + vec3i(0, 1, 0)), vol_voxel(i + vec3i(1, 1, 0)), f.x), f.y);
    let b = mix(mix(vol_voxel(i + vec3i(0, 0, 1)), vol_voxel(i + vec3i(1, 0, 1)), f.x),
                mix(vol_voxel(i + vec3i(0, 1, 1)), vol_voxel(i + vec3i(1, 1, 1)), f.x), f.y);
    return mix(a, b, f.z);
}

// Unit cube [0, 1]^3 hit by the ray o + t d: (t_near, t_far), t_far < t_near when missed.
fn vol_box(o: vec3f, d: vec3f) -> vec2f {
    let inv = 1.0 / select(d, vec3f(1e-8), abs(d) < vec3f(1e-8));
    let t0 = (vec3f(0.0) - o) * inv;
    let t1 = (vec3f(1.0) - o) * inv;
    let lo = min(t0, t1);
    let hi = max(t0, t1);
    return vec2f(max(max(lo.x, lo.y), max(lo.z, 0.0)), min(hi.x, min(hi.y, hi.z)));
}

// A direction from the Camera Position widget, in a Y-up space: X = azimuth (a full turn, 0.5 = facing the
// image's bottom side), Y = elevation (0 = below, 0.5 = level, 1 = above, kept just short of the poles).
// Returned in our world (Z up = the slice axis): its X, Z, Y.
fn vol_dir_from_xy(pos: vec2f) -> vec3f {
    let q = pos - 0.5;
    let e = clamp(q.y * 0.5, -0.2499, 0.2499) * 2.0 * PI - PI;
    let a = -q.x * 2.0 * PI - PI;
    return vec3f(cos(e) * sin(a), cos(e) * cos(a), -sin(e));
}

// The camera: at Camera Distance (cube widths) from the
// cube's centre along vol_dir_from_xy(Camera Position), looking at it; the screen spans -0.5..0.5 x
// tan(FOV / 2) on its shorter side. Returns the ray direction for output uv.
fn vol_camera(pos: vec2f, dist: f32, fov_deg: f32, uv: vec2f, eye: ptr<function, vec3f>) -> vec3f {
    let c = vol_dir_from_xy(pos);
    *eye = vec3f(0.5) + c * dist;
    // In the Y-up space the right vector is cross(Y, c), the up one cross(right, c).
    let cd = c.xzy;
    let right = normalize(cross(vec3f(0.0, 1.0, 0.0), cd) + vec3f(1e-7, 0.0, 0.0));
    let up = cross(right, cd);
    let ratio = p.size_ / min(p.size_.x, p.size_.y);
    let s = (uv - 0.5) * ratio * tan(radians(clamp(fov_deg, 1.0, 179.0)) * 0.5);
    return normalize((right * s.x + up * s.y - cd).xzy);
}
