// Ray marching in a height field. The node defines `ray_height(uv) -> f32`: the surface height at uv
// in UV units (1 = the image's width).

// Geometric normal of the height field (Z up, +Y up) from central differences at one pixel.
fn ray_normal(uv: vec2f) -> vec3f {
    let e = 1.0 / p.size_;
    let sx = (ray_height(uv + vec2f(e.x, 0.0)) - ray_height(uv - vec2f(e.x, 0.0))) / (2.0 * e.x);
    let sy = (ray_height(uv + vec2f(0.0, e.y)) - ray_height(uv - vec2f(0.0, e.y))) / (2.0 * e.y);
    return fx_slope_normal(vec2f(sx, sy));
}

// Distance along the 3D direction d (Z up, +Y up) from (uv, height) to the first point under the
// surface, within max_t (UV); -1 when the ray escapes. `jit` in [0, 1) offsets the steps.
fn ray_march(uv: vec2f, d: vec3f, max_t: f32, steps: i32, jit: f32) -> f32 {
    let h0 = ray_height(uv) + 0.75 / p.size_.x;  // start just above the surface
    let img = vec2f(d.x, -d.y);                  // rows grow downwards
    for (var i = 0; i < steps; i++) {
        let f = (f32(i) + jit) / f32(steps);
        let t = max_t * f * f;                   // denser near the start
        if (t < 0.5 / p.size_.x) { continue; }
        if (ray_height(uv + img * t) > h0 + d.z * t) { return t; }
    }
    return -1.0;
}

// Direction i of n in a cone of half angle `spread` x 90 degrees around n, cosine-distributed
// (cosine = true) or uniform in solid angle; r rotates the set per pixel.
fn ray_dir(i: i32, n: i32, r: vec2f, axis: vec3f, spread: f32, cosine: bool) -> vec3f {
    let f = fx_fib(i, n, r);
    let cmax = cos(clamp(spread, 0.0, 1.0) * 0.5 * PI);
    var ct: f32;
    if (cosine) { ct = sqrt(1.0 - f.x * (1.0 - cmax * cmax)); }
    else { ct = 1.0 - f.x * (1.0 - cmax); }
    let st = sqrt(max(1.0 - ct * ct, 0.0));
    let ph = f.y * 2.0 * PI;
    return fx_basis(axis) * vec3f(st * cos(ph), st * sin(ph), ct);
}
