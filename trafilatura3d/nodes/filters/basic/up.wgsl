// Resampling of `input` to the output size along one axis (AXIS: vec2i(1, 0) or vec2i(0, 1); the other axis keeps
// its size) with a cubic B-spline of 4 texels: smooth to the second derivative, so an enlarged blur shows no creases
// along the rows and columns of its texels (a bilinear enlargement is only continuous, its slope jumps there). It
// adds a variance of 1/3 texel^2 of the input (BSPLINE_SPREAD). Two passes, one per axis, make the 2D one.
@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let c = vec2i(id.xy);
    if (any(c >= size())) { return; }
    let x = dot(uv_at(c) * vec2f(size_input()), vec2f(AXIS)) - 0.5;
    let t = fract(x);
    let s = 1.0 - t;
    let t2 = t * t;
    let t3 = t2 * t;
    let w = vec4f(s * s * s, 3.0 * t3 - 6.0 * t2 + 4.0, -3.0 * t3 + 3.0 * t2 + 3.0 * t + 1.0, t3) / 6.0;
    let q = c * (vec2i(1) - AXIS) + (i32(floor(x)) - 1) * AXIS;
    store(c, w.x * texel_input(q) + w.y * texel_input(q + AXIS) + w.z * texel_input(q + 2 * AXIS)
             + w.w * texel_input(q + 3 * AXIS));
}
