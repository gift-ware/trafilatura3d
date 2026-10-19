// Slope Blur's first pass, over the Slope map's texels: its central differences there, in uv ((s[x+1] - s[x-1]) / 2
// times its width, the same along y), as a grayscale value with its alpha (x, y). Their bilinear sample at a position
// is the difference of the slope's bilinear samples a texel either side of it (the interpolation is linear in the
// texels): the walk reads one sample per step instead of four.
@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let c = vec2i(id.xy);
    if (any(c >= size())) { return; }
    let gx = texel_slope(c + vec2i(1, 0)).r - texel_slope(c - vec2i(1, 0)).r;
    let gy = texel_slope(c + vec2i(0, 1)).r - texel_slope(c - vec2i(0, 1)).r;
    let g = vec2f(gx, gy) * 0.5 * vec2f(size_slope());
    store(c, vec4f(g.x, 0.0, 0.0, g.y));
}
