@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let uv = (vec2f(c) + 0.5) / vec2f(d);
    // Convention: the matrix (column-major: a b = first column) and the offset say where to sample,
    // input point = M * (output point - centre) + centre + offset (Noise Upscale's 2 0 0 2 tiles twice; a positive
    // offset moves the image left / up, as the Clone documentation says).
    let m = p.transformation_matrix;
    let sampling = mat2x2f(m.x, m.y, m.z, m.w);
    let q = sampling * (uv - 0.5) + 0.5 + p.offset;
    let out_x = q.x < 0.0 || q.x > 1.0;
    let out_y = q.y < 0.0 || q.y > 1.0;
    if ((out_x && !tile_h()) || (out_y && !tile_v())) {
        var k = p.matte_color;
        if (INPUT_GRAY) { k = vec4f(k.r, k.r, k.r, 1.0); }
        store(c, k);
        return;
    }
    // Footprint of one output pixel in input texels (for downscaling).
    let scale = vec2f(length(sampling[0]), length(sampling[1])) *
        vec2f(size_input()) / vec2f(d);
    var level = 0.0;
    if (p.mipmap_mode == 1) { level = f32(p.mipmap_level); }
    else if (p.filtering == 0) { level = max(log2(max(scale.x, scale.y)), 0.0); }
    let taps = clamp(i32(exp2(floor(level))), 1, 8);
    if (p.filtering == 2) {
        store(c, texel_input(vec2i(floor(q * vec2f(size_input())))));
        return;
    }
    var acc = vec4f(0.0);
    let step = exp2(level) / vec2f(size_input()) / f32(taps);
    for (var j = 0; j < taps; j++) {
        for (var i = 0; i < taps; i++) {
            acc += sample_input(q + (vec2f(f32(i), f32(j)) - f32(taps - 1) * 0.5) * step);
        }
    }
    store(c, acc / f32(taps * taps));
}
