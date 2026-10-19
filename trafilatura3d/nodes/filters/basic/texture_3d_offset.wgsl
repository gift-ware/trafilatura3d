fn grid() -> i32 { return 1 << u32(round(log2(f32(size_input().x)) / 3.0)); }

fn voxel(q: vec3i, n: i32, g: i32) -> vec4f {
    let w = wrap(q.xy, vec2i(n));
    let z = wrap1(q.z, g * g);
    return texel_input(vec2i(z % g, z / g) * n + w);
}

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let g = grid();
    let n = max(size_input().x / g, 1);
    let depth = g * g;
    // This output pixel's voxel (the output uses the input's layout, scaled to the output size).
    let ip = vec2i(floor(uv_at(c) * vec2f(size_input())));
    let slice = ip / n;
    let z = slice.x + slice.y * g;
    let pos = (vec3f(vec2f(ip % vec2i(n)), f32(z)) + 0.5) / vec3f(f32(n), f32(n), f32(depth)) - p.offset;
    let x = pos * vec3f(f32(n), f32(n), f32(depth)) - 0.5;
    let i = vec3i(floor(x));
    let f = fract(x);
    let a = mix(mix(voxel(i, n, g), voxel(i + vec3i(1, 0, 0), n, g), f.x),
                mix(voxel(i + vec3i(0, 1, 0), n, g), voxel(i + vec3i(1, 1, 0), n, g), f.x), f.y);
    let b = mix(mix(voxel(i + vec3i(0, 0, 1), n, g), voxel(i + vec3i(1, 0, 1), n, g), f.x),
                mix(voxel(i + vec3i(0, 1, 1), n, g), voxel(i + vec3i(1, 1, 1), n, g), f.x), f.y);
    store(c, mix(a, b, f.z));
}
