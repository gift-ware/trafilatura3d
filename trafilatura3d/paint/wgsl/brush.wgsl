// Stamps of a brush stroke accumulated in texture space. For every texel of the Texture Set (its world
// position, normal and tangent from the UV maps) each stamp gives a coverage: the brush alpha at the texel's
// place in the stamp, times the stamp's flow; coverages accumulate as s += (1 - s) * a (overlapping stamps
// build up, Stroke Opacity caps the result at commit), or with Stamps Blending Lighten as s = max(s, a) (overlaps
// never exceed the strongest stamp).
//
// Alignment (paint-brush page): 0 Camera (the stamp faces the viewport: texels projected to the screen, hidden
// ones skipped), 1 Tangent | Wrap (the stamp on the surface plane at the hit, following the surface within the
// stamp's radius in depth), 2 Tangent | Planar (the same, fading with the distance to the plane), 3 UV (the
// stamp in the surface plane turned by the mesh tangent), 4 Texture (painting in the 2D view: the stamp in
// texture space, center and radius in texels). Backface culling: texels whose normal is further than the angle
// from the stamp direction are skipped (not in texture space).

struct Uniforms {
    view_proj: mat4x4f,
    eye: vec4f,           // camera position, stamps blending (0 accumulate, 1 lighten)
    viewport: vec4f,      // width, height (pixels of the camera buffer), depth tolerance, block size (texels)
    params: vec4f,        // alignment, stamp count, cos(backface angle), hardness of the planar fade
};
struct Stamp {
    center: vec4f,        // world hit, radius (world)
    normal: vec4f,        // surface normal at the hit, flow
    axis_u: vec4f,        // stamp X axis in world (tangent alignments), angle (radians)
    screen: vec4f,        // center in camera buffer pixels, radius in pixels, depth in radii (0: 1)
};
@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> stamps: array<Stamp>;
@group(0) @binding(2) var pos_map: texture_2d<f32>;
@group(0) @binding(3) var nrm_map: texture_2d<f32>;
@group(0) @binding(4) var cam_pos: texture_2d<f32>;
@group(0) @binding(5) var alpha_tex: texture_2d<f32>;
@group(0) @binding(6) var alpha_sampler: sampler;
@group(0) @binding(7) var stroke: texture_storage_2d<r32float, read_write>;
// The texels written since the viewport last showed the stroke: min x, min y, max x, max y (Stroke.take_dirty).
@group(0) @binding(8) var<storage, read_write> dirty: array<atomic<u32>, 4>;
// The blocks this pass changed, a bit each, row by row (Stroke._lay saves them before the commit).
@group(0) @binding(9) var<storage, read_write> blocks: array<atomic<u32>>;

// The angle carries Flip copy (Radial symmetry): + 1000 flips the alpha in U, + 2000 in V (brush.FLIP_CODE).
fn brush_alpha(q: vec2f, coded: f32) -> f32 {
    let flip = floor((coded + 500.0) / 1000.0);
    let angle = coded - flip * 1000.0;
    let r = mat2x2f(cos(angle), -sin(angle), sin(angle), cos(angle));
    var d = r * q;
    if (flip == 1.0) { d.x = -d.x; }
    if (flip == 2.0) { d.y = -d.y; }
    if (abs(d.x) > 1.0 || abs(d.y) > 1.0) { return 0.0; }
    return textureSampleLevel(alpha_tex, alpha_sampler, d * vec2f(0.5, -0.5) + 0.5, 0.0).r;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let c = vec2i(id.xy);
    if (any(c >= vec2i(textureDimensions(pos_map)))) { return; }
    let pw = textureLoad(pos_map, c, 0);
    if (pw.a < 0.5) { return; }
    let p = pw.xyz;
    let n = normalize(textureLoad(nrm_map, c, 0).xyz);
    let mode = i32(u.params.x);
    var s = textureLoad(stroke, c).r;
    let s0 = s;
    var visible = true;
    var screen = vec2f(0.0);
    if (mode == 0) {
        let clip = u.view_proj * vec4f(p, 1.0);
        if (clip.w <= 0.0) { return; }
        let ndc = clip.xy / clip.w;
        screen = vec2f(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5) * u.viewport.xy;
        let q = vec2i(floor(screen));
        if (any(q < vec2i(0)) || any(q >= vec2i(u.viewport.xy))) { return; }
        let seen = textureLoad(cam_pos, q, 0);
        visible = seen.a > 0.0 && distance(seen.xyz, u.eye.xyz) + u.viewport.z >= distance(p, u.eye.xyz);
        if (!visible) { return; }
    }
    let count = i32(u.params.y);
    for (var i = 0; i < count; i++) {
        let st = stamps[i];
        var a = 0.0;
        if (mode == 4) {
            a = brush_alpha((vec2f(c) + 0.5 - st.screen.xy) / st.screen.z, st.axis_u.w);
        } else if (mode == 0) {
            if (dot(n, normalize(u.eye.xyz - p)) < u.params.z) { continue; }
            a = brush_alpha((screen - st.screen.xy) / st.screen.z, st.axis_u.w);
        } else {
            let nh = normalize(st.normal.xyz);
            if (dot(n, nh) < u.params.z) { continue; }
            let d = (p - st.center.xyz) / st.center.w;
            let w = dot(d, nh);
            // Depth reached from the stamp's plane, in stamp radii: 1, or the path's Projection depth.
            let depth = select(1.0, st.screen.w, st.screen.w > 0.0);
            if (abs(w) > depth) { continue; }
            let ax = normalize(st.axis_u.xyz);
            let ay = cross(nh, ax);
            a = brush_alpha(vec2f(dot(d, ax), dot(d, ay)), st.axis_u.w);
            if (mode == 2) { a *= 1.0 - smoothstep(u.params.w * depth, depth, abs(w)); }
        }
        let k = clamp(a * st.normal.w, 0.0, 1.0);
        s = select(s + (1.0 - s) * k, max(s, k), u.eye.w > 0.5);
    }
    if (s != s0) {
        textureStore(stroke, c, vec4f(s, 0.0, 0.0, 0.0));
        atomicMin(&dirty[0], u32(c.x));
        atomicMin(&dirty[1], u32(c.y));
        atomicMax(&dirty[2], u32(c.x));
        atomicMax(&dirty[3], u32(c.y));
        let size = i32(u.viewport.w);
        let per_row = (i32(textureDimensions(pos_map).x) + size - 1) / size;
        let b = u32((c.y / size) * per_row + c.x / size);
        atomicOr(&blocks[b >> 5u], 1u << (b & 31u));
    }
}
