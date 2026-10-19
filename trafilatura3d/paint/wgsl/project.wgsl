// The values the Projection tool (or the stencil) lays, in texture space: every texel of the Texture Set projected
// to the viewport of the stroke and the resource sampled there through the projection (projection.wgsl). A
// uniform channel (no resource) is its value everywhere. Alpha: how much is laid: 0 outside the projected image,
// times the falloff by angle (projection_weight) and 0 on surfaces hidden from the view (occlusion, from the
// camera buffer) when those are asked.

struct Uniforms {
    view_proj: mat4x4f,
    xf: vec4f,       // offset x, offset y, rotation, scale
    params: vec4f,   // aspect, tiling (+ flips), has image (1), gray (1)
    color: vec4f,    // the uniform value; A = 1 when the image is a single channel texture
    eye: vec4f,      // camera position, occlusion (1)
    fade: vec4f,     // cos of the fade start and end angles, falloff on (1), depth tolerance
    viewport: vec4f, // camera buffer width, height
};
@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var pos_map: texture_2d<f32>;
@group(0) @binding(2) var image: texture_2d<f32>;
@group(0) @binding(3) var dst: texture_storage_2d<rgba32float, write>;
@group(0) @binding(4) var nrm_map: texture_2d<f32>;
@group(0) @binding(5) var cam_pos: texture_2d<f32>;

//INCLUDE projection.wgsl

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let c = vec2i(id.xy);
    if (any(c >= vec2i(textureDimensions(pos_map)))) { return; }
    let pw = textureLoad(pos_map, c, 0);
    var weight = 1.0;
    let clip = u.view_proj * vec4f(pw.xyz, 1.0);
    var s = vec2f(-1.0);
    if (pw.a > 0.5 && clip.w > 0.0) {
        let ndc = clip.xy / clip.w;
        s = vec2f(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    } else {
        weight = 0.0;
    }
    if (weight > 0.0 && u.fade.z > 0.5) {
        weight *= projection_weight(pw.xyz, textureLoad(nrm_map, c, 0).xyz, u.eye.xyz, u.fade.xy);
    }
    if (weight > 0.0 && u.eye.w > 0.5) {  // only the surface the view sees
        let q = vec2i(floor(s * u.viewport.xy));
        if (any(q < vec2i(0)) || any(q >= vec2i(u.viewport.xy))) {
            weight = 0.0;
        } else {
            let seen = textureLoad(cam_pos, q, 0);
            if (seen.a <= 0.0 || distance(seen.xyz, u.eye.xyz) + u.fade.w < distance(pw.xyz, u.eye.xyz)) {
                weight = 0.0;
            }
        }
    }
    var v = vec4f(u.color.rgb, weight);
    if (u.params.z > 0.5) {
        var t = vec4f(0.0);
        var inside = 0.0;
        if (all(s >= vec2f(-0.5))) {
            let q = projection_uv(s, u.params.x, u.xf, i32(u.params.y));
            t = bilinear_wrap(image, q.xy);
            if (u.color.a > 0.5) { t = vec4f(t.rrr, 1.0); }  // a single channel image (grayscale)
            inside = q.z;
        }
        v = vec4f(t.rgb, t.a * inside * weight);
        if (u.params.w > 0.5) { v = vec4f(vec3f(dot(t.rgb, vec3f(0.2126, 0.7152, 0.0722))), v.a); }
    }
    textureStore(dst, c, v);
}
