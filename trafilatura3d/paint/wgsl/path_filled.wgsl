// Filled path: the coverage of the shape the path closes. Every texel of the Texture Set (its world position) is
// projected on the path's best fitting plane; inside the projected curve (even-odd rule), within `depth` of the
// plane and on a face turned toward the plane's normal it is covered (ours: the documentation gives no rule).

struct Uniforms {
    center: vec4f,  // plane origin, point count
    ax: vec4f,      // plane X axis, depth (world)
    ay: vec4f,      // plane Y axis
    nz: vec4f,      // plane normal
    bounds: vec4f,  // the curve's 2D bounding box: min xy, max xy
};
@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> poly: array<vec4f>;
@group(0) @binding(2) var pos_map: texture_2d<f32>;
@group(0) @binding(3) var nrm_map: texture_2d<f32>;
@group(0) @binding(4) var coverage: texture_storage_2d<r32float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let c = vec2i(id.xy);
    if (any(c >= vec2i(textureDimensions(pos_map)))) { return; }
    let pw = textureLoad(pos_map, c, 0);
    var inside = false;
    if (pw.a > 0.5) {
        let d = pw.xyz - u.center.xyz;
        let q = vec2f(dot(d, u.ax.xyz), dot(d, u.ay.xyz));
        let n = textureLoad(nrm_map, c, 0).xyz;
        if (abs(dot(d, u.nz.xyz)) <= u.ax.w && dot(n, u.nz.xyz) > 0.0 && all(q >= u.bounds.xy) && all(q <= u.bounds.zw)) {
            let count = i32(u.center.w);
            for (var i = 0; i < count; i++) {
                let a = poly[i].xy;
                let b = poly[(i + 1) % count].xy;
                if ((a.y > q.y) != (b.y > q.y) && q.x < (b.x - a.x) * (q.y - a.y) / (b.y - a.y) + a.x) {
                    inside = !inside;
                }
            }
        }
    }
    if (inside) { textureStore(coverage, c, vec4f(1.0, 0.0, 0.0, 0.0)); }  // added to what is there (symmetry)
}
