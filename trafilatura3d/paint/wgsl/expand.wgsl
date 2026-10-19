// Painted pages into a whole texture (paint.Pages.expand): every texel of the area `u.area` (x0, y0, x1, y1) from
// its page's layer (`table`: page -> layer + 1), 0 where nothing is painted. FORMAT is replaced by the
// destination's storage format; SRC_TWO / DST_TWO tell a two channel storage (value, how much is painted): read
// as RGBA (value repeated), written back in two.

struct Uniforms {
    area: vec4u,
    grid: vec4u,  // page side, pages per row
};
@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var pages: texture_2d_array<f32>;
@group(0) @binding(2) var<storage, read> table: array<u32>;
@group(0) @binding(3) var dst: texture_storage_2d<FORMAT, write>;
const SRC_IS_TWO = SRC_TWO;
const DST_IS_TWO = DST_TWO;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let c = id.xy + u.area.xy;
    if (any(c >= u.area.zw)) { return; }
    let side = u.grid.x;
    let layer = table[(c.y / side) * u.grid.y + c.x / side];
    var v = vec4f(0.0);
    if (layer != 0u) { v = textureLoad(pages, c % side, i32(layer) - 1, 0); }
    if (SRC_IS_TWO) { v = v.rrrg; }
    if (DST_IS_TWO) { v = vec4f(v.r, v.a, 0.0, 0.0); }
    textureStore(dst, vec2i(c), v);
}
