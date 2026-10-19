// 3D textures: the slices of a unit cube in a grid atlas (the node defines
// `vol_grid() -> vec2i`, columns x rows; 16x16 = 256 slices by default), slice k at column k % columns,
// row k / columns. A voxel's X/Y run across its slice (Y down, like the image), its Z is k / slices.

fn fx_slices() -> i32 { let g = vol_grid(); return g.x * g.y; }

fn fx_slice_res(atlas: vec2i) -> vec2i { return max(atlas / vol_grid(), vec2i(1)); }

fn fx_voxel_texel(v: vec3i, res: vec2i) -> vec2i {
    let k = clamp(v.z, 0, fx_slices() - 1);
    let g = vol_grid().x;
    return vec2i(k % g, k / g) * res + clamp(v.xy, vec2i(0), res - 1);
}

fn fx_texel_voxel(t: vec2i, res: vec2i) -> vec3i {
    let cell = t / res;
    return vec3i(t - cell * res, cell.y * vol_grid().x + cell.x);
}

// Voxel coordinates (continuous: voxel centres at +0.5 across a slice, slice k at k) of a point of the unit cube.
fn fx_cube_voxel(q: vec3f, res: vec2i) -> vec3f { return q * vec3f(vec2f(res), f32(fx_slices())); }

// Voxel size in unit-cube units.
fn fx_voxel_size(res: vec2i) -> vec3f { return 1.0 / vec3f(vec2f(res), f32(fx_slices())); }
