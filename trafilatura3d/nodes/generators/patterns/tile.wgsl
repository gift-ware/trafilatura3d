// A tile of the tile-grid nodes (Tile Generator, Tile Sampler, Splatter, Shape Splatter): where it is, how its
// local frame maps to the image, what it draws; and its antialiasing width.

struct Tile {
    center: vec2f,   // offset of the tile centre from its cell's base position, in uv
    inv: mat2x2f,    // uv offset from the centre -> local q (the tile spans [-1, 1]^2)
    color: vec4f,    // node data: usually rgb multiplier and opacity
    extra: vec4f,    // node data
    kind: i32,       // node data: usually the pattern (>= 0 shape, < 0 input -kind-1)
    spec: f32,       // node data: usually Pattern Specific
    key: f32,        // render order
    visible: bool,
};

// Antialiasing width in q units for a tile (one output pixel through its inverse transform).
fn tg_aa(t: Tile) -> f32 {
    return pixel_uv() * max(length(t.inv[0]), length(t.inv[1]));
}
