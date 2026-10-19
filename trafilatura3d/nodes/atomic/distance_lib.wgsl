// The seeds: the nearest white pixel's coordinates (-1: none yet) packed in the two channels of a 16 bit texture
// (_jfa_lib.wgsl), stored as (x, 0, 0, y) and read back as (x, x, x, y).
fn seed_at(q: vec2i) -> vec2f { return jfa_unpack(texel_seeds(q)).xw; }
