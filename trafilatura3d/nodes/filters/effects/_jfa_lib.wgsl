// Jump flooding seeds packed for its passes: a coordinate c (or -1, none) as (c + 1) / 65535, which a 16 bit
// integer texture holds exactly up to 65534: half a 32 bit float's memory traffic, the same coordinates.
fn jfa_pack(s: vec4f) -> vec4f { return (s + 1.0) / 65535.0; }
fn jfa_unpack(v: vec4f) -> vec4f { return round(v * 65535.0) - 1.0; }
