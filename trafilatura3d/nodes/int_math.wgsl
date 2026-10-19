// Integer remainders and wrapping, for every shader (nodes through common.wgsl, the paint passes that do not include
// it). Never WGSL's % on signed integers: NVIDIA (Vulkan) computes a negative dividend's remainder as if it were
// unsigned (-3 % 29 = 13 instead of -3); a - b * (a / b) is right on every GPU. b == 0 gives 0.

// a % b as WGSL defines it: truncated, the sign of a.
fn imod(a: i32, b: i32) -> i32 { return select(a - b * (a / b), 0, b == 0); }
fn imod_v2(a: vec2i, b: vec2i) -> vec2i { return select(a - b * (a / b), vec2i(0), b == vec2i(0)); }
fn imod_v3(a: vec3i, b: vec3i) -> vec3i { return select(a - b * (a / b), vec3i(0), b == vec3i(0)); }
fn imod_v4(a: vec4i, b: vec4i) -> vec4i { return select(a - b * (a / b), vec4i(0), b == vec4i(0)); }

// i wrapped into 0..n-1 (n > 0).
fn wrap1(i: i32, n: i32) -> i32 { let m = imod(i, n); return select(m, m + n, m < 0); }
fn wrap(c: vec2i, d: vec2i) -> vec2i { let m = imod_v2(c, d); return select(m, m + d, m < vec2i(0)); }
fn wrap3(q: vec3i, d: vec3i) -> vec3i { let m = imod_v3(q, d); return select(m, m + d, m < vec3i(0)); }
