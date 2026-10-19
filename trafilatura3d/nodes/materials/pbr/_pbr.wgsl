// ---- pbr/_pbr.wgsl ----
// Linear F0 (reflectance at normal incidence) of pure metals: Gold, Silver, Aluminium, Iron, Copper,
// Titanium, Nickel, Cobalt, Platinum.
fn metal_f0(i: i32) -> vec3f {
    switch i {
        case 0: { return vec3f(1.000, 0.766, 0.336); }
        case 1: { return vec3f(0.972, 0.960, 0.915); }
        case 2: { return vec3f(0.913, 0.922, 0.924); }
        case 3: { return vec3f(0.560, 0.570, 0.580); }
        case 4: { return vec3f(0.955, 0.638, 0.538); }
        case 5: { return vec3f(0.542, 0.497, 0.449); }
        case 6: { return vec3f(0.660, 0.609, 0.526); }
        case 7: { return vec3f(0.662, 0.655, 0.634); }
        default: { return vec3f(0.672, 0.637, 0.585); }
    }
}
// Linear F0 range (min, max) of dielectric families: Plastic, Wood, Stone, Brick, Sand, Concrete,
// Fabric, Rusted Metal, Water, Ice, Glass.
fn dielectric_f0_range(i: i32) -> vec2f {
    switch i {
        case 0: { return vec2f(0.040, 0.050); }
        case 1: { return vec2f(0.030, 0.045); }
        case 2: { return vec2f(0.035, 0.056); }
        case 3: { return vec2f(0.035, 0.045); }
        case 4: { return vec2f(0.030, 0.046); }
        case 5: { return vec2f(0.035, 0.045); }
        case 6: { return vec2f(0.040, 0.056); }
        case 7: { return vec2f(0.030, 0.056); }
        case 8: { return vec2f(0.020, 0.020); }
        case 9: { return vec2f(0.018, 0.018); }
        default: { return vec2f(0.040, 0.050); }
    }
}
fn ior_f0(n: f32) -> f32 { let r = (n - 1.0) / (n + 1.0); return r * r; }
