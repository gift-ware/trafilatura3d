// The atomic Blend's blending modes (its Blending mode index), shared with the library's blend filters.
fn blend_atomic(b: vec3f, f: vec3f, mode: i32) -> vec3f {
    switch mode {
        case 1: { return b + f; }                                    // Add (linear dodge)
        case 2: { return b - f; }                                    // Subtract (below 0 in a float format)
        case 3: { return b * f; }                                    // Multiply
        case 4: { return b + 2.0 * f - 1.0; }                        // Add sub: above 0.5 adds, below subtracts
        case 5: { return max(b, f); }                                // Max (lighten)
        case 6: { return min(b, f); }                                // Min (darken)
        case 8: { return b / max(f, vec3f(1e-4)); }                  // Divide
        case 9: { return select(1.0 - 2.0 * (1.0 - b) * (1.0 - f), 2.0 * b * f, b < vec3f(0.5)); } // Overlay
        case 10: { return 1.0 - (1.0 - b) * (1.0 - f); }             // Screen
        case 11: {                                                   // Soft light
            let lo = 2.0 * b * f + b * b * (1.0 - 2.0 * f);
            let hi = 2.0 * b * (1.0 - f) + sqrt(max(b, vec3f(0.0))) * (2.0 * f - 1.0);
            return select(hi, lo, f < vec3f(0.5));
        }
        default: { return f; }                                       // Copy, Switch
    }
}
