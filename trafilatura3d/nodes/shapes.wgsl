// The built-in pattern shapes, one definition for the FX-Map Quadrant patterns and every node drawing a shape
// (Shape, Tile Generator, Tile Sampler, splatters...). The Shape node's documentation montage shows the
// patterns, and at Pattern Specific / Variation 0 each shape here is fitted to it (mean error
// under 0.003, Crescent 0.024, Capsule 0.02). The variation changes only Brick,
// Gradation, Waves, Half Bell, Crescent and Capsule; what it does to them is ours (the files only carry the index;
// Half Bell's is fitted on the preview of Crystal 2).
// No uniforms: included before the node's own code (patterns/lib.wgsl) or the FX-Map draw shader.

// Coverage of a signed distance d (positive inside) with an antialiasing width w (same units).
fn edge(d: f32, w: f32) -> f32 { return clamp(d / max(w, 1e-6) + 0.5, 0.0, 1.0); }
// A ramp of width s from the border inwards (never sharper than the antialiasing width).
fn ramp(d: f32, s: f32, w: f32) -> f32 {
    let k = max(s, w);
    return clamp((d + 0.5 * w) / k, 0.0, 1.0);
}

// The pattern `kind`, the pattern index - 2 (Square 0 ... Capsule 13, Cone 14, Gradation w. offset 15: the
// plain gradation, the FX-Map adds its offset; Hemisphere 16) on the local square q in [-1, 1]^2 (0
// outside), y down. `s`: Pattern Specific / Variation (0-1), `w`: the antialiasing width in q units.
fn shape(kind: i32, q: vec2f, s: f32, w: f32) -> f32 {
    let r = length(q);
    let a = abs(q);
    let m = max(a.x, a.y);
    let box = edge(1.0 - m, w);
    let dome = max(1.0 - r * r, 0.0);
    switch kind {
        case 0: { return edge(1.0 - m, w); }                                      // Square
        case 1: { return edge(1.0 - r, w); }                                      // Disc
        case 2: { return dome; }                                                  // Paraboloid
        case 3: { return pow(dome, 4.0); }                                        // Bell
        case 4: { return exp2(-16.0 * r * r) * box; }                             // Gaussian
        case 5: { return pow(max(1.0 - r, 0.0), 4.0); }                           // Thorn
        case 6: { return max(1.0 - m, 0.0); }                                     // Pyramid
        case 7: {                                                                 // Brick: s = 2:1
            let e = max(a.x, a.y / (1.0 - 0.5 * s));
            return 1.0 - smoothstep(0.65, 1.0, e);
        }
        case 8: {                                                                 // Gradation
            let t = clamp(q.x * 0.5 + 0.5, 0.0, 1.0);
            return box * mix(t, t * t * (3.0 - 2.0 * t), s);
        }
        case 9: {                                                                 // Waves: s = more waves
            return dome * (0.5 + 0.5 * cos(3.14159265 * (3.0 + 2.0 * round(4.0 * s)) * q.x));
        }
        case 10: {                                                                // Half Bell: s = cut side
            // Streaks with one hard edge (fitted at 0.84 on Crystal 2's preview): the variation
            // dims the x > 0 half to 1 - s.
            return pow(max(1.0 - 1.2 * q.x * q.x - q.y * q.y, 0.0), 4.0) * mix(1.0, 1.0 - s, edge(q.x, w));
        }
        case 11: { return pow(dome, 4.0) * pow(max(1.0 - a.x, 0.0), 2.0); }                  // Ridged Bell
        case 12: {                                                                // Crescent: s = thinner
            let t = (r - 0.548) / 0.517;                                          // pow of a negative base is NaN
            let ring = pow(max(1.0 - t * t, 0.0), 3.0);
            let side = pow(max(0.5 - 0.5 * q.x / max(r, 1e-6), 0.0), 1.09 * (1.0 + 3.0 * s));
            return ring * (0.13 + 0.87 * side);
        }
        case 13: {                                                                // Capsule: s = thicker
            let h = 0.145 + 0.355 * s;
            return edge(h - length(vec2f(max(a.x - (0.52 - h), 0.0), q.y)), w);
        }
        case 14: { return max(1.0 - r, 0.0); }                                    // Cone
        case 15: { return box * clamp(q.x * 0.5 + 0.5, 0.0, 1.0); }               // Gradation w. offset
        case 16: { return sqrt(dome); }                                           // Hemisphere
        default: { return 0.0; }
    }
}

// The pattern nodes (Shape, Tile Generator, Tile Sampler, Splatter Circular, Tile Random, Arc Pavement) draw the
// shapes with hard edges: a pixel is in or out by its centre. The Stamp Map keeps one pixel of antialiasing.
fn pattern_shape(kind: i32, q: vec2f, s: f32) -> f32 { return shape(kind, q, s, 0.0); }
