// Kuwahara smoothing of one channel (`channel_`, its Smoothness `radius_` = radius in pixels at 256 px): the mean
// of the least varied of four quadrants, each sampled on a 4x4 grid (constant cost at any size); global
// Smoothness mixes it. The grid is separable: the horizontal pass (_h) averages the 4 samples of each row on
// both sides, the vertical pass (_v) averages 4 rows of them (bilinear samples: the same result as the grid).
fn fetch(uv: vec2f) -> vec4f {
    switch p.channel_ {
        case 0: { return sample_basecolor(uv); }
        case 1: { return sample_roughness(uv); }
        case 2: { return sample_metallic(uv); }
        case 3: { return sample_normal(uv); }
        case 4: { return sample_height(uv); }
        default: { return sample_ambient_occlusion(uv); }
    }
}

// The grid step in UV on each axis (a third of the radius).
fn grid_step() -> vec2f { return p.radius_ * mx_px() / p.size_ / 3.0; }
