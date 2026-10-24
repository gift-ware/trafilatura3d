# Library shaders

Trafilatura 3D's shaders are a table (`trafilatura3d/shaders.py`: OpenPBR, Metallic roughness, Unlit…). A project can
add its own: one `.wgsl` file per shader in the `shaders` folder of a library it declares. They appear in every shader
list next to the table's (the 3D View's materials, New graph templates, the Paint's New Project and Shader settings) and
render in the rasterizer and in the path tracer.

## Libraries

A library is a folder with one sub-folder per kind of resource (`project.LIBRARY_KINDS`):

```
studio_library/
├── shaders/          .wgsl library shaders (this page)
├── export_presets/   .json output templates of the Paint's Export textures (read only)
├── materials/        packages (the Paint's Assets)
└── smart_materials/  smart materials (the Paint's Assets)
```

A project file declares it (relative to the project file):

```json
{"trafilatura3d_project": 1, "libraries": {"paths": ["studio_library"]}}
```

Libraries are read when Trafilatura 3D starts, from the projects of its configuration (`TRAFILATURA3D_CONFIG` or the
CLI's `--config`): a pipeline loads its library by starting Trafilatura 3D with its configuration, nobody else sees it.

## A shader file

```wgsl
//: shader "Studio/Lit"
//: channel baseColor baseColor "Base map" = 1,1,1
//: channel emissive emissive "Emissive map" = 1,1,1
//: param color base_tint "Base tint" = 1,1,1
//: param float smoothness "Smoothness" = 0.5 [0,1]
//: blending opaque
fn shade(m: M, c: Ch, L: Lobes) -> Lobes {
    var out = L;
    out.albedo = lin(c.base) * base_tint(m);
    out.rough = 1.0 - smoothness(m);
    out.emission = lin(c.emit);
    return out;
}
```

The header, one `//:` line each:

| Line | |
|---|---|
| `shader "Name"` | the name in the shader lists; unique (a name the table or another library shader has is refused) |
| `channel <usage> <slot> "Label" [= v,v,v]` | a channel: graph outputs of this usage and the Texture Set channel of that name feed the renderer slot (`shaders.SLOTS`); its row is the slot's value without a texture (default: the slot's) |
| `param <float\|color> <key> "Label" [= v[,v,v]] [[lo,hi]]` | a parameter (at most 8): read with `<key>(m)`, an `f32` or a `vec3f` |
| `blending opaque` / `alpha_test <threshold>` / `alpha_blend` / `choice` | how the opacity is used; `choice`: the Blending mode and Alpha threshold rows |

The code defines `fn shade(m: M, c: Ch, L: Lobes) -> Lobes` (and any helper functions). `c` holds the channels as
sampled (colors sRGB encoded: `lin()` decodes them), `L` the lobes at their defaults (albedo, roughness, emission and
opacity from the channels, a dielectric of IOR 1.5); `shade` returns them filled, and the renderers light them like the
table's shaders. `out.model = MODEL_UNLIT` draws `out.albedo` (linear) as is. The opacity `shade` returns drives the
rasterizer's blending; shadows and the path tracer's rays read the Opacity channel.

Every function of the file is renamed for the shader (`shade` becomes `shade_l0`…), so two files can define the same
helper names. A file that breaks the format is reported at start and left out; a WGSL error is reported by the
renderer when it builds its shaders.
