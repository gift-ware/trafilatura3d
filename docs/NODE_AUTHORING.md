# Writing a node

A node is defined by what it takes, what it gives and its parameters (names, ranges, menus). `catalog` names the
node's entry in the reference index (`reference/graph/*.json`, `reference/paint/*.json`), and the conformance test
checks its parameters against it.

## Files

A node is a Python module plus a `.wgsl` file with the same stem, in `trafilatura3d/nodes/<area>/`
(packages are discovered recursively, every folder needs an `__init__.py`; adding the two files
adds the node). Parameter names are snake_case of the label and must not be WGSL reserved words
(the registry refuses them with a message). Multiple outputs: name them after their labels, snake_case.

```python
from trafilatura3d.nodes.base import Input, Node, Output, Param, ParamKind, PortType

class Threshold(Node):
    """One line saying what it does."""
    type = "threshold"                      # stable id, snake_case, unique
    label = "Threshold"                     # name shown in the Library
    category = "Filters/Adjustments"        # Library category
    catalog = "threshold"                   # id in the reference index
    inputs = {"input": Input(PortType.GRAY)}
    outputs = {"output": Output(PortType.GRAY)}
    params = {"threshold": Param(ParamKind.FLOAT, 0.5, "Threshold", 0.0, 1.0)}
```

- **Parameter labels** are the names shown in Properties. The conformance test checks every parameter of the
  node's reference entry exists by label, or is listed in `unsupported = {"Label": "reason"}`.
- **Defaults**: choose the value that gives the node's typical look, and keep it sensible at 256–4096 px.
- **Ranges** (`min`, `max`) are slider ranges.
- **Ports**: `PortType.GRAY`, `COLOR`, or `ANY` (follows what is connected: color if any `ANY`
  input is color). A node with a grayscale and a color variant is **one node with `ANY` ports** when
  the variants differ only in the image type; two nodes when they differ in parameters.
- `Input(default=(r, g, b, a))` is what an unconnected input reads.
- `main_input = "port"` when the size/format inheritance should follow another input than the first.

## Shader

The generated header gives (see `trafilatura3d/nodes/base.py`):

- `p.<param>` — every parameter (ENUM: index in `choices`; BOOL: 0/1; COLOR/FLOAT4: vec4f),
  `p.size_` (output size, vec2f), `p.seed_` (random seed, f32), `p.tiling_` (bits: 1 H, 2 V).
- per input `x`: `load_x(c)` (texel at output pixel c; an input of another size is read bilinearly at the
  pixel's centre), `texel_x(q)` (texel at the input's own pixel q, tiling applied), `sample_x(uv)` (bilinear),
  `size_x()`, `X_CONNECTED`, `X_GRAY`. Grayscale inputs read as `vec4f(v, v, v, 1)`.
- per output `y`: `store_y(c, v)`; `store(c, v)` for the first output. Quantization and clamping
  to the output's bit depth are automatic (`quantize_depth` in `common.wgsl`: an 8 or 16 bit result is exactly
  k/255 or k/65535, the same whether the device stores 16 bits as integers or as 32 bit floats).
- `lut_<param>(t)` for GRADIENT and CURVE parameters.
- `common.wgsl`: `tile_coord`, `wrap`, `pcg`, `hash21`, `hash22`, `hash2`, `luminance`, `rgb_to_hsl`,
  `hsl_to_rgb`, `hsl_shift` (the HSL node), `contrast_luminosity`, `PI`. Colour maths shared by several areas
  lives here, once; don't copy it into a folder's lib.

Rules:

- **Every declared input and parameter must be used by the shader** (`layout="auto"` drops unused
  bindings and the bind group no longer matches). Use `X_CONNECTED` to branch, not omission.
- **Resolution independence**: express distances in UV (0–1) or "pixels at 256 px" scaled by
  `p.size_.x / 256`, never raw pixels, so a graph looks the same at every output size.
- **Tiling**: generators must tile (periodic noise lattices, wrap-around); filters read through
  `texel_/sample_` which honour the node's tiling.
- **Randomness**: only from `hash*`/`pcg` seeded with `p.seed_` (and parameters), so results are
  identical on every GPU and cook.
- Write grayscale results as `vec4f(v, v, v, 1.0)`.
- Multi-pass nodes override `encode(self, ctx, job)`: `ctx.scratch(job, gray, depth="32f", size=None)`
  gives temporary textures, `ctx.dispatch(node, job, values, srcs={"port": (texture, is_gray)},
  dsts={"output": texture}, size=(w, h))` records a pass; add pass indices to `internal_uniforms`. A pass writing
  into a scratch texture is not quantized; one writing into the job's own result (`ctx.ev.texture(job, out)`) is.
  Passes with another shader use **helper nodes**: a `Node` subclass with `registered = False`, its
  own `inputs`/`outputs`/`params`/`internal_uniforms` and `wgsl()` returning its shader file; job
  inputs with the same port name are bound automatically. Examples: `atomic/blur.py` (two passes,
  same shader), `atomic/distance.py` (jump flooding: seed, flood passes, final pass helpers).
- Data made on the CPU: `ctx.ev.upload(key, make)` returns a cached rgba32float texture from
  `make()` (HxWx4 float32); `extra(values, package)` adds data to the job key (files and mtimes);
  example `atomic/raster.py` (Bitmap, SVG, Text).
- Nodes with a static **Color mode** set `color_mode = "<param name>"` (choices
  `("Grayscale", "Color")` or `("Color", "Grayscale")`); `OUT_<NAME>_GRAY` tells
  the shader whether an output is grayscale.
- Look at the atomic nodes in `trafilatura3d/nodes/atomic/` before writing new ones: they show the
  idioms (UV maths, resolution scaling, tiling, alpha handling).
- **Outputs**: any number (each has its own quantization field `q_<output>`). **Inputs**: at most
  16 sampled textures per shader (inputs + LUTs): split into passes beyond that.
- **Value ports** (`PortType.VALUE`): a single value is a 1x1 rgba32float texture; a value input
  reads it with `value_<port>()`, a value output is written by `store_<port>(c, v)` (only the
  invocation at (0,0) writes). Values never drive base-parameter inheritance. Examples:
  `atomic/value_processor.py`, `atomic/io.py` (Input value).
- **Ports that depend on parameters** (Pixel processor's input count): override
  `instantiate(values)` to return a copy with the right `inputs` (see `atomic/pixel_processor.py`).
- Numbered inputs enabled by a count parameter ("Pattern Input Number", "Input Number"):
  declare `counted_inputs = {"pattern_input_number": "pattern_input_"}`. Keep every numbered input in
  `inputs` (the shader reads them); the graph shows and accepts only the first *count* of them.
- **Render passes** (instanced drawing with hardware blending): `ctx.render(record)` records
  `record(encoder)` in order with the dispatches; targets are `rgba16float` scratch textures (they
  have RENDER_ATTACHMENT). Example: `atomic/stamp_map.py`.
- **Function and FX-Map parameters** (`ParamKind.FUNCTION`, `ParamKind.FXMAP`) are compiled by
  the node itself (`trafilatura3d/functions.py`: `FunctionGraph`, `WGSLCodegen`, `evaluate`;
  `trafilatura3d/fxmap.py`: `walk`); the job's `extra["graph_params"]` carries the graph parameters
  and input values they may Get.

## Rules for parallel work

- Write only in your area folder `trafilatura3d/nodes/<area>/` and `tests/test_<area>.py`. **Never edit
  core files** (`base.py`, `common.wgsl`, `evaluator.py`, `compiler.py`, `package.py`, other areas):
  if you need a core feature, stop and report what and why.
- Run your checks with `TRAFILATURA3D_CATEGORY="<category prefix>" python -m unittest tests.test_nodes`
  and `python -m unittest tests.test_<area>`; other areas may be mid-work.
- Render your category with `tools/node_sheet.py` and **look at the image**: a node that cooks but
  doesn't show the look it is meant to give is not done.

## Tests

`tests/test_nodes.py` cooks every registered node at 64 px with test inputs on every input and
checks: no NaN/inf, the output size, determinism, tiling for generators, and catalog conformance.
For each category also write `tests/test_<area>.py` with
behavioural checks (e.g. "Invert of 0.2 is 0.8", "Threshold output is only 0 and 1").

Look at the result: `python tools/node_sheet.py "Filters/Adjustments" out.png` renders every node of
a category with its default parameters on the standard test inputs.
