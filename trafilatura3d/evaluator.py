"""Cooks compiled jobs on the GPU. A job's key is a hash of everything that determines its result
(node, values, size, formats, tiling, seed, the keys of its inputs), so the cache needs no
invalidation: after an edit the graph is recompiled and only jobs with new keys are cooked.
Results are kept up to a memory budget, least recently used first out."""

from __future__ import annotations

import queue
import threading
import time
from collections import OrderedDict
from types import GeneratorType
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import wgpu
from wgpu.backends.wgpu_native.extras import write_timestamp

from trafilatura3d.nodes.base import LUT_SIZE, WORKGROUP, Node, ParamKind, PortType, Variant, two_channels

UNIFORM_ALIGN = 256  # minUniformBufferOffsetAlignment guaranteed by WebGPU
COMPACT_FEATURE = "texture-adapter-specific-format-features"
NORM16_FEATURE = "texture-format16bit-norm"  # 16 bit integer formats (r16unorm...), storage with COMPACT_FEATURE
BLENDABLE_FEATURE = "float32-blendable"  # FX-Map patterns drawn into a 32 bit float layer
FILTERABLE_FEATURE = "float32-filterable"  # 32 bit float textures read through the sampler (Variant.filtered)
TIMESTAMP_FEATURES = ("timestamp-query", "timestamp-query-inside-encoders")  # per-node GPU timings (`timing`)

# Bit depths -> quantization levels and storage format (gray, color, gray with its alpha). The one place where a
# depth becomes a format: the evaluator replaces what the device cannot store (`storage_formats`).
DEPTHS = {"8": (255.0, "r8unorm", "rgba8unorm", "rg8unorm"), "16": (65535.0, "r16unorm", "rgba16unorm", "rg16unorm"),
          "16f": (0.0, "r16float", "rgba16float", "rg16float"), "32f": (0.0, "r32float", "rgba32float", "rg32float")}
# Formats that need device features to be written by a shader, and their portable replacement. It holds the same
# values: store() quantizes a 16 bit result to k/65535 before writing it, which a 32 bit float keeps exactly.
_NEEDS = {"r8unorm": ((COMPACT_FEATURE,), "r32float"), "r16float": ((COMPACT_FEATURE,), "r32float"),
          "rg8unorm": ((COMPACT_FEATURE,), "rg32float"), "rg16float": ((COMPACT_FEATURE,), "rg32float"),
          "r16unorm": ((COMPACT_FEATURE, NORM16_FEATURE), "r32float"),
          "rg16unorm": ((COMPACT_FEATURE, NORM16_FEATURE), "rg32float"),
          "rgba16unorm": ((COMPACT_FEATURE, NORM16_FEATURE), "rgba32float")}
NORM16 = {"r16unorm", "rg16unorm", "rgba16unorm"}

BYTES_PER_PIXEL = {"r8unorm": 1, "r16float": 2, "r32float": 4, "rg8unorm": 2, "rg16float": 4, "rg32float": 8,
                   "rgba8unorm": 4, "rgba16float": 8, "rgba32float": 16, "rgba16uint": 8,
                   "r16unorm": 2, "rg16unorm": 4, "rgba16unorm": 8}
NUMPY = {"r8unorm": (np.uint8, 1), "r16float": (np.float16, 1), "r32float": (np.float32, 1),
         "rg8unorm": (np.uint8, 2), "rg16float": (np.float16, 2), "rg32float": (np.float32, 2),
         "rgba8unorm": (np.uint8, 4), "rgba16float": (np.float16, 4), "rgba32float": (np.float32, 4),
         "r16unorm": (np.uint16, 1), "rg16unorm": (np.uint16, 2), "rgba16unorm": (np.uint16, 4)}


def storage_formats(features) -> dict[str, str]:
    """Each storage format a depth may use (DEPTHS) -> the one a device with these features writes."""
    return {fmt: (fmt if all(f in features for f in needs) else other) for fmt, (needs, other) in _NEEDS.items()}

HALO = 8  # texels computed around a region (recook_region): what the pointwise nodes read around a texel, and more
POOL_MID_COOK = 512 << 20  # the free textures a cook keeps for its next jobs before giving them back
PREVIEW = "preview"
PARTITION_SHARE = {"": 1.0, PREVIEW: 0.25}  # each partition's budget, of `Evaluator.budget`

_device = None


def default_device() -> wgpu.GPUDevice:
    global _device
    if _device is None:
        adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
        if adapter is None:
            raise RuntimeError("no GPU adapter available")
        features = [f for f in (COMPACT_FEATURE, NORM16_FEATURE, BLENDABLE_FEATURE, FILTERABLE_FEATURE,
                                *TIMESTAMP_FEATURES) if f in adapter.features]
        _device = adapter.request_device_sync(
            required_features=features,
            required_limits={"max-texture-dimension-2d": adapter.limits["max-texture-dimension-2d"],
                             "max-storage-textures-per-shader-stage":
                                 adapter.limits["max-storage-textures-per-shader-stage"],
                             # the 3D view's materials: a texture per shader channel (render3d, pathtrace)
                             "max-sampled-textures-per-shader-stage":
                                 adapter.limits["max-sampled-textures-per-shader-stage"]})
    return _device


@dataclass(eq=False)
class Job:
    """One node to cook, fully resolved by the compiler."""
    key: str
    path: str  # where it comes from, e.g. "blend1" or "rock_instance/blur2"
    node: Node
    values: dict[str, Any]
    inputs: dict[str, tuple[Job, str] | None]  # port -> (source job, its output) or None
    output_types: dict[str, PortType]
    size: tuple[int, int]
    depths: dict[str, str]  # per output: "8" | "16" | "16f" | "32f"
    tiling: int  # bit 0 horizontal, bit 1 vertical
    seed: float
    extra: dict[str, Any] = field(default_factory=dict)  # node-specific data (e.g. a bitmap path)
    # Cached after the cook (within the budget). False: an intermediate, freed once its last reader is recorded,
    # computed again when needed (the Paint stack keeps only where an edit restarts from: layers, effects, masks).
    keep: bool = True
    # The cache partition: "" the main one; PREVIEW the reduced copies a viewport shows while a value is dragged,
    # with their own share of the budget (they never push the full resolution results out).
    partition: str = ""
    # Cheaper to compute again than to keep: a copy of data the GPU holds anyway (a painting's pages expanded). Never
    # a restart point (`restart_points`): kept, it doubled the paintings in the cache.
    cheap: bool = False

    def is_gray(self, output: str) -> bool:
        """One value per texel (with its alpha for GRAY_ALPHA)."""
        return self.output_types[output] in (PortType.GRAY, PortType.GRAY_ALPHA)

    def is_value(self, output: str) -> bool:
        return self.output_types[output] is PortType.VALUE


class TexturePool:
    USAGE = wgpu.TextureUsage.STORAGE_BINDING | wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_SRC \
        | wgpu.TextureUsage.COPY_DST
    RENDER = USAGE | wgpu.TextureUsage.RENDER_ATTACHMENT  # float targets drawn with blending (FX-Map layers)

    def __init__(self, device):
        self.device = device
        self.free: dict[tuple, list] = {}
        self.bytes = 0  # what the free textures hold
        self._doomed: list[tuple] = []  # (Fence, textures) trimmed while the GPU may still use them (`trim`)

    def acquire(self, fmt, w, h):
        stack = self.free.get((fmt, w, h))
        if stack:
            self.bytes -= w * h * BYTES_PER_PIXEL[fmt]
            return stack.pop()
        usage = self.RENDER if fmt in ("rgba16float", "rgba32float", "r16float", "r32float") else self.USAGE
        return self.device.create_texture(size=(w, h, 1), format=fmt, usage=usage)

    def release(self, texture):
        self.free.setdefault((texture.format, texture.width, texture.height), []).append(texture)
        self.bytes += texture.width * texture.height * BYTES_PER_PIXEL[texture.format]

    def trim(self, keep: int = 0, fence: Fence | None = None):
        """Destroys free textures until at most `keep` bytes are left. With `fence` (the work just submitted, which
        may still use them) they are destroyed once it is done (`reap`): destroying a texture the GPU is using
        waits for the GPU (~60 ms for a 4096 one, on the interface thread)."""
        self.reap()
        doomed = []
        for key in list(self.free):
            stack = self.free[key]
            while stack and self.bytes > keep:
                t = stack.pop()
                self.bytes -= t.width * t.height * BYTES_PER_PIXEL[t.format]
                doomed.append(t)
            if not stack:
                del self.free[key]
        if fence is not None and doomed:
            self._doomed.append((fence, doomed))
            return
        for t in doomed:
            t.destroy()

    def reap(self) -> None:
        """The textures `trim` left to their fence, destroyed where the GPU is done with them."""
        left = []
        for fence, textures in self._doomed:
            if fence.done():
                for t in textures:
                    t.destroy()
            else:
                left.append((fence, textures))
        self._doomed = left


@dataclass(frozen=True)
class _Mark:
    """A job's work starts (or ends) here: a GPU timestamp between passes."""
    key: str
    end: bool


class Scratch:
    """A job's temporary texture (`CookContext.scratch`): its format and size now, its GPU texture once the job is
    recorded (`CookContext.release_scratches`); anything else asked of it is asked of that texture (bound at once
    when it has none yet: a read back)."""

    def __init__(self, ctx: CookContext, fmt: str, width: int, height: int):
        self.ctx, self.format, self.width, self.height = ctx, fmt, width, height
        self.texture = None

    def __getattr__(self, name):
        if self.texture is None:
            self.ctx._bind_now(self)
        return getattr(self.texture, name)


class _Render:
    """A render pass recorded in order with the dispatches (`CookContext.render`) and the temporaries it uses."""

    def __init__(self, record, uses: list):
        self.record, self.uses = record, uses

    def __call__(self, encoder):
        self.record(encoder)


def _scratches_of(item) -> list:
    """The temporaries a recorded item uses: a dispatch's bound textures, a render pass's declared ones."""
    if isinstance(item, _Render):
        return item.uses
    if isinstance(item, tuple):
        return [v for v in item[3] if isinstance(v, Scratch)]
    return []


def _view(v):
    """A bound resource: a temporary's texture view, a buffer whole, else as it is."""
    if isinstance(v, Scratch):
        return v.texture.create_view()
    return {"buffer": v} if isinstance(v, wgpu.GPUBuffer) else v


class CookContext:
    """What a node's `encode` records into: one compute pass, uniforms staged in one buffer."""

    def __init__(self, ev: Evaluator):
        self.ev = ev
        self.uniforms = bytearray()
        self.pending = []  # (pipeline, uniform offset, size, entries without the uniform, workgroups)
        self.scratches = []
        self.transients = []  # uploads living for this cook only (`upload`)
        self.region = None  # (x0, y0, x1, y1): the jobs' own dispatches cover only that (Evaluator.recook_region)
        self.clip = None  # (x0, y0, x1, y1): with a region, where the whole results are written (its inner part)
        self.regional: dict = {}  # job key -> {output: texture of the region only} (recook_region's intermediates)
        self.alias: dict = {}  # job key -> (job, output) its readers read instead (recook_region's passed jobs)
        self._submitting = False  # within `submit`: a temporary without its texture is an undeclared use

    def dispatch(self, node: Node, job: Job, values: dict, srcs: dict | None = None,
                 dsts: dict | None = None, size: tuple[int, int] | None = None, bufs: dict | None = None) -> None:
        """One dispatch of `node`'s shader over `size` (default: the job's).

        srcs: port -> (texture, is_gray) overriding the job's inputs (e.g. a scratch texture);
        dsts: output -> texture overriding the job's results; bufs: each of the node's `buffers` (`buffer`).
        """
        w, h = size or job.size
        x0, y0 = 0, 0
        gw, gh = w, h  # the texels the dispatch covers
        if self.region is not None and size is None and not dsts:
            x0, y0, x1, y1 = self.region
            gw, gh = x1 - x0, y1 - y0
        variant, uniform, entries = self._bindings(node, job, values, srcs, dsts, (w, h), (x0, y0), size)
        entries += [bufs[name] for name in node.buffers]
        offset = len(self.uniforms)
        self.uniforms += uniform + bytes(-len(uniform) % UNIFORM_ALIGN)
        groups = ((gw + WORKGROUP - 1) // WORKGROUP, (gh + WORKGROUP - 1) // WORKGROUP)
        self.pending.append((self.ev.pipeline(node, variant), offset, len(uniform), entries, groups))

    def draw(self, node: Node, job: Job, values: dict, target, instances: int, srcs: dict | None = None) -> None:
        """Instanced quads of `node`'s shader blended into `target` (what it holds stays under them), in order with
        the dispatches: entry points `vs` (six vertices per instance) and `fs`, the blend state the node's
        `blend(values)` (a node without outputs: its inputs, uniforms and LUTs bound as `dispatch` binds them, with
        the target's size as `size()`)."""
        size = (target.width, target.height)
        variant, uniform, entries = self._bindings(node, job, values, srcs, None, size, (0, 0), size)
        pipeline, layout = self.ev.render_pipeline(node, variant, target.format, node.blend(values))
        device = self.ev.device

        def record(encoder):
            ubuf = device.create_buffer_with_data(data=uniform, usage=wgpu.BufferUsage.UNIFORM)
            group = device.create_bind_group(layout=layout, entries=[
                {"binding": 0, "resource": {"buffer": ubuf}}] + [
                {"binding": i + 1, "resource": _view(view)} for i, view in enumerate(entries)])
            rp = encoder.begin_render_pass(color_attachments=[{
                "view": _view(target), "load_op": "load", "store_op": "store"}])
            rp.set_pipeline(pipeline)
            rp.set_bind_group(0, group)
            rp.draw(6, instances)
            rp.end()
        if instances:
            self.render(record, uses=[target, *entries])

    def _bindings(self, node: Node, job: Job, values: dict, srcs, dsts, size, origin, explicit_size):
        """What `dispatch` and `draw` bind: the shader variant, the uniform bytes and the texture views after the
        uniform (inputs, outputs, LUTs). `explicit_size`: the size the caller gave (None: the job's, its region)."""
        ev = self.ev
        w, h = size
        x0, y0 = origin
        inputs = []
        for port, spec in node.inputs.items():
            if srcs and port in srcs:
                tex, gray = srcs[port]
                inputs.append((tex, True, gray))
            elif job.inputs.get(port):
                src, out = job.inputs[port]
                while src.key in self.alias:
                    src, out = self.alias[src.key]
                if src.key in self.regional:
                    inputs.append((self.regional[src.key][out], True, src.is_gray(out), True))
                    continue
                inputs.append((ev.texture(src, out), True, src.is_gray(out)))
            else:
                gray = spec.type is PortType.GRAY
                inputs.append((ev.constant(spec.default), False, gray))
        inputs = [i if len(i) == 4 else (*i, False) for i in inputs]
        regional = self.regional.get(job.key) if not dsts else None
        outputs = [(dsts or {}).get(o) or (regional[o] if regional else ev.texture(job, o)) for o in node.outputs]
        pages = job.extra.get("pages_on") if node.pointwise and explicit_size is None and not dsts else None
        variant = Variant(tuple((c, g, two_channels(t.format), r, c and not r and (t.width, t.height) != (w, h))
                                for t, c, g, r in inputs),
                          tuple(t.format for t in outputs), tuple(bool(regional) for _ in outputs), pages is not None,
                          ev.filters)
        clip = (self.clip if self.region is not None and explicit_size is None and not dsts and self.clip
                else (0, 0, 0, 0))
        builtin = {"size_": [float(w), float(h)], "seed_": float(job.seed), "tiling_": job.tiling,
                   "origin_": [float(x0), float(y0)], "clip_": [float(v) for v in clip],
                   **{f"pages{i}_": (pages[i] if pages else -1) for i in range(8)}}
        own = {id(t): o for o, t in ev.cache.get(job.key, {}).items()}  # the job's results, by texture
        for o in node.outputs:  # scratch targets are not quantized, the job's results are (whoever writes them)
            if dsts and o in dsts:
                mine = own.get(id(dsts[o]))
                builtin[f"q_{o}"] = DEPTHS[job.depths.get(mine, "16f")][0] if mine else 0.0
            else:
                builtin[f"q_{o}"] = DEPTHS[job.depths.get(o, "16f")][0]
        uniform = node.pack({**values, **builtin})
        # a temporary is kept as it is: its texture and view come when it is bound (`release_scratches`, `submit`)
        entries = [t if isinstance(t, Scratch) else t.create_view() for t in [i[0] for i in inputs] + outputs]
        entries += [ev.lut(node.params[name], values[name]).create_view() for name in node.lut_params()]
        if variant.filtered:
            entries.append(ev.sampler(job.tiling))
        return variant, uniform, entries

    def scratch(self, job: Job, gray: bool, depth: str = "32f", size: tuple[int, int] | None = None,
                alpha: bool = False):
        """A temporary texture (default: the job's size; `alpha`: a gray value with its alpha, two channels): its GPU
        texture is chosen once the job is recorded (`release_scratches`), shared with the job's other temporaries
        whose passes do not overlap."""
        w, h = size or job.size
        t = Scratch(self, self.ev.storage_format(depth, gray, alpha), w, h)
        self.scratches.append(t)
        return t

    def buffer(self, size: int):
        """A temporary storage buffer of `size` bytes for the dispatches' `bufs`, zeroed (WebGPU clears a new buffer);
        `read` reads it back. Released with the job's other work."""
        return self.ev.device.create_buffer(size=max(int(size), 4),
                                            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC)

    def render(self, record, uses=()) -> None:
        """Record a render pass in order with the dispatches: `record(encoder)` gets the command
        encoder between compute passes (FX-Map draws its patterns this way). `uses`: the textures it reads or writes
        (the temporaries among them get a texture before it runs)."""
        self.pending.append(_Render(record, [t for t in uses if isinstance(t, Scratch)]))

    def mark(self, key: str, end: bool) -> None:
        """Where a job's work starts or ends, for its timing (only with the evaluator's `timing` on): a GPU timestamp,
        or, where they don't work, what is recorded so far submitted and waited for."""
        ev = self.ev
        if not ev.timing:
            return
        if ev.timestamps_work():
            self.pending.append(_Mark(key, end))
            return
        scratches, self.scratches = self.scratches, []  # as `read`: the job's scratches stay its own
        self.submit()
        self.scratches, self.pending, self.uniforms, self.transients = scratches, [], bytearray(), []
        ev.wait()
        now = time.perf_counter()
        if not end:
            ev._timing_open[key] = now
        elif key in ev._timing_open:
            ev.timings[key] = max(0.0, now - ev._timing_open.pop(key) - ev._idle_wait) * 1000

    def submit(self) -> int:
        device = self.ev.device
        if not self.pending:
            return 0
        for item in self.pending:  # within a job (a read, a timing): its temporaries so far get their textures now
            for t in _scratches_of(item):
                if t.texture is None:
                    self._bind_now(t)
        self._submitting = True
        try:
            return self._submit(device)
        finally:
            self._submitting = False

    def _submit(self, device) -> int:
        encoder = device.create_command_encoder()
        cpass = None
        ubuf = device.create_buffer_with_data(data=bytes(self.uniforms or bytes(16)), usage=wgpu.BufferUsage.UNIFORM)
        marks = [m for m in self.pending if isinstance(m, _Mark)]
        stamps = device.create_query_set(type="timestamp", count=len(marks)) if marks else None
        written = 0
        for item in self.pending:
            if isinstance(item, _Mark):  # a timestamp between passes
                if cpass is not None:
                    cpass.end()
                    cpass = None
                write_timestamp(encoder, stamps, written)
                written += 1
                continue
            if callable(item):  # a render pass: close the compute pass around it
                if cpass is not None:
                    cpass.end()
                    cpass = None
                item(encoder)
                continue
            pipeline, offset, size, views, (gx, gy) = item
            if cpass is None:
                cpass = encoder.begin_compute_pass()
            entries = [{"binding": 0, "resource": {"buffer": ubuf, "offset": offset, "size": size}}]
            entries += [{"binding": i + 1, "resource": _view(v)} for i, v in enumerate(views)]
            group = device.create_bind_group(layout=pipeline.get_bind_group_layout(0), entries=entries)
            cpass.set_pipeline(pipeline)
            cpass.set_bind_group(0, group)
            cpass.dispatch_workgroups(gx, gy, 1)
        if cpass is not None:
            cpass.end()
        resolved = None
        if marks:
            resolved = device.create_buffer(size=8 * len(marks),
                                            usage=wgpu.BufferUsage.QUERY_RESOLVE | wgpu.BufferUsage.COPY_SRC)
            encoder.resolve_query_set(stamps, 0, len(marks), resolved, 0)
        device.queue.submit([encoder.finish()])
        if marks:  # read now (the timings wait for the GPU: only when they are shown)
            ns = np.frombuffer(device.queue.read_buffer(resolved), np.uint64)
            for m, t in zip(marks, ns):
                if not m.end:
                    self.ev._timing_open[m.key] = int(t)
                elif m.key in self.ev._timing_open:
                    self.ev.timings[m.key] = (int(t) - self.ev._timing_open.pop(m.key)) / 1e6
        self.release_scratches()
        for t in self.transients:  # freed once the submitted work is done (WebGPU defers it)
            t.destroy()
        return len(self.pending) - len(marks)

    def flush(self) -> Fence:
        """What is recorded so far submitted (the job's temporaries stay its own) and a fence on it: a node that reads
        back its own work yields it (its `encode` a generator) and reads once it is done, the event loop running
        meanwhile in a cook in steps (`Evaluator._cook`)."""
        scratches, self.scratches = self.scratches, []
        self.submit()
        self.scratches, self.pending, self.uniforms, self.transients = scratches, [], bytearray(), []
        return self.ev.fence()

    def read(self, tex) -> np.ndarray:
        """A small texture's values, now: what is recorded so far is submitted (the job's scratches stay its own)
        and the GPU waited for. For a node whose next passes depend on a result (Flood Fill sizes a buffer by the
        shapes it counted); (height, width, channels) as `Evaluator.read` gives them, a buffer (`buffer`) as u32."""
        scratches, self.scratches = self.scratches, []
        self.submit()
        self.scratches, self.pending, self.uniforms, self.transients = scratches, [], bytearray(), []
        if isinstance(tex, wgpu.GPUBuffer):
            return np.frombuffer(self.ev.device.queue.read_buffer(tex), np.uint32)
        if isinstance(tex, Scratch):
            if tex.texture is None:
                self._bind_now(tex)
            tex = tex.texture
        dtype, channels = NUMPY[tex.format]
        data = self.ev.device.queue.read_texture({"texture": tex, "origin": (0, 0, 0)},
                                                 {"bytes_per_row": tex.width * BYTES_PER_PIXEL[tex.format]},
                                                 (tex.width, tex.height, 1))
        return _unpacked(np.frombuffer(data, dtype).reshape(tex.height, tex.width, channels), tex.format)

    def release_scratches(self) -> None:
        """Once a job is recorded: its temporaries get their textures, in the order of its passes, each given back to
        the pool after the last pass using it so that the next ones reuse it (never within the pass that frees it: a
        texture is not read and written by one pass); then all of them back to the pool (later jobs reuse them: the
        dispatches are ordered). A job's temporaries take what its passes use at once, not their sum (Flood Fill at
        4096 held 1.5 GB of them to the end)."""
        spans: dict[int, list] = {}
        for i, item in enumerate(self.pending):
            for t in _scratches_of(item):
                if t.texture is None:
                    span = spans.setdefault(id(t), [i, i, t])
                    span[1] = i
        active = []
        for first, last, t in sorted(spans.values(), key=lambda s: s[0]):
            for done in [a for a in active if a[1] < first]:
                self.ev.pool.release(done[2].texture)
                active.remove(done)
            t.texture = self.ev.pool.acquire(t.format, t.width, t.height)
            active.append((first, last, t))
        for _, _, t in active:
            self.ev.pool.release(t.texture)
        for t in self.scratches:  # the ones bound before (`_bind_now`), kept to here
            if id(t) not in spans and t.texture is not None:
                self.ev.pool.release(t.texture)
        self.scratches.clear()

    def _bind_now(self, t: Scratch) -> None:
        """A temporary used other than by the passes (read back, or by a submit within its job): its texture now,
        kept to the end of its job."""
        if self._submitting:
            raise RuntimeError("a temporary texture used by a render pass that does not declare it (`render` uses)")
        t.texture = self.ev.pool.acquire(t.format, t.width, t.height)

    def upload(self, make):
        """A texture of a CPU image (float32 HxWx4 from `make()`) for this cook only: what a job reads once into
        its own result, which the cache keeps (kept uploads doubled the memory: a 4096 image is 256 MB)."""
        t = self.ev.texture_of(make())
        self.transients.append(t)
        return t


def _chains(schedule: list[Job], wanted: set) -> dict:
    """The chains of jobs one dispatch computes (Node.fuse_input): {last job's key: [jobs, inputs first]}. A job
    joins the next one when it is that job's fused input, read by nothing else in the cook, at the same size and not
    `wanted` (targets, held, precious): its result is never written, and is computed again if needed later."""
    readers: dict[str, int] = {}
    for job in schedule:
        for src in job.inputs.values():
            if src:
                readers[src[0].key] = readers.get(src[0].key, 0) + 1
    scheduled = {j.key: j for j in schedule}
    chains = {}
    for job in schedule:  # inputs first
        port = job.node.fuse_input
        src = job.inputs.get(port) if port else None
        prev = scheduled.get(src[0].key) if src else None
        if prev is None or prev.node.type != job.node.type or readers[prev.key] != 1 or prev.key in wanted \
                or tuple(prev.size) != tuple(job.size):
            continue
        chain = chains.pop(prev.key, [prev])
        if len(chain) >= job.node.max_fused:
            chains[prev.key] = chain
            continue
        chains[job.key] = chain + [job]
    return {k: c for k, c in chains.items() if len(c) > 1}


class Fence:
    """The GPU work submitted before it, asked whether done without waiting (`Evaluator.fence`): a helper thread waits
    for the mapping of a 4 byte buffer copied after that work, then sets `_event`."""

    def __init__(self, buffer, event: threading.Event):
        self._buffer, self._event = buffer, event

    def done(self) -> bool:
        if not self._event.is_set():
            return False
        if self._buffer is not None:
            self._buffer.destroy()
            self._buffer = None
        return True

    def wait(self) -> None:
        self._event.wait()
        self.done()


def _waiter(waits: queue.Queue) -> None:
    """The fences' helper thread: each mapping waited for, its event set (`Fence`)."""
    while True:
        promise, event = waits.get()
        try:
            promise.sync_wait()
        finally:
            event.set()


def _sources(job: Job) -> list[Job]:
    """The jobs a job reads, each once, in its ports' order."""
    return list({src[0].key: src[0] for src in job.inputs.values() if src}.values())


def run(steps):
    """A generator of steps (`Evaluator.cook_steps`) run to its end, now (each fence it gives waited for): its return
    value."""
    while True:
        try:
            step = next(steps)
        except StopIteration as done:
            return done.value
        if isinstance(step, Fence):
            step.wait()


def restart_points(targets, below: set) -> dict:
    """What the next edit of a selection reads, besides what is below it (`below`, keys of the results it is
    computed from): the other inputs of everything computed from those results (a Paint stack's masks and contents
    of the layers above, a graph's inputs beside the selected node's chain to the outputs) and every step of what
    starts from `below` (an edit of a layer's effect restarts from the one before). The cache gives them up last
    (`Evaluator.cook` precious). {key: job}."""
    if not below:
        return {}
    memo, nodes = {}, {}

    def reads_below(job):  # every input visited: the whole graph is walked
        if job.key not in memo:
            nodes[job.key] = job
            reads = [reads_below(src[0]) for src in job.inputs.values() if src]
            memo[job.key] = job.key in below or any(reads)
        return memo[job.key]

    for job in targets:
        reads_below(job)
    redone = {k for k, r in memo.items() if r and k not in below}
    keep, own = {}, []
    for k in redone:
        ins = [src[0] for src in nodes[k].inputs.values() if src and src[0].key not in redone and not src[0].cheap]
        keep.update((j.key, j) for j in ins)
        if any(j.key in below for j in ins):  # the selection's first step: its other inputs are its own
            own += [j for j in ins if j.key not in below]
    seen = set()
    while own:
        job = own.pop()
        if job.key not in seen:
            seen.add(job.key)
            if not job.cheap:
                keep[job.key] = job
            own += [src[0] for src in job.inputs.values() if src]
    return {k: j for k, j in keep.items() if k not in below}


@dataclass
class CookStats:
    cooked: list[str]  # paths of the jobs computed, in order
    dispatches: int
    encode_s: float


class Evaluator:
    def __init__(self, device=None, budget_bytes: int = 2 << 30, features=None):
        """`budget_bytes` caps the cached results: 0 streams (each result freed after its last use). `features`: the
        device features the storage formats may count on (default all the device has; fewer to test the fallbacks)."""
        self.device = device or default_device()
        self.formats = storage_formats(self.device.features if features is None else features)
        self.budget = budget_bytes
        self.pool = TexturePool(self.device)
        self.pipelines: dict[tuple, Any] = {}
        self.cache: OrderedDict[str, dict[str, Any]] = OrderedDict()  # job key -> output -> texture
        self._bytes = 0  # what `cache` holds
        self._partition: dict[str, str] = {}  # job key -> its cache partition (Job.partition)
        self._partition_bytes = dict.fromkeys(PARTITION_SHARE, 0)  # what each partition holds
        self._known: set[str] = set()  # keys cooked before: a key cooked again is stable (not after an edit)
        self._held: dict[str, set[str]] = {}  # owner -> keys whose textures it uses outside the cooks (`hold`)
        self._counted: set[str] = set()  # the owners whose held results count in the budget
        self._fence = None  # a texel read to wait for the GPU (`wait`)
        self._inflight = None  # the cook `cook_steps` is running, between two of its steps
        self._waits: queue.Queue | None = None  # the fences' helper thread's queue (`fence`)
        self._fence_src = None
        # Per-node timings: with `timing` on, the time of each job cooked goes in `timings` (job key -> milliseconds,
        # its last cook). GPU timestamps around its work where they work (some devices report them and give zeros),
        # else the job is submitted alone and waited for, less the wait of an idle GPU: much slower cooks.
        self.timing = False
        self.timings: dict[str, float] = {}
        self._timing_open: dict[str, float] = {}  # job key -> when its work started (ns, or seconds when waited)
        self._timestamps: bool | None = None  # `timestamps_work`
        self._idle_wait = 0.0  # seconds a wait takes with nothing to do
        self._truncates = None  # `half_store_truncates`
        self._constants: dict[tuple, Any] = {}
        self._luts: dict[str, Any] = {}
        self._uploads: dict[str, Any] = {}
        self._samplers: dict[int, Any] = {}  # tiling -> the inputs' bilinear sampler (`sampler`)

    # --- resources ----------------------------------------------------------------------------

    def storage_format(self, depth: str, gray: bool, alpha: bool = False) -> str:
        """`alpha`: a grayscale value with its alpha (PortType.GRAY_ALPHA), two channels."""
        fmt = DEPTHS[depth][3 if gray and alpha else 1 if gray else 2]
        return self.formats.get(fmt, fmt)

    def pipeline(self, node: Node, variant: Variant):
        key = (node.type, getattr(node, "pipeline_key", ""), variant)
        if key not in self.pipelines:
            d = self.device
            module = d.create_shader_module(label=node.type, code=node.shader_source(variant))
            layout = self._layout(node, variant, wgpu.ShaderStage.COMPUTE)
            self.pipelines[key] = d.create_compute_pipeline(
                label=node.type, layout=d.create_pipeline_layout(bind_group_layouts=[layout]),
                compute={"module": module, "entry_point": "main"})
        return self.pipelines[key]

    def _layout(self, node: Node, variant: Variant, stage):
        """The bindings of a node's shader as its header declares them (Node.shader_source): the uniform, the inputs,
        the outputs (write only), the LUTs, the sampler with `filtered`. Declared, not taken from the shader: a binding
        it does not read stays valid (an automatic layout leaves it out, and the bind group no longer matches)."""
        tex = {"sample_type": "float" if variant.filtered else "unfilterable-float"}
        entries = [{"binding": 0, "visibility": stage, "buffer": {"type": "uniform"}}]
        entries += [{"binding": 1 + i, "visibility": stage, "texture": tex} for i in range(len(variant.inputs))]
        entries += [{"binding": len(entries) + i, "visibility": stage, "storage_texture": {"access": "write-only",
                                                                                         "format": fmt}}
                    for i, fmt in enumerate(variant.outputs)]
        entries += [{"binding": len(entries) + i, "visibility": stage, "texture": tex}
                    for i in range(len(node.lut_params()))]
        if variant.filtered:
            entries.append({"binding": len(entries), "visibility": stage, "sampler": {"type": "filtering"}})
        entries += [{"binding": len(entries) + i, "visibility": stage, "buffer": {"type": "storage"}}
                    for i in range(len(node.buffers))]
        return self.device.create_bind_group_layout(entries=entries)

    def sampler(self, tiling: int):
        """The bilinear sampler of a node's inputs: repeating along the axes the node tiles, else at the edge (as
        tile_coord reads texels)."""
        if tiling not in self._samplers:
            mode = lambda bit: "repeat" if tiling & bit else "clamp-to-edge"
            self._samplers[tiling] = self.device.create_sampler(
                mag_filter="linear", min_filter="linear", address_mode_u=mode(1), address_mode_v=mode(2))
        return self._samplers[tiling]

    @property
    def filters(self) -> bool:
        """The device filters 32 bit float textures: every input reads through the sampler (Variant.filtered)."""
        return FILTERABLE_FEATURE in self.device.features

    def render_pipeline(self, node: Node, variant: Variant, fmt: str, blend: dict):
        """(pipeline, bind group layout) of a node's instanced quads (`CookContext.draw`), drawn into `fmt` with the
        `blend` state: its bindings as for a dispatch (`_layout`: the node has no outputs), each visible to both
        stages."""
        key = (node.type, getattr(node, "pipeline_key", ""), variant, fmt, repr(blend))
        if key not in self.pipelines:
            d = self.device
            bgl = self._layout(node, variant, wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT)
            module = d.create_shader_module(label=node.type, code=node.shader_source(variant))
            pipe = d.create_render_pipeline(
                label=node.type, layout=d.create_pipeline_layout(bind_group_layouts=[bgl]),
                vertex={"module": module, "entry_point": "vs"},
                fragment={"module": module, "entry_point": "fs", "targets": [{"format": fmt, "blend": blend}]},
                primitive={"topology": "triangle-list"})
            self.pipelines[key] = (pipe, bgl)
        return self.pipelines[key]

    def constant(self, value: tuple):
        """A 1x1 texture holding `value` (what an unconnected input reads)."""
        key = tuple(float(x) for x in value)
        if key not in self._constants:
            t = self.device.create_texture(size=(1, 1, 1), format="rgba32float",
                                           usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST)
            self.device.queue.write_texture({"texture": t}, np.array(key, np.float32).tobytes(),
                                            {"bytes_per_row": 16}, (1, 1, 1))
            self._constants[key] = t
        return self._constants[key]

    def lut(self, param, value):
        """A 256x1 lookup table baked from a gradient or curve value."""
        key = repr((param.kind, value))
        if key not in self._luts:
            t = self.device.create_texture(size=(LUT_SIZE, 1, 1), format="rgba32float",
                                           usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST)
            data = bake_lut(param.kind, value)
            self.device.queue.write_texture({"texture": t}, data.tobytes(), {"bytes_per_row": LUT_SIZE * 16},
                                            (LUT_SIZE, 1, 1))
            if len(self._luts) > 512:
                self._luts.clear()
            self._luts[key] = t
        return self._luts[key]

    def texture_of(self, pixels) -> Any:
        """A new texture of a CPU image (HxWx4, or HxWx1 for a grayscale one: one channel, `one_channel`): half floats
        as (r|rgba)16float, the rest as (r|rgba)32float."""
        half = np.asarray(pixels).dtype == np.float16
        a = np.ascontiguousarray(pixels, np.float16 if half else np.float32)
        h, w = a.shape[:2]
        one = a.ndim == 3 and a.shape[2] == 1
        fmt = ("r" if one else "rgba") + ("16float" if half else "32float")
        t = self.device.create_texture(size=(w, h, 1), format=fmt,
                                       usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST)
        self.device.queue.write_texture({"texture": t}, memoryview(a).cast("B"), {"bytes_per_row": w * BYTES_PER_PIXEL[fmt]},
                                        (w, h, 1))  # the array itself, not a copy (128 MB for an image at 4096)
        return t

    def upload(self, key: str, make) -> Any:
        """A texture made from a CPU image (HxWx4 from `make()`, `texture_of`), cached by `key` (what is bound outside a
        cook; a job reading an image into its result uploads it with `CookContext.upload`)."""
        if key not in self._uploads:
            t = self.texture_of(make())
            if len(self._uploads) > 64:
                for old in self._uploads.values():
                    old.destroy()
                self._uploads.clear()
            self._uploads[key] = t
        return self._uploads[key]

    def texture(self, job: Job, output: str):
        """The texture of a job's output (allocated when the job is being cooked)."""
        return self.cache[job.key][output]

    # --- cooking ------------------------------------------------------------------------------

    def cook(self, targets: list[Job], related: list[Job] = (), precious: frozenset = frozenset(),
             each=None) -> CookStats:
        """Make the targets' results current on the GPU (submitted, not waited for). `related`: the other targets of
        the same update, cooked apart (one tile at a time): their graph counts as used when the budget chooses what
        goes. `precious`: keys the caller expects the next cooks to read (a Paint edit's restart points), given up
        last. `each(ctx, job)`: records what reads a target as soon as its result is there, into the cook's own
        dispatches (a thumbnail); the targets are then kept like the other results, within the budget (the
        thumbnails of a whole graph at 4096 held every result at once: 20 GB)."""
        self.finish()
        return run(self._cook(targets, related, precious, each, turns=False))

    def cook_steps(self, targets: list[Job], related: list[Job] = (), precious: frozenset = frozenset(),
                   each=None):
        """`cook` as a generator that yields after each job, submitted, a `Fence` on its work: the caller lets its
        event loop run and resumes the cook once the fence is done (the GPU's work is what a large cook spends its
        time on; `cook` waits only to keep the memory bounded, after many jobs at once). Its return value
        (StopIteration) is the CookStats, None when `finish` completed it. One at a time: any other cook or eviction
        first completes it (`finish`); closing it at a yield abandons it (what it computed stays cached)."""
        self.finish()
        inner = self._cook(targets, related, precious, each, turns=True)
        self._inflight = inner
        try:
            stats = yield from inner
        finally:
            if self._inflight is inner:
                self._inflight = None
        return stats

    def _needs(self):
        """need(job): the memory its results take to compute, the cached ones free, its inputs computed the
        hungriest first: what the order of a cook keeps alive at once (as for an expression tree: the inputs computed
        first stay alive while the next are computed, so the larger needs go first). Ours, after Sethi and Ullman's
        register count; on a 357 node graph at 4096 it halved the peak of what a cook holds against the inputs' port
        order (2.46 -> 1.32 GB)."""
        memo: dict[str, int] = {}

        def size(job):
            return sum(job.size[0] * job.size[1] * BYTES_PER_PIXEL[self.storage_format(
                job.depths[o], job.is_gray(o), job.output_types[o] is PortType.GRAY_ALPHA)] for o in job.node.outputs)

        def need(job):
            if job.key in self.cache:
                return 0
            if job.key not in memo:
                held = top = 0
                for n, s in sorted(((need(src), size(src)) for src in _sources(job)), reverse=True):
                    top = max(top, held + n)
                    held += s
                memo[job.key] = max(top, held + size(job))
            return memo[job.key]
        return need

    def finish(self) -> None:
        """The cook in flight (`cook_steps`), if any, completed now."""
        inner, self._inflight = self._inflight, None
        if inner is None:
            return
        if inner.gi_running:
            raise RuntimeError("a cook started while another one is recording")
        for _ in inner:
            pass

    def _cook(self, targets: list[Job], related: list[Job], precious: frozenset, each, turns: bool):
        """`cook` (`turns` False: it never yields) and `cook_steps`."""
        t0 = time.perf_counter()
        schedule, seen, used = [], set(), set()  # `used`: every result the targets depend on, cooked or reused
        need = self._needs()

        def visit(job):
            used.add(job.key)
            if job.key in self.cache:  # reused: recently used for the LRU (else the next evictions take it)
                self.cache.move_to_end(job.key)
                if not job.partition:  # a preview result the full resolution shares (a small Texture Set)
                    self._move(job.key, "")
                return
            if job.key in seen:
                return
            seen.add(job.key)
            for src in sorted(_sources(job), key=need, reverse=True):  # the hungriest first (`_needs`)
                visit(src)
            schedule.append(job)

        for job in sorted(targets, key=need, reverse=True):
            visit(job)

        def touch(job):  # a related target's whole graph: used (what is not in it is stale), not scheduled
            if job.key in touched:
                return
            touched.add(job.key)
            used.add(job.key)
            for src in job.inputs.values():
                if src:
                    touch(src[0])

        touched = set()

        for job in related:
            touch(job)
        target_keys = {j.key for j in targets}
        held = set().union(*self._held.values())  # what a viewport or a stroke uses: never freed here
        chains = _chains(schedule, target_keys | held | set(precious))
        absorbed = {j.key for chain in chains.values() for j in chain[:-1]}
        schedule = [j for j in schedule if j.key not in absorbed]
        reads = {j.key: [src for c in chains.get(j.key, [j]) for src in c.inputs.values()
                         if src and src[0].key not in absorbed] for j in schedule}
        last_use = {}
        for i, job in enumerate(schedule):
            for src in reads[job.key]:
                last_use[src[0].key] = i
        new = {j.key for j in schedule if j.key not in self._known}
        self._known |= {j.key for j in schedule}
        ctx = CookContext(self)
        kept = set() if each else target_keys  # what the cook gives back only within the budget, at its end
        if each:
            scheduled = {j.key for j in schedule}
            for job in {j.key: j for j in targets if j.key not in scheduled}.values():  # reused from the cache
                each(ctx, job)
        dispatches = 0
        # Memory stays bounded while cooking: intermediates go after their last reader (their textures reused by
        # the next jobs: the dispatches are ordered), uploads are submitted and freed at once, and past the budget
        # the cache gives up what this cook does not read any more.
        for i, job in enumerate(schedule):
            w, h = job.size
            outs = {o: (self.pool.acquire("rgba32float", 1, 1) if job.is_value(o) else
                        self.pool.acquire(self.storage_format(job.depths[o], job.is_gray(o),
                                                              job.output_types[o] is PortType.GRAY_ALPHA), w, h))
                    for o in job.node.outputs}
            self.cache[job.key] = outs
            self._partition[job.key] = job.partition
            self._partition_bytes[job.partition] += _size(outs)
            self._bytes += _size(outs)
            ctx.mark(job.key, end=False)
            if job.key in chains:  # the chain's time is its last job's (the others are computed in its dispatch)
                job.node.encode_fused(ctx, chains[job.key])
            else:
                work = job.node.encode(ctx, job)
                if isinstance(work, GeneratorType):  # a node waiting for its own GPU work (a read back): its fences
                    if turns:
                        yield from work
                    else:
                        run(work)
            ctx.mark(job.key, end=True)
            ctx.release_scratches()
            if each and job.key in target_keys:
                each(ctx, job)
            for src in reads[job.key]:  # an intermediate freed after its last reader, unless someone holds it
                if src and last_use.get(src[0].key) == i and src[0].key not in kept \
                        and (self.budget == 0 or not src[0].keep) and src[0].key not in held:
                    self._evict(src[0].key)
            self._fit_budget(used, new, lambda k, i=i: k in kept or last_use.get(k, -1) > i, precious)
            if turns or ctx.transients or self.pool.bytes > min(self.budget // 4, POOL_MID_COOK):  # submitted, freed
                dispatches += ctx.submit()
                ctx = CookContext(self)
                # the GPU frees destroyed textures once their work is done: waiting for it keeps the memory bounded
                # (without it the queue ran gigabytes ahead)
                if turns:  # the caller's event loop runs while the GPU works: resumed once the fence is done
                    fence = self.fence()
                    self.pool.trim(fence=fence)  # destroyed after it, not waiting for the GPU here
                    yield fence
                    held = set().union(*self._held.values())  # a viewport may hold others now
                else:
                    self.pool.trim()
                    self.wait()
        dispatches += ctx.submit()
        for job in targets:
            if job.key in self.cache:
                self.cache.move_to_end(job.key)
        self._fit_budget(used, new, lambda k: k in kept, precious)
        self.pool.trim(fence=self.fence() if turns else None)
        return CookStats([j.path for j in schedule], dispatches, time.perf_counter() - t0)

    def half_store_truncates(self) -> bool:
        """Whether this GPU writes half float textures rounding toward zero (Apple's do) rather than to the nearest:
        what reproducing a stored result in a shader needs (BlendChain). Probed once, on a 1x1 texture."""
        if self._truncates is None:
            d = self.device
            v = 1.0 + 0.75 / 1024  # between two halves, nearer the larger one
            code = (f"@group(0) @binding(0) var dst: texture_storage_2d<rgba16float, write>;\n"
                    f"@compute @workgroup_size(1) fn main() {{ textureStore(dst, vec2i(0), vec4f({v!r})); }}")
            pipe = d.create_compute_pipeline(layout="auto", compute={
                "module": d.create_shader_module(code=code), "entry_point": "main"})
            t = d.create_texture(size=(1, 1, 1), format="rgba16float",
                                 usage=wgpu.TextureUsage.STORAGE_BINDING | wgpu.TextureUsage.COPY_SRC)
            enc = d.create_command_encoder()
            cp = enc.begin_compute_pass()
            cp.set_pipeline(pipe)
            cp.set_bind_group(0, d.create_bind_group(layout=pipe.get_bind_group_layout(0),
                                                     entries=[{"binding": 0, "resource": t.create_view()}]))
            cp.dispatch_workgroups(1)
            cp.end()
            d.queue.submit([enc.finish()])
            got = np.frombuffer(d.queue.read_texture({"texture": t}, {"bytes_per_row": 256}, (1, 1, 1))[:8], np.float16)
            self._truncates = bool(got[0] == np.float16(1.0))
            t.destroy()
        return self._truncates

    def wait(self) -> None:
        """Until the GPU has done the work submitted (a one texel read: queue.on_submitted_work_done_sync fails in
        this wgpu)."""
        if self._fence is None:
            self._fence = self.device.create_texture(size=(1, 1, 1), format="r8unorm",
                                                     usage=wgpu.TextureUsage.COPY_SRC)
        self.device.queue.read_texture({"texture": self._fence, "origin": (0, 0, 0)}, {"bytes_per_row": 256},
                                       (1, 1, 1))

    def fence(self) -> Fence:
        """The work submitted so far, asked whether done without waiting (`Fence`): a 4 byte copy after it into a
        buffer whose mapping a helper thread waits for. What lets a cook in steps give the event loop the time the GPU
        works (waiting for it on the interface thread stalled it for tens to hundreds of milliseconds a job)."""
        if self._waits is None:
            self._waits = queue.Queue()
            threading.Thread(target=_waiter, args=(self._waits,), daemon=True).start()
            self._fence_src = self.device.create_buffer(size=4, usage=wgpu.BufferUsage.COPY_SRC)
        buffer = self.device.create_buffer(size=4, usage=wgpu.BufferUsage.MAP_READ | wgpu.BufferUsage.COPY_DST)
        encoder = self.device.create_command_encoder()
        encoder.copy_buffer_to_buffer(self._fence_src, 0, buffer, 0, 4)
        self.device.queue.submit([encoder.finish()])
        event = threading.Event()
        self._waits.put((buffer.map_async("READ"), event))
        return Fence(buffer, event)

    def timestamps_work(self) -> bool:
        """Whether this device's GPU timestamps measure anything, probed once on a small busy pass (some report the
        features and write zeros). Without them the timings wait for each job: `_idle_wait` measured then. Not probed
        on Apple GPUs: they report the features, but timestamps between passes are not supported there (wgpu: "Metal
        (AMD & Intel, not Apple GPUs)"), and with the editor's views on screen the probe's work never completed and
        every later wait hung."""
        if self._timestamps is None:
            self._timestamps = False
            info = self.device.adapter.info
            apple = info["backend_type"] == "Metal" and info["device"].startswith("Apple")
            if not apple and all(f in self.device.features for f in TIMESTAMP_FEATURES):
                d = self.device
                code = ("@group(0) @binding(0) var<storage, read_write> b: array<f32>;\n"
                        "@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) i: vec3u) {\n"
                        "  var x = f32(i.x); for (var k = 0; k < 256; k++) { x = sin(x); } b[i.x] = x; }")
                pipe = d.create_compute_pipeline(layout="auto", compute={
                    "module": d.create_shader_module(code=code), "entry_point": "main"})
                buf = d.create_buffer(size=4 * 64 * 256, usage=wgpu.BufferUsage.STORAGE)
                stamps = d.create_query_set(type="timestamp", count=2)
                enc = d.create_command_encoder()
                write_timestamp(enc, stamps, 0)
                cp = enc.begin_compute_pass()
                cp.set_pipeline(pipe)
                cp.set_bind_group(0, d.create_bind_group(layout=pipe.get_bind_group_layout(0),
                                                         entries=[{"binding": 0, "resource": {"buffer": buf}}]))
                cp.dispatch_workgroups(256)
                cp.end()
                write_timestamp(enc, stamps, 1)
                resolved = d.create_buffer(size=16, usage=wgpu.BufferUsage.QUERY_RESOLVE | wgpu.BufferUsage.COPY_SRC)
                enc.resolve_query_set(stamps, 0, 2, resolved, 0)
                d.queue.submit([enc.finish()])
                a, b = np.frombuffer(d.queue.read_buffer(resolved), np.uint64)
                self._timestamps = bool(b > a > 0)
            if not self._timestamps:
                waits = []
                for _ in range(5):
                    t0 = time.perf_counter()
                    self.wait()
                    waits.append(time.perf_counter() - t0)
                self._idle_wait = min(waits)
        return self._timestamps

    def recook_region(self, jobs: list[Job], rect: tuple[int, int, int, int], full: set, whole: set,
                      skip: set = frozenset(), passed: set = frozenset()) -> None:
        """Compute again these jobs' results (in order, inputs first) where only the region `rect` (x0, y0, x1, y1)
        changed upstream. The `full` ones are cached results, written in place; the others write textures of the
        region only, given back as soon as read for the last time (a long chain costs what the region does, not what
        the texture does). The `whole` ones (a filter reading far, and what depends on it: the change spreads) are
        computed over all of their results, and read only `full` results. The `skip` ones keep their results (full);
        the `passed` ones are not computed either, their readers read their main input instead (a filter seen
        through). The keys are not changed: the caller knows these results are stale."""
        self.finish()
        ctx = CookContext(self)
        # The region computed is `rect` with a halo around, in whole workgroups; the whole results are written only
        # inside `rect` (ctx.clip): there the texels around (the normal reads its neighbors) were computed too.
        # Without it the edge of a region wrote wrong texels over what an earlier frame had made right.
        x0, y0, x1, y1 = rect
        size = max((j.size for j in jobs), default=(x1, y1))
        g = WORKGROUP
        ctx.clip = rect
        rect = (max(x0 - HALO, 0) // g * g, max(y0 - HALO, 0) // g * g, min(-(-(x1 + HALO) // g) * g, size[0]),
                min(-(-(y1 + HALO) // g) * g, size[1]))
        # the region textures in sizes of powers of two (larger than the region: the pool reuses them frame after
        # frame, a region of its own size each frame piled new ones up)
        w, h = (1 << max(rect[2] - rect[0] - 1, 1).bit_length(), 1 << max(rect[3] - rect[1] - 1, 1).bit_length())
        for job in jobs:
            if job.key in passed:
                ctx.alias[job.key] = job.inputs[job.node.primary_input()]

        def resolved(src):
            while src[0].key in ctx.alias:
                src = ctx.alias[src[0].key]
            return src[0].key

        last = {resolved(src): i for i, job in enumerate(jobs) if job.key not in passed
                for src in job.inputs.values() if src}
        for i, job in enumerate(jobs):
            if job.key in skip or job.key in passed:
                continue
            if job.key not in full:
                ctx.regional[job.key] = {
                    o: self.pool.acquire(self.storage_format(job.depths[o], job.is_gray(o),
                                                             job.output_types[o] is PortType.GRAY_ALPHA), w, h)
                    for o in job.node.outputs}
            elif job.key not in self.cache:
                continue
            ctx.region = None if job.key in whole else rect
            work = job.node.encode(ctx, job)
            if isinstance(work, GeneratorType):  # its fences waited for (`_cook`)
                run(work)
            ctx.release_scratches()
            for src in job.inputs.values():  # a region read for the last time goes back to the pool: the next jobs
                key = resolved(src) if src else None  # reuse it (ordered dispatches)
                if key is not None and last.get(key) == i and key in ctx.regional:
                    for t in ctx.regional.pop(key).values():
                        self.pool.release(t)
        ctx.submit()
        for outs in ctx.regional.values():  # read by none of these jobs
            for t in outs.values():
                self.pool.release(t)
        self.pool.trim(POOL_MID_COOK)

    def drop(self, jobs) -> None:
        """Results the caller has copied (a renderer's textures): back to the pool now, not left to fill the budget
        until the next cooks."""
        self.finish()
        for job in jobs:
            self._evict(job.key)

    def forget(self, jobs) -> None:
        """Results computed again by the next cook (resetting the timings), but those a view holds."""
        self.finish()
        held = set().union(*self._held.values())
        for job in jobs:
            if job.key not in held:
                self._evict(job.key)

    def hold(self, owner: str, jobs, counted: bool = False) -> None:
        """`owner` uses these jobs' textures outside the cooks (a viewport draws them): the cache keeps them until
        the owner holds others, outside its budget (they are in use, not kept for later); `counted`: inside it (the
        others make room: a painted stroke's inputs, which can be gigabytes)."""
        self._held[owner] = {j.key for j in jobs}
        if counted:
            self._counted.add(owner)
        else:
            self._counted.discard(owner)

    def result(self, job: Job, output: str):
        return self.results([(job, output)])[0]

    def results(self, items: list[tuple[Job, str]]) -> list:
        """The textures of job outputs used together (a 3D view's channels), cooked together: one by one, each cook
        could give back the previous ones (their textures recycled or destroyed before being read)."""
        if any(job.key not in self.cache for job, _ in items):
            self.cook([job for job, _ in items])  # the cached ones are targets too: kept
        for job, _ in items:
            self.cache.move_to_end(job.key)
        return [self.cache[job.key][out] for job, out in items]

    def read(self, job: Job, output: str) -> np.ndarray:
        """A job output as a (height, width, channels) array in its storage type (16 bit integers as the float32 values
        k/65535 the shaders read, whatever the device stores them in); waits for the GPU. A grayscale value with its
        alpha is read as the shaders read it: (v, v, v, a)."""
        t = self.result(job, output)
        dtype, channels = NUMPY[t.format]
        bpp = BYTES_PER_PIXEL[t.format]
        data = self.device.queue.read_texture({"texture": t, "origin": (0, 0, 0)},
                                              {"bytes_per_row": t.width * bpp}, (t.width, t.height, 1))
        return _unpacked(np.frombuffer(data, dtype).reshape(t.height, t.width, channels), t.format)

    def read_pixel(self, job: Job, output: str, u: float, v: float) -> np.ndarray:
        """One texel of a job output at texture coordinates (u, v) in 0-1 (v down), as `read` gives it (a grayscale
        value with its alpha as (v, v, v, a))."""
        t = self.result(job, output)
        dtype, channels = NUMPY[t.format]
        x, y = min(int(u * t.width), t.width - 1), min(int(v * t.height), t.height - 1)
        data = self.device.queue.read_texture({"texture": t, "origin": (max(x, 0), max(y, 0), 0)},
                                              {"bytes_per_row": BYTES_PER_PIXEL[t.format]}, (1, 1, 1))
        return _unpacked(np.frombuffer(data, dtype)[:channels], t.format).copy()

    def thumbnails(self, items: list[tuple[Job, str]], size: int = 128, related: list[Job] = (),
                   precious: frozenset = frozenset()) -> list[np.ndarray]:
        """Small RGBA8 images (size x size, aspect kept inside) of job outputs, cooked if needed,
        downsampled on the GPU (each in the cook, right after its result) and read back together. `related`,
        `precious`: as `cook`'s."""
        return run(self.thumbnail_steps(items, size, related, precious))

    def thumbnail_steps(self, items: list[tuple[Job, str]], size: int = 128, related: list[Job] = (),
                        precious: frozenset = frozenset()):
        """`thumbnails` as a generator, a step per wait of its cook (`cook_steps`)."""
        from trafilatura3d.nodes.thumbnail import THUMBNAIL
        wanted: dict[str, set[str]] = {}  # job key -> its outputs shown
        for job, out in items:
            wanted.setdefault(job.key, set()).add(out)
        made = {}  # (job key, output) -> thumbnail texture

        def thumbnail(ctx, job):  # recorded as soon as the result is there: the cache keeps to its budget
            w, h = job.size
            k = size / max(w, h)
            tw, th = max(1, round(w * k)), max(1, round(h * k))
            for out in wanted[job.key]:
                t = made[job.key, out] = self.pool.acquire("rgba8unorm", tw, th)
                ctx.dispatch(THUMBNAIL, job, {}, srcs={"image": (self.texture(job, out), job.is_gray(out))},
                             dsts={"output": t}, size=(tw, th))

        try:
            yield from self.cook_steps([job for job, _ in items], related=related, precious=precious, each=thumbnail)
            yield self.fence()  # read back once the GPU is done: reading earlier waits for it on this thread
        except GeneratorExit:  # abandoned: the thumbnails recorded so far back to the pool
            for t in made.values():
                self.pool.release(t)
            raise
        read = {}
        for key, t in made.items():
            row = (t.width * 4 + 255) // 256 * 256
            data = self.device.queue.read_texture({"texture": t, "origin": (0, 0, 0)},
                                                  {"bytes_per_row": row}, (t.width, t.height, 1))
            read[key] = np.frombuffer(data, np.uint8).reshape(t.height, row)[:, :t.width * 4].reshape(
                t.height, t.width, 4).copy()
            self.pool.release(t)
        return [read[job.key, out] for job, out in items]

    def histogram(self, job: Job, output: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """A job output's histogram on the GPU, over every texel, cooked if needed: (counts (BINS, 4) per channel of
        the values in 0-1, those outside in the end bins; minimum (4,); maximum (4,)). A grayscale output is read as
        (v, v, v, 1)."""
        from trafilatura3d.nodes.histogram import BINS, GROUPS, HISTOGRAM
        rows = min(GROUPS, job.size[1])
        made = {}

        def histogram(ctx, target):
            t = made["t"] = self.pool.acquire("rgba32float", BINS + 2, rows)
            ctx.dispatch(HISTOGRAM, target, {}, srcs={"image": (self.texture(target, output), target.is_gray(output))},
                         dsts={"output": t}, size=(WORKGROUP * rows, WORKGROUP))

        self.cook([job], each=histogram)
        t = made["t"]
        row = (t.width * 16 + 255) // 256 * 256
        data = self.device.queue.read_texture({"texture": t, "origin": (0, 0, 0)}, {"bytes_per_row": row},
                                              (t.width, t.height, 1))
        self.pool.release(t)
        texels = np.frombuffer(data, np.float32).reshape(t.height, row // 4)[:, :t.width * 4].reshape(
            t.height, t.width, 4)
        return texels[:, :BINS].sum(axis=0), texels[:, BINS].min(axis=0), texels[:, BINS + 1].max(axis=0)

    # --- cache --------------------------------------------------------------------------------

    def _evict(self, key: str) -> None:
        outs = self.cache.pop(key, {})
        if key in self._partition:
            self._partition_bytes[self._partition.pop(key)] -= _size(outs)
        self._bytes -= _size(outs)
        for t in outs.values():
            self.pool.release(t)

    def cached_bytes(self) -> int:
        return self._bytes

    def _over_budget(self, partition: str) -> bool:
        """A partition's cached results beyond its share of the budget, the held ones not counted (but those held
        `counted`)."""
        held = set().union(*(keys for owner, keys in self._held.items() if owner not in self._counted))
        size = self._partition_bytes[partition] - sum(_size(self.cache[k]) for k in held
                                                      if k in self.cache and self._partition[k] == partition)
        return size > self.budget * PARTITION_SHARE[partition]

    def _move(self, key: str, partition: str) -> None:
        size = _size(self.cache[key])
        self._partition_bytes[self._partition[key]] -= size
        self._partition[key] = partition
        self._partition_bytes[partition] += size

    def _fit_budget(self, used: set[str], new: set[str], needed, precious=frozenset()) -> None:
        """Every partition back within its budget, never taking what is `needed` or held: first what these targets
        do not depend on, then what this cook computed for the first time (downstream of an edit: the next edit
        computes it again), then what proved stable, last what is `precious`; the least recently used first in
        each."""
        held = set().union(*self._held.values())
        for partition in PARTITION_SHARE:
            if not self._over_budget(partition):
                continue
            order = sorted((k for k in self.cache if self._partition[k] == partition),
                           key=lambda k: (k in used, k in precious, k in used and k not in new))  # LRU within
            for key in order:
                if not self._over_budget(partition):
                    break
                if key not in held and not needed(key):
                    self._evict(key)

    def clear(self) -> None:
        self.finish()
        for key in list(self.cache):
            self._evict(key)
        self.pool.trim()

    def drop_partition(self, partition: str) -> None:
        """A partition's results freed now, but those held (a viewport shows them): the preview's once the full
        resolution is shown (kept, they stayed in memory until the next drag)."""
        self.finish()
        held = set().union(*self._held.values())
        for key in [k for k, p in self._partition.items() if p == partition and k not in held]:
            self._evict(key)
        self.pool.trim(fence=self.fence())  # the viewport's last refresh may still be using them


def one_channel(texture) -> bool:
    """A texture holding a grayscale value only (r8unorm, r16float…): read as gray (`(texture, True)` sources)."""
    return texture.format.startswith("r") and not texture.format.startswith("rg")


def _unpacked(a: np.ndarray, fmt: str) -> np.ndarray:
    """A grayscale value with its alpha (two channels) as (v, v, v, a); other formats as they are, but 16 bit
    integers as their values k/65535 in float32 (what a device without them stores)."""
    if fmt in NORM16:
        a = a.astype(np.float32) / np.float32(65535)
    return np.concatenate([a[..., :1]] * 3 + [a[..., 1:2]], -1) if two_channels(fmt) else a


def _size(outs: dict) -> int:
    return sum(t.width * t.height * BYTES_PER_PIXEL[t.format] for t in outs.values())


def bake_lut(kind: ParamKind, value) -> np.ndarray:
    """Gradient: linear interpolation between keys. Curve: monotone cubic through the points."""
    t = np.linspace(0.0, 1.0, LUT_SIZE)
    out = np.zeros((LUT_SIZE, 4), np.float32)
    if kind is ParamKind.GRADIENT:
        keys = value["keys"]
        pos = np.array([k["pos"] for k in keys])
        colors = np.array([k["color"] for k in keys])
        mode = value["interpolation"]
        if mode == "none" or len(keys) == 1:  # each key's color holds until the next key
            idx = np.clip(np.searchsorted(pos, t, side="right") - 1, 0, len(keys) - 1)
            return colors[idx].astype(np.float32)
        tc = np.clip(t, pos[0], pos[-1])
        i = np.clip(np.searchsorted(pos, tc, side="right") - 1, 0, len(keys) - 2)
        span = np.maximum(pos[i + 1] - pos[i], 1e-9)
        s = np.clip((tc - pos[i]) / span, 0.0, 1.0)
        if mode == "flat":  # horizontal tangents at the keys, Midpoint moves the halfway point
            mid = min(max(value["midpoint"], 1e-3), 1 - 1e-3)
            s = s ** (np.log(0.5) / np.log(mid))
            s = s * s * (3 - 2 * s)
        w = s[:, None]
        out[:] = colors[i] * (1 - w) + colors[i + 1] * w
        if mode == "smooth":  # a smooth curve through the keys, Smoothness 0 = linear
            for c in range(4):
                out[:, c] = (1 - value["smoothness"]) * out[:, c] + value["smoothness"] * _monotone_cubic(
                    pos, colors[:, c], tc)
    else:
        out[:, :3] = curve_values(value, t)[:, None]
        out[:, 3] = 1.0
    return out


def curve_values(points, t) -> np.ndarray:
    """A curve's values at t: a cubic Bezier between two points that both carry handles ([x, y, left x, left y,
    right x, right y], absolute), the monotone cubic through the points elsewhere; flat outside the first and last."""
    t = np.asarray(t, np.float64)
    x = np.array([pt[0] for pt in points], np.float64)
    y = np.array([pt[1] for pt in points], np.float64)
    out = _monotone_cubic(x, y, t)
    tc = np.clip(t, x[0], x[-1])
    for i in range(len(points) - 1):
        a, b = points[i], points[i + 1]
        if len(a) < 6 or len(b) < 6:
            continue
        inside = (tc >= x[i]) & (tc <= x[i + 1])
        if inside.any():
            out[inside] = _bezier_y(a[:2], a[4:6], b[2:4], b[:2], tc[inside])
    return out


def _bezier_y(p0, p1, p2, p3, xs):
    """y of the cubic Bezier p0 p1 p2 p3 where its x is xs (x(s) found by bisection on s in [0, 1])."""
    def coord(k, s):
        u = 1.0 - s
        return u * u * u * p0[k] + 3 * u * u * s * p1[k] + 3 * u * s * s * p2[k] + s * s * s * p3[k]
    lo, hi = np.zeros_like(xs), np.ones_like(xs)
    for _ in range(40):
        mid = (lo + hi) * 0.5
        right = coord(0, mid) < xs
        lo, hi = np.where(right, mid, lo), np.where(right, hi, mid)
    return coord(1, (lo + hi) * 0.5)


def _monotone_cubic(x, y, t):
    """Fritsch-Carlson monotone cubic interpolation (no overshoot between control points)."""
    n = len(x)
    if n == 2:
        return np.interp(t, x, y)
    h = np.diff(x)
    h[h == 0] = 1e-9
    delta = np.diff(y) / h
    m = np.zeros(n)
    m[1:-1] = (delta[:-1] + delta[1:]) / 2
    m[0], m[-1] = delta[0], delta[-1]
    for i in range(n - 1):
        if delta[i] == 0:
            m[i] = m[i + 1] = 0
        else:
            a, b = m[i] / delta[i], m[i + 1] / delta[i]
            s = a * a + b * b
            if s > 9:
                k = 3 / np.sqrt(s)
                m[i], m[i + 1] = k * a * delta[i], k * b * delta[i]
    tc = np.clip(t, x[0], x[-1])
    i = np.clip(np.searchsorted(x, tc) - 1, 0, n - 2)
    s = (tc - x[i]) / h[i]
    h00, h10 = 2 * s**3 - 3 * s**2 + 1, s**3 - 2 * s**2 + s
    h01, h11 = -2 * s**3 + 3 * s**2, s**3 - s**2
    return h00 * y[i] + h10 * h[i] * m[i] + h01 * y[i + 1] + h11 * h[i] * m[i + 1]
