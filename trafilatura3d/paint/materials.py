"""Materials for Paint (Material mode of fill layers and fill effects): Trafilatura 3D graphs whose Output nodes have
channel usages (baseColor, normal, roughness…) (decision 23: the graphs appear in Paint where it has
materials). A material is referenced by its package file and graph
(linked, like a library resource) with the values of its exposed parameters; it is compiled at the Texture
Set's size and each output feeds the channel of its usage (outputs without a usage: by identifier).

Where the materials come from: the packages of the folders the caller gives (the editor: its libraries and the
library folders of the project configuration).
"""

from __future__ import annotations

import json
from pathlib import Path

from trafilatura3d.compiler import application_context, compile_graph
from trafilatura3d.package import SUFFIX, DocumentError, Package

# Graph usage -> Paint channel (both spellings).
USAGE_TO_CHANNEL = {
    "baseColor": "basecolor", "basecolor": "basecolor", "diffuse": "diffuse", "normal": "normal",
    "roughness": "roughness", "metallic": "metallic", "height": "height", "ambientOcclusion": "ambientocclusion",
    "ambientocclusion": "ambientocclusion", "emissive": "emissive", "opacity": "opacity",
    "specularLevel": "specularlevel", "specularlevel": "specularlevel", "specular": "specular",
    "glossiness": "glossiness", "anisotropyLevel": "anisotropylevel", "anisotropyAngle": "anisotropyangle",
    "displacement": "displacement", "scattering": "scattering", "ior": "ior", "transmissive": "transmissive",
}
EXAMPLES = Path(__file__).resolve().parent.parent.parent / "examples"


class Materials:
    """Loaded packages (by path and modification time) and compiled materials (by everything that decides
    their pixels), so a stack refresh does not recompile them."""

    MAX_TEXTURES = 8  # materials kept cooked (their output textures)

    def __init__(self):
        self._packages: dict[str, tuple[float, Package]] = {}
        self._compiled: dict[str, dict] = {}
        self._textures: dict[str, dict] = {}  # material key -> {channel: (texture, gray)}

    def package(self, path: str) -> Package:
        p = Path(path)
        mtime = p.stat().st_mtime
        cached = self._packages.get(str(p))
        if cached is None or cached[0] != mtime:
            cached = (mtime, Package.load(p))
            self._packages[str(p)] = cached
        return cached[1]

    def channels_of(self, path: str, graph: str) -> dict[str, str]:
        """{channel: output identifier} of a material graph."""
        g = self.package(path).graphs[graph]
        out = {}
        for node in g.output_nodes():
            ident = node.params.get("identifier", node.id)
            usage = node.params.get("usage", "")
            ch = USAGE_TO_CHANNEL.get(usage) or USAGE_TO_CHANNEL.get(ident)
            if ch and ch not in out:
                out[ch] = ident
        return out

    def available(self, folders) -> list[tuple[str, str, str]]:
        """(package path, graph id, label) of every material graph in the folders."""
        out = []
        for folder in map(Path, folders):
            if not folder.is_dir():
                continue
            for path in sorted(folder.rglob(f"*{SUFFIX}")):
                if ".autosave" in path.parts:
                    continue
                try:
                    pkg = self.package(str(path))
                except (DocumentError, OSError, ValueError, KeyError):
                    continue
                for gid, g in pkg.graphs.items():
                    try:
                        if self.channels_of(str(path), gid):
                            out.append((str(path), gid, g.label or gid))
                    except (DocumentError, KeyError):
                        continue
        return out

    def outputs(self, ref: dict, size, evaluator) -> dict:
        """{channel: (job, output)} of a material reference ({"package", "graph", "values"}) at `size`."""
        path, gid, values = ref["package"], ref["graph"], ref.get("values", {})
        mtime = Path(path).stat().st_mtime
        key = json.dumps([path, mtime, gid, values, list(size)], sort_keys=True, default=str)
        if key not in self._compiled:
            pkg = self.package(path)
            log2 = tuple(max(int(s).bit_length() - 1, 0) for s in size)
            compiled = compile_graph(pkg, gid, parent=application_context(log2), evaluator=evaluator, values=values)
            by_ident = compiled.outputs
            if len(self._compiled) > 32:  # bounded: the oldest compiled material goes
                self._compiled.pop(next(iter(self._compiled)))
            self._compiled[key] = {ch: by_ident[ident] for ch, ident in self.channels_of(path, gid).items()
                                   if ident in by_ident}
        return self._compiled[key]

    def textures(self, ref: dict, size, evaluator) -> tuple[str, dict]:
        """(key, {channel: (texture, is_gray)}) of a material cooked at `size`: its outputs copied into textures
        of their own and the graph's intermediate results freed from the evaluator's cache, so a 4096 material
        does not keep dozens of 4096 textures nor gets cooked again when the cache is full."""
        path, gid, values = ref["package"], ref["graph"], ref.get("values", {})
        key = json.dumps([path, Path(path).stat().st_mtime, gid, values, list(size)], sort_keys=True, default=str)
        if key not in self._textures:
            import wgpu
            pkg = self.package(path)
            log2 = tuple(max(int(s).bit_length() - 1, 0) for s in size)
            compiled = compile_graph(pkg, gid, parent=application_context(log2), evaluator=evaluator, values=values)
            outs = {ch: compiled.outputs[ident] for ch, ident in self.channels_of(path, gid).items()
                    if ident in compiled.outputs}
            evaluator.cook([job for job, _o in outs.values()])
            device = evaluator.device
            enc = device.create_command_encoder()
            made = {}
            for ch, (job, out) in outs.items():
                src = evaluator.texture(job, out)
                t = device.create_texture(size=(src.width, src.height, 1), format=src.format,
                                          usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST)
                enc.copy_texture_to_texture({"texture": src}, {"texture": t}, (src.width, src.height, 1))
                made[ch] = (t, job.is_gray(out))
            device.queue.submit([enc.finish()])
            for job in compiled.jobs.values():
                evaluator._evict(job.key)
            if len(self._textures) >= self.MAX_TEXTURES:
                for t, _g in self._textures.pop(next(iter(self._textures))).values():
                    t.destroy()
            self._textures[key] = made
        return key, self._textures[key]

    def parameters(self, ref: dict):
        """The material's exposed parameters (package.ExposedParam), in order."""
        return list(self.package(ref["package"]).graphs[ref["graph"]].parameters.values())


def normal_is_directx() -> bool:
    """The normal format of Trafilatura 3D graphs (project settings > General: default normal format)."""
    from trafilatura3d.project import configuration
    return configuration().setting("general", "normal_format") == "DirectX"
