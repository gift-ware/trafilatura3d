"""The 3D View's scene: mesh objects (with submeshes on materials), materials (a
shader of shaders.TABLE, its parameters and which graph output feeds each channel), the legacy ambient light and
two point lights, the camera, plus state files, 3D scene import/export and picking.

Formats: OBJ, glTF/GLB, PLY, STL, OFF through trimesh; USD/USDA/USDC/USDZ through Pixar's usd-core.
FBX needs Autodesk's proprietary SDK and is not available.
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from trafilatura3d import meshio
from trafilatura3d import shaders
from trafilatura3d.editor import meshes
from trafilatura3d.meshio import MeshObject, Submesh

AUTO, NONE = "auto", "none"  # channel source: the graph output with that usage / nothing
EXPORT_FORMATS = ("obj", "glb", "gltf", "ply", "stl", "usd", "usda", "usdc", "usdz")
STATE_SUFFIX = ".trafilatura3dscene"
STATE_VERSION = 2  # 2: materials with a shader of shaders.TABLE and its parameter values


@dataclass
class Material:
    """A scene material: its shader (shaders.TABLE), where each of the shader's channels comes from, and the values
    of the shader's parameters the user changed (the others are the shader's defaults)."""
    name: str = "Default material"
    shader: str = shaders.DEFAULT
    sources: dict = field(default_factory=dict)  # channel (usage) -> AUTO | NONE | node id (missing = AUTO)
    values: dict = field(default_factory=dict)  # parameter key -> value (missing = the shader's default)

    def source(self, channel):
        return self.sources.get(channel, AUTO)

    def value(self, key):
        p = next(p for p in shaders.BY_NAME[self.shader].params if p.key == key)
        return self.values.get(key, p.default)

    def spec(self) -> dict:
        """The renderer's material spec without textures: shader index, uniforms, slot constants."""
        return {"shader": shaders.index(self.shader), "uniforms": shaders.uniforms(self.shader, self.values),
                "constants": shaders.constants(self.shader, self.values)}


@dataclass
class PointLight:
    enabled: bool = False
    position: list = field(default_factory=lambda: [2.0, 2.5, 2.0])
    color: list = field(default_factory=lambda: [1.0, 1.0, 1.0])
    intensity: float = 10.0


@dataclass
class Lights:
    ambient_color: list = field(default_factory=lambda: [1.0, 1.0, 1.0])
    ambient_intensity: float = 0.0
    points: list = field(default_factory=lambda: [PointLight(), PointLight(position=[-2.5, 1.5, -1.5],
                                                                          color=[0.8, 0.9, 1.0])])


@dataclass
class CameraProps:
    fov: float = 40.0
    near: float = 0.05
    far: float = 100.0
    render_width: int = 1920
    render_height: int = 1080


@dataclass
class SceneCamera:
    name: str
    eye: list
    target: list
    fov: float


class Scene:
    def __init__(self):
        self.objects: list[MeshObject] = []
        self.materials: list[Material] = [Material()]
        self.lights = Lights()
        self.camera = CameraProps()
        self.cameras: list[SceneCamera] = []  # presets from a loaded scene
        self.source = "primitive:Rounded Cube"  # what the meshes come from
        self.fit = (np.zeros(3), 1.0)  # a loaded scene's transform into the view (meshio.fit)
        self.scene_materials: list[Material] = []  # materials as loaded (Reset material to scene state)

    # --- meshes -------------------------------------------------------------------------------

    def set_primitive(self, name):
        self.objects = [_object(name, *meshes.PRIMITIVES[name](), material=0)]
        self.materials = self.materials[:1] or [Material()]
        self.scene_materials = []
        self.cameras = []
        self.source = f"primitive:{name}"
        self.fit = (np.zeros(3), 1.0)

    def load(self, path):
        objects, materials, cameras, self.fit = load_scene(path)
        self.objects, self.materials, self.cameras = objects, materials, cameras
        self.scene_materials = copy.deepcopy(materials)
        self.source = str(path)

    @property
    def mesh_name(self):
        return self.source.split(":", 1)[1] if self.source.startswith("primitive:") else Path(self.source).name

    def bounds(self):
        vis = [o for o in self.objects if o.visible] or self.objects
        lo = np.min([o.bounds()[0] for o in vis], 0)
        hi = np.max([o.bounds()[1] for o in vis], 0)
        return lo, hi

    def add_material(self):
        n = len(self.materials) + 1
        self.materials.append(Material(name=f"Material {n}"))

    # --- state files --------------------------------------------------------------------------

    def state(self, view) -> dict:
        """The scene state with the camera orbit and the renderer look of `view` (a Viewport3D)."""
        return {"trafilatura3d_scene": STATE_VERSION, "source": self.source,
                "materials": [asdict(m) for m in self.materials],
                "lights": asdict(self.lights), "camera": asdict(self.camera),
                "orbit": {"yaw": view.yaw, "pitch": view.pitch, "distance": view.distance,
                          "target": [float(x) for x in view.target]},
                "look": asdict(view.look),
                "visible": [o.visible for o in self.objects]}

    def apply_state(self, data: dict, view, with_mesh=False):
        if data.get("trafilatura3d_scene") != STATE_VERSION:
            raise ValueError("not a Trafilatura 3D 3D scene state")
        if with_mesh:
            src = data["source"]
            self.set_primitive(src.split(":", 1)[1]) if src.startswith("primitive:") else self.load(src)
        self.materials = [Material(**m) for m in data["materials"]] or [Material()]
        lights = data["lights"]
        self.lights = Lights(lights["ambient_color"], lights["ambient_intensity"],
                             [PointLight(**p) for p in lights["points"]])
        self.camera = CameraProps(**data["camera"])
        o = data["orbit"]
        view.yaw, view.pitch, view.distance = o["yaw"], o["pitch"], o["distance"]
        view.target = np.array(o["target"], np.float32)
        for k, v in data["look"].items():
            if hasattr(view.look, k) and k != "preview":
                setattr(view.look, k, v)
        for obj, vis in zip(self.objects, data.get("visible", [])):
            obj.visible = vis

    def save_state(self, path, view):
        Path(path).write_text(json.dumps(self.state(view), indent=1), encoding="utf-8")

    # --- picking ------------------------------------------------------------------------------

    def pick(self, origin, direction):
        """The index of the nearest visible object hit by the ray, or None."""
        best, hit = math.inf, None
        for i, o in enumerate(self.objects):
            if not o.visible:
                continue
            t = _ray_triangles(origin, direction, o.pos[o.triangles])
            if t < best:
                best, hit = t, i
        return hit


def _object(name, pos, nrm, tan, uv, idx, material=0):
    return MeshObject(name, pos, nrm, tan, uv, idx, [Submesh(0, idx.size, material)])


def _ray_triangles(o, d, tri):
    """Nearest positive hit distance of a ray against (n, 3, 3) triangles (Moller-Trumbore)."""
    e1, e2 = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    p = np.cross(d, e2)
    det = np.sum(e1 * p, -1)
    ok = np.abs(det) > 1e-9
    inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
    s = o - tri[:, 0]
    u = np.sum(s * p, -1) * inv
    q = np.cross(s, e1)
    v = np.sum(d * q, -1) * inv
    t = np.sum(e2 * q, -1) * inv
    hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 1e-5)
    return float(t[hit].min()) if hit.any() else math.inf


# --- import ---------------------------------------------------------------------------------------

def load_scene(path) -> tuple[list[MeshObject], list[Material], list[SceneCamera], tuple]:
    """A 3D scene file for the 3D View: meshes fitted to the primitives' size (cameras follow), a
    material per name, and the fit transform (center, scale) for meshes shown with them."""
    objects, names, cams = meshio.load(path)
    center, k = meshio.fit(objects)
    moved = lambda p: [float(x) for x in (np.asarray(p, np.float64) - center) * k]
    return objects, [Material(name=n) for n in names], \
        [SceneCamera(c["name"], moved(c["eye"]), moved(c["target"]), c["fov"]) for c in cams], (center, k)

# --- export ---------------------------------------------------------------------------------------

def export_scene(path, objects, materials, textures=None):
    """Write the visible meshes (positions already displaced by the caller if wanted). `textures`:
    material index -> {channel: (H, W, 4) uint8 image} embedded or written beside the file."""
    path = Path(path)
    fmt = path.suffix.lower().lstrip(".")
    if fmt not in EXPORT_FORMATS:
        raise ValueError(f"unsupported format {fmt!r} (FBX needs Autodesk's proprietary SDK)")
    objects = [o for o in objects if o.visible]
    textures = textures or {}
    if fmt.startswith("usd"):
        return _export_usd(path, objects, materials, textures)
    import trimesh
    from trimesh.visual.material import PBRMaterial
    from trimesh.visual.texture import TextureVisuals
    scene = trimesh.Scene()
    for o in objects:
        for k, s in enumerate(o.submeshes):
            tri = o.idx[s.start:s.start + s.count].reshape(-1, 3)
            used, inverse = np.unique(tri, return_inverse=True)
            uv = o.uv[used].copy()
            uv[:, 1] = 1.0 - uv[:, 1]
            mat = materials[s.material]
            pbr = PBRMaterial(name=mat.name, **_pbr_images(textures.get(s.material, {})))
            visual = TextureVisuals(uv=uv, material=pbr)
            mesh = trimesh.Trimesh(vertices=o.pos[used], faces=inverse.reshape(-1, 3), vertex_normals=o.nrm[used],
                                   visual=visual, process=False)
            scene.add_geometry(mesh, node_name=f"{o.name}_{k}" if len(o.submeshes) > 1 else o.name)
    scene.export(str(path))


def _pbr_images(tex):
    from PIL import Image
    out = {}
    if "baseColor" in tex:
        out["baseColorTexture"] = Image.fromarray(tex["baseColor"])
    if "normal" in tex:
        out["normalTexture"] = Image.fromarray(tex["normal"])
    if "roughness" in tex or "metallic" in tex:
        h, w = next(iter(tex.values())).shape[:2]
        mr = np.zeros((h, w, 4), np.uint8)
        mr[..., 3] = 255
        if "roughness" in tex:
            mr[..., 1] = tex["roughness"][..., 0]
        if "metallic" in tex:
            mr[..., 2] = tex["metallic"][..., 0]
        out["metallicRoughnessTexture"] = Image.fromarray(mr)
    if "ambientOcclusion" in tex:
        out["occlusionTexture"] = Image.fromarray(tex["ambientOcclusion"])
    if "emissive" in tex:
        out["emissiveTexture"] = Image.fromarray(tex["emissive"])
    return out


def _export_usd(path, objects, materials, textures):
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, UsdUtils, Vt
    package = path.suffix.lower() == ".usdz"
    layer_path = path.with_suffix(".usdc") if package else path
    stage = Usd.Stage.CreateNew(str(layer_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    root = UsdGeom.Xform.Define(stage, "/Scene")
    stage.SetDefaultPrim(root.GetPrim())
    tex_dir = path.parent
    written = []
    usd_materials = []
    for i, mat in enumerate(materials):
        name = "".join(c if c.isalnum() else "_" for c in mat.name) or f"Material{i}"
        m = UsdShade.Material.Define(stage, f"/Scene/Materials/{name}")
        shader = UsdShade.Shader.Define(stage, f"/Scene/Materials/{name}/Surface")
        shader.CreateIdAttr("UsdPreviewSurface")
        m.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        st = UsdShade.Shader.Define(stage, f"/Scene/Materials/{name}/st")
        st.CreateIdAttr("UsdPrimvarReader_float2")
        st.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        for channel, img in textures.get(i, {}).items():
            inp = {"baseColor": ("diffuseColor", "rgb", Sdf.ValueTypeNames.Color3f),
                   "normal": ("normal", "rgb", Sdf.ValueTypeNames.Normal3f),
                   "roughness": ("roughness", "r", Sdf.ValueTypeNames.Float),
                   "metallic": ("metallic", "r", Sdf.ValueTypeNames.Float),
                   "ambientOcclusion": ("occlusion", "r", Sdf.ValueTypeNames.Float),
                   "emissive": ("emissiveColor", "rgb", Sdf.ValueTypeNames.Color3f),
                   "opacity": ("opacity", "r", Sdf.ValueTypeNames.Float)}.get(channel)
            if inp is None:
                continue
            file = tex_dir / f"{path.stem}_{name}_{channel}.png"
            _write_png(file, img)
            written.append(file)
            t = UsdShade.Shader.Define(stage, f"/Scene/Materials/{name}/{channel}")
            t.CreateIdAttr("UsdUVTexture")
            t.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(file.name)
            t.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st.ConnectableAPI(), "result")
            if channel == "normal":
                t.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(2, 2, 2, 1))
                t.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(-1, -1, -1, 0))
            out = t.CreateOutput(inp[1], Sdf.ValueTypeNames.Float3 if inp[1] == "rgb" else Sdf.ValueTypeNames.Float)
            shader.CreateInput(inp[0], inp[2]).ConnectToSource(out)
        usd_materials.append(m)
    for o in objects:
        oname = "".join(c if c.isalnum() else "_" for c in o.name) or "Mesh"
        mesh = UsdGeom.Mesh.Define(stage, f"/Scene/{oname}")
        mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(o.pos.astype(np.float32)))
        tris = o.triangles
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(tris), 3, np.int32)))
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(tris.reshape(-1).astype(np.int32)))
        mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(o.nrm.astype(np.float32)))
        mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        uv = o.uv.copy()
        uv[:, 1] = 1.0 - uv[:, 1]
        pv = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
        pv.Set(Vt.Vec2fArray.FromNumpy(uv.astype(np.float32)))
        for k, s in enumerate(o.submeshes):
            faces = np.arange(s.start // 3, (s.start + s.count) // 3, dtype=np.int32)
            target = mesh.GetPrim() if len(o.submeshes) == 1 else UsdGeom.Subset.CreateGeomSubset(
                mesh, f"subset{k}", UsdGeom.Tokens.face, Vt.IntArray.FromNumpy(faces),
                UsdShade.Tokens.materialBind).GetPrim()
            UsdShade.MaterialBindingAPI.Apply(target).Bind(usd_materials[s.material])
    stage.GetRootLayer().Save()
    if package:
        UsdUtils.CreateNewUsdzPackage(Sdf.AssetPath(str(layer_path)), str(path))
        layer_path.unlink(missing_ok=True)
        for f in written:
            f.unlink(missing_ok=True)


def _write_png(path, img):
    import OpenImageIO as oiio
    h, w, c = img.shape
    buf = oiio.ImageBuf(oiio.ImageSpec(w, h, c, oiio.UINT8))
    buf.set_pixels(oiio.ROI(), np.ascontiguousarray(img))
    buf.write(str(path))

