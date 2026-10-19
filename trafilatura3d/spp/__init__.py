"""Import of .spp projects into Paint projects (one way, as the .sbs importer): the mesh, the Texture Sets, their
channels, mesh maps and layer stacks; painted effects are made again by replaying their strokes.

    session = import_spp("project.spp")
    session.load_warnings  # what could not be carried over
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from trafilatura3d import meshio
from trafilatura3d.spp.container import Container, ContainerError, decode_image
from trafilatura3d.spp.convert import Converter
from trafilatura3d.spp.document import DocumentError
from trafilatura3d.spp.scene import SceneError, parse as parse_scene

SUFFIX = ".spp"
ImportError_ = (ContainerError, DocumentError, SceneError)


def import_spp(path, evaluator=None, replay: bool = True, progress=None):
    """A Session (paint/session.py) made from a .spp file; `progress(text)` is told each step. Its
    `load_warnings` list what was left out or approximated."""
    from trafilatura3d.paint.session import Session
    from trafilatura3d.paint.stack import Resources
    path = Path(path)
    say = progress or (lambda text: None)
    with Container(path) as c:
        say("Reading the document")
        doc = c.document()
        scene = parse_scene(c.bytes("paint/scene.assbin"))
        conv = Converter(doc, scene, c)
        project, names = conv.project(str(path))
        objects = conv.meshes(names)
        fit = meshio.fit(objects)
        project.validate()
        say(f"Reading {sum(len(t) for t in conv.images.values())} images")
        jobs = [(f"{name}.{udim}" if udim else name, dataset)
                for name, tiles in conv.images.items() for udim, dataset in tiles.items()]

        def load(job):
            name, dataset = job
            pixels, _head = decode_image(c.bytes(dataset))
            return name, Resources.prepare(pixels)

        with ThreadPoolExecutor(max_workers=8) as pool:
            loaded = list(pool.map(load, jobs))
    session = Session(project, objects, names, evaluator)
    session.fit = fit
    session.source = str(path)
    for name, (image, stamp) in loaded:
        session.resources.put(name, image, stamp)
    if replay and conv.paint:
        from trafilatura3d.spp.replay import Replayer
        Replayer(session, conv, scene, path, say).run()
    session.load_warnings = list(conv.notes)
    return session
