from pathlib import Path

from trafilatura3d.nodes.base import Input, Node, Output, PortType


class Thumbnail(Node):
    """Helper: box-filters any result down to a small RGBA8 image for node thumbnails and previews
    (grayscale shown as gray, values clamped to 0-1)."""

    type = "thumbnail"
    registered = False
    inputs = {"image": Input()}
    outputs = {"output": Output(PortType.COLOR)}

    def wgsl(self):
        return (Path(__file__).parent / "thumbnail.wgsl").read_text(encoding="utf-8")


THUMBNAIL = Thumbnail()
