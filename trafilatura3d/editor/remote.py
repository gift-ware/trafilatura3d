"""The editor's local channel: commands from another process (the MCP bridge, a script) on the open graphs.

A QLocalServer (a unix domain socket, a named pipe on Windows; the protocol and the client are in
`trafilatura3d/channel.py`) driven by the Qt event loop: no thread, no polling, it answers while a modal dialog
is open too. Commands go through the editor's own paths: the document's `edit` (one undo step per command, the
views follow), the main window's open and save, the views' cooks. A failing command answers an error, it never
takes the editor down."""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

from PySide6.QtCore import QBuffer, QByteArray, QIODevice, QObject
from PySide6.QtGui import QImage
from PySide6.QtNetwork import QLocalServer, QLocalSocket

from trafilatura3d import channel, graphtools
from trafilatura3d.editor.graph.nodemenu import fuzzy_score, graph_nodes
from trafilatura3d.nodes import NODES
from trafilatura3d import shaders
from trafilatura3d.package import TEMPLATES, DocumentError

MAX_REQUEST = 64 << 20


def png(img) -> str:
    """An RGBA8 image (h, w, 4) as base64 PNG."""
    h, w = img.shape[:2]
    q = QImage(img.data, w, h, w * 4, QImage.Format_RGBA8888)
    data = QByteArray()
    buf = QBuffer(data)
    buf.open(QIODevice.WriteOnly)
    q.save(buf, "PNG")
    return base64.b64encode(bytes(data)).decode()


class RemoteServer(QObject):
    def __init__(self, window, name: str):
        super().__init__(window)
        self.window = window
        self.address = channel.address(name)
        self.server = QLocalServer(self)
        self.server.setSocketOptions(QLocalServer.UserAccessOption)  # this user only
        self.server.newConnection.connect(self._accept)
        if sys.platform != "win32":
            Path(self.address).parent.mkdir(parents=True, exist_ok=True)  # the user folder
        if self._alive():
            window.paint.log(f"Channel {name}: another Trafilatura 3D has it, this one does not listen")
            return
        if not self.server.listen(self.address):
            QLocalServer.removeServer(self.address)  # left by an editor that did not close (no one answers)
            if not self.server.listen(self.address):
                window.paint.log(f"Channel {name}: {self.server.errorString()}", error=True)

    def _alive(self) -> bool:
        probe = QLocalSocket()
        probe.connectToServer(self.address)
        alive = probe.waitForConnected(300)
        probe.abort()
        return alive

    def listening(self) -> bool:
        return self.server.isListening()

    def close(self) -> None:
        self.server.close()  # removes the socket file

    # --- connections ----------------------------------------------------------------------------------

    def _accept(self) -> None:
        while self.server.hasPendingConnections():
            sock = self.server.nextPendingConnection()
            buf = bytearray()
            sock.readyRead.connect(lambda s=sock, b=buf: self._read(s, b))
            sock.disconnected.connect(sock.deleteLater)

    def _read(self, sock: QLocalSocket, buf: bytearray) -> None:
        if sock.property("answered"):
            return
        buf += bytes(sock.readAll())
        if b"\n" not in buf and len(buf) < MAX_REQUEST:
            return
        sock.setProperty("answered", True)
        reply = self.handle(bytes(buf).split(b"\n", 1)[0])
        sock.write(json.dumps(reply, separators=(",", ":")).encode() + b"\n")
        sock.disconnectFromServer()  # after the reply is written

    def handle(self, line: bytes) -> dict:
        try:
            request = json.loads(line)
            cmd, args = request["cmd"], request.get("args") or {}
            fn = getattr(self, f"cmd_{cmd}", None)
            if fn is None:
                raise DocumentError(f"unknown command {cmd!r}")
            return {"ok": True, "result": fn(**args)}
        except Exception as e:  # noqa: BLE001 - a command never takes the editor down
            return {"ok": False, "error": f"{type(e).__name__}: {e}" if not isinstance(e, DocumentError) else str(e)}

    # --- targets --------------------------------------------------------------------------------------

    def _target(self, graph: str | None = None):
        """(package, graph id): the graph in the Graph view, or the graph `graph` of the open packages (the
        Graph view's package first)."""
        doc, panel = self.window.doc, self.window.graph
        if graph is None:
            if panel.package is None or panel.graph is None:
                raise DocumentError("no graph is open")
            return panel.package, panel.graph_id
        for pkg in ([panel.package] if panel.package in doc.packages else []) + doc.packages:
            if graph in pkg.graphs:
                return pkg, graph
        raise DocumentError(f"no open graph {graph!r}")

    def _show(self, pkg, gid) -> None:
        """Edits happen where they can be seen: the graph comes to the Graph view."""
        panel = self.window.graph
        if (panel.package, panel.graph_id) != (pkg, gid):
            self.window.open_graph(pkg, gid)

    def _edit(self, pkg, label: str, fn):
        """`doc.edit` (one undo step), its error raised (the document emits it to the status bar too). `fn` returns
        something: None means the edit was refused and undone."""
        errors = []
        self.window.doc.error.connect(errors.append)
        try:
            result = self.window.doc.edit(pkg, label, fn)
        finally:
            self.window.doc.error.disconnect(errors.append)
        if result is None:  # the first error is the edit's (the views' cooks may report theirs after a change)
            raise DocumentError(errors[0] if errors else "edit refused")
        return result

    # --- commands -------------------------------------------------------------------------------------

    def cmd_status(self):
        doc, panel = self.window.doc, self.window.graph
        return {"packages": [{"path": str(p.path) if p.path else None, "dirty": id(p) in doc.dirty,
                              "graphs": list(p.graphs)} for p in doc.packages],
                "current": {"path": str(panel.package.path) if panel.package and panel.package.path else None,
                            "graph": panel.graph_id or None},
                "workspace": self.window.shown, "undo": doc.undo.undoText(), "redo": doc.undo.redoText()}

    def cmd_open_package(self, path: str):
        pkg = self.window.load_package(path)
        return {"path": str(pkg.path) if pkg.path else None, "graphs": list(pkg.graphs)}

    def cmd_new_package(self, template: str = shaders.DEFAULT_TEMPLATE, name: str = "New_graph"):
        """As the New graph dialog with "New package": the template's nodes and the 3D View's shader, at the default
        size."""
        if template not in TEMPLATES:
            raise DocumentError(f"unknown template {template!r} (templates: {', '.join(TEMPLATES)})")
        if not name or not name.replace("_", "").isalnum():
            raise DocumentError(f"graph name {name!r}: letters, digits and _ only")
        pkg, gid = self.window.create_graph(name, (11, 11), None, template)
        return {"graphs": list(pkg.graphs), "graph": gid, "outputs": graphtools.describe(pkg.graphs[gid])["outputs"]}

    def cmd_save(self, path: str | None = None, graph: str | None = None):
        pkg, _ = self._target(graph)
        if path is None and pkg.path is None:
            raise DocumentError("the package was never saved: give a path")
        self.window.doc.save(pkg, path)
        self.window.remember(pkg.path)
        self.window.status(f"Saved {pkg.path}")
        return {"path": str(pkg.path)}

    def cmd_get_graph(self, graph: str | None = None):
        pkg, gid = self._target(graph)
        return graphtools.describe(pkg.graphs[gid])

    def cmd_node_types(self, query: str = ""):
        nodes = graph_nodes()
        if not query:  # the categories, then a query lists their nodes
            counts = {}
            for d in nodes.values():
                counts[d.category] = counts.get(d.category, 0) + 1
            return {"categories": dict(sorted(counts.items()))}
        found = []
        for t, d in nodes.items():
            if d.category.lower().startswith(query.lower()):
                found.append((10000, t, d))
                continue
            score = max((s for s in (fuzzy_score(query, d.label), fuzzy_score(query, t)) if s is not None),
                        default=None)
            if score is not None:
                found.append((score, t, d))
        found.sort(key=lambda x: (-x[0], x[1]))
        return [[t, d.label, d.category] for _, t, d in found[:60]]

    def cmd_node_info(self, type: str):
        if type not in NODES:
            raise DocumentError(f"unknown node type {type!r}")
        return graphtools.node_info(NODES[type])

    def cmd_edit(self, ops: list, graph: str | None = None):
        pkg, gid = self._target(graph)
        self._show(pkg, gid)
        g = pkg.graphs[gid]
        results = self._edit(pkg, "Remote edit", lambda: graphtools.apply_ops(g, ops))
        if g.id != gid and g.id in pkg.graphs:  # renamed (set_graph): the Graph view follows, as after a rename there
            self.window.open_graph(pkg, g.id)
        g = pkg.graphs.get(g.id, g)
        return {"results": results, "warnings": graphtools.graph_warnings(g)}

    def cmd_layout(self, graph: str | None = None, dots: bool = False):
        pkg, gid = self._target(graph)
        self._show(pkg, gid)
        g = pkg.graphs[gid]
        before = len(g.nodes)
        self._edit(pkg, "Layout", lambda: graphtools.layout(g, dots=bool(dots)))
        self.window.graph.scene.clearSelection()
        self.window.graph.view.focus()  # the whole graph framed
        return {"nodes": len(g.nodes), "dots_added": len(g.nodes) - before, **graphtools.layout_stats(g)}

    def cmd_focus(self, node: str, graph: str | None = None):
        """As picking a node from the Graph view's node list, then "View output in 2D view": the node selected and
        framed, its parameters in Properties, its output in the 2D View."""
        pkg, gid = self._target(graph)
        self._show(pkg, gid)
        panel = self.window.graph
        if node not in panel.scene.nodes:
            raise DocumentError(f"{node}: no such node")
        panel.scene.clearSelection()
        panel.scene.nodes[node].setSelected(True)  # Properties follows the selection
        panel.view.focus()
        panel.view_2d_node(node)
        self.window.docks["view2d"].show()
        self.window.docks["view2d"].raise_()
        return {"graph": gid, "node": node}

    def cmd_undo(self):
        self.window.doc.undo.undo()
        return {"undo": self.window.doc.undo.undoText(), "redo": self.window.doc.undo.redoText()}

    def cmd_redo(self):
        self.window.doc.undo.redo()
        return {"undo": self.window.doc.undo.undoText(), "redo": self.window.doc.undo.redoText()}

    def _compiled(self, pkg, gid):
        compiled = self.window.doc.compiled(pkg, gid)
        if compiled is None:
            raise DocumentError(f"{gid}: does not compile (see the Graph view's warnings)")
        return compiled

    def cmd_preview(self, node: str, output: str | None = None, size: int = 256, graph: str | None = None):
        pkg, gid = self._target(graph)
        outs = self._compiled(pkg, gid).nodes.get(node)
        if not outs:
            raise DocumentError(f"{node}: no such node, or it gives nothing to show")
        if output is None:
            output = next(iter(outs))
        if output not in outs:
            raise DocumentError(f"{node}: no output {output!r} (has {', '.join(outs)})")
        job, out = outs[output]
        img = self.window.doc.ev.thumbnails([(job, out)], max(8, min(int(size), 4096)))[0]
        return {"png": png(img), "size": [job.size[0], job.size[1]], "gray": job.is_gray(out)}

    def cmd_view3d(self, size: int = 512, **settings):
        v = self.window.view3d
        if v.package is None or v.graph_id not in v.package.graphs:
            v.show_graph(*self._target())
        try:
            state = v.configure(**settings)
        except (TypeError, ValueError) as e:
            raise DocumentError(f"view3d: {e}") from None
        size = max(16, min(int(size), 4096))
        img = v.render_image(size, size)
        return {"png": png(img), "graph": v.graph_id, "view": state}

    def cmd_timings(self, graph: str | None = None):
        pkg, gid = self._target(graph)
        ev = self.window.doc.ev
        # Its own cache within the editor's budget (streaming, 0, would cook the inputs again for every node).
        return graphtools.timings(ev.device, self._compiled(pkg, gid), ev.budget or 2 << 30)
