"""The running editor's local channel, client side (no Qt): one JSON request per connection.

The editor listens on a local socket (a unix domain socket in the user folder, a named pipe on Windows) named
per user, or `TRAFILATURA3D_CHANNEL`. A request is one line `{"cmd": ..., "args": {...}}`, the reply one line
`{"ok": true, "result": ...}` or `{"ok": false, "error": "..."}`; the editor then closes the connection."""

from __future__ import annotations

import getpass
import json
import os
import socket
import sys

TIMEOUT = 300.0  # seconds: a command may cook a large graph


class ChannelError(RuntimeError):
    """The editor refused a command (the message says why)."""


class EditorNotRunning(ChannelError):
    pass


def name() -> str:
    """The channel's name: `TRAFILATURA3D_CHANNEL`, else one per user."""
    user = "".join(c for c in getpass.getuser() if c.isalnum() or c in "-_") or "user"
    return os.environ.get("TRAFILATURA3D_CHANNEL") or f"trafilatura3d-{user}"


def address(channel: str | None = None) -> str:
    """Where the editor listens: the pipe name on Windows (Qt adds \\\\.\\pipe\\), else the socket's full path, in
    the user folder (not the temporary one: TMPDIR is not passed on to every process, an MCP server's
    environment lacks it) or as given when absolute."""
    channel = channel or name()
    if sys.platform == "win32" or os.path.isabs(channel):
        return channel
    from trafilatura3d.project import user_dir
    return str(user_dir() / channel)


def call(cmd: str, args: dict | None = None, channel: str | None = None, timeout: float = TIMEOUT):
    """Send a command to the running editor and return its result; ChannelError when it fails."""
    request = (json.dumps({"cmd": cmd, "args": args or {}}) + "\n").encode()
    where = address(channel)
    try:
        if sys.platform == "win32":
            with open(rf"\\.\pipe\{where}", "r+b", buffering=0) as pipe:
                pipe.write(request)
                data = _read(pipe.read)
        else:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
                s.settimeout(timeout)
                s.connect(where)
                s.sendall(request)
                data = _read(lambda n: s.recv(n))
    except (FileNotFoundError, ConnectionRefusedError) as e:
        raise EditorNotRunning(f"Trafilatura 3D is not running (no channel at {where}): open Trafilatura 3D "
                               f"first") from e
    except OSError as e:
        raise ChannelError(f"channel {where}: {e}") from e
    if not data:
        raise ChannelError("the editor closed the connection without a reply")
    reply = json.loads(data)
    if not reply.get("ok"):
        raise ChannelError(reply.get("error", "failed"))
    return reply.get("result")


def _read(recv) -> bytes:
    buf = bytearray()
    while not buf.endswith(b"\n"):
        chunk = recv(1 << 16)
        if not chunk:
            break
        buf += chunk
    return bytes(buf)
