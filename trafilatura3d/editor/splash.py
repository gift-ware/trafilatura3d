"""The splash screen shown while the editor starts: `splash.png` at its own size in device pixels (half as many points
on a 2x screen), smaller only when the screen cannot hold it, on top of every window until the main window is
shown. Over the image: "Open Source" with the GitHub mark and "by Gift-Ware" at the bottom left, the step being run
and a progress bar along the bottom. The startup reports its steps with `step(text)`."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QByteArray, QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QGuiApplication, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import QSplashScreen

IMAGE = Path(__file__).resolve().parent / "splash.png"
FIT = 0.9  # the largest share of the screen's available size it may take
STEPS = 12  # the bar's length in steps: about 8 the startup reports (MainWindow), the rest the first opening's
# refreshes (`follow`), full at "Ready"
# The GitHub mark (Octicons "mark-github", MIT; use as in GitHub's logo guidelines).
GITHUB_MARK = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path fill="#ffffff" d="M10.226 17.284'
               'c-2.965-.36-5.054-2.493-5.054-5.256 0-1.123.404-2.336 1.078-3.144-.292-.741-.247-2.314.09-2.965.898-.112 '
               '2.111.36 2.83 1.01.853-.269 1.752-.404 2.853-.404 1.1 0 1.999.135 2.807.382.696-.629 1.932-1.1 2.83-.988'
               '.315.606.36 2.179.067 2.942.72.854 1.101 2 1.101 3.167 0 2.763-2.089 4.852-5.098 5.234.763.494 1.28 1.572 '
               '1.28 2.807v2.336c0 .674.561 1.056 1.235.786 4.066-1.55 7.255-5.615 7.255-10.646C23.5 6.188 18.334 1 '
               '11.978 1 5.62 1 .5 6.188.5 12.545c0 4.986 3.167 9.12 7.435 10.669.606.225 1.19-.18 1.19-.786V20.63a2.9 '
               '2.9 0 0 1-1.078.224c-1.483 0-2.359-.808-2.987-2.313-.247-.607-.517-.966-1.034-1.033-.27-.023-.359-.135'
               '-.359-.27 0-.27.45-.471.898-.471.652 0 1.213.404 1.797 1.235.45.651.921.943 1.483.943.561 0 .92-.202 '
               '1.437-.719.382-.381.674-.718.944-.943"/></svg>')
ACCENT = QColor("#f5a623")  # the logo's orange

_current: Splash | None = None


class Splash(QSplashScreen):
    def __init__(self, screen, pix: QPixmap):
        super().__init__(screen, pix, Qt.WindowStaysOnTopHint)  # above every window, the editor's included
        self.text, self.done = "Starting", 0
        self.mark = QSvgRenderer(QByteArray(GITHUB_MARK.encode()))

    def step(self, text: str, last: bool = False) -> None:
        self.text, self.done = text, STEPS if last else min(self.done + 1, STEPS - 1)
        self.repaint()
        QGuiApplication.processEvents()

    def drawContents(self, p: QPainter) -> None:  # noqa: N802 - Qt's name
        w, h = self.width(), self.height()
        u = h / 470.0  # sizes in points of the image at its own size on a 2x screen
        p.setRenderHint(QPainter.Antialiasing)
        # a darker band under the text
        p.fillRect(QRectF(0, h - 64 * u, w, 64 * u), QColor(10, 12, 18, 150))
        f = QFont(self.font())
        f.setPixelSize(max(9, round(11 * u)))
        p.setFont(f)
        x, y = 20 * u, h - 44 * u
        self.mark.render(p, QRectF(x, y - 7 * u, 16 * u, 16 * u))
        p.setPen(QColor("#ffffff"))
        p.drawText(QRectF(x + 22 * u, y - 7 * u, 200 * u, 16 * u), Qt.AlignLeft | Qt.AlignVCenter, "Open Source")
        p.setPen(QColor(255, 255, 255, 110))
        bx = x + 22 * u + p.fontMetrics().horizontalAdvance("Open Source") + 10 * u
        p.drawLine(QPointF(bx, y - 6 * u), QPointF(bx, y + 8 * u))
        p.setPen(QColor(255, 255, 255, 190))
        p.drawText(QRectF(bx + 10 * u, y - 7 * u, 200 * u, 16 * u), Qt.AlignLeft | Qt.AlignVCenter, "by Gift-Ware")
        # the step being run, right aligned, and the bar along the bottom
        p.setPen(QColor(255, 255, 255, 200))
        p.drawText(QRectF(w / 2, y - 7 * u, w / 2 - 20 * u, 16 * u), Qt.AlignRight | Qt.AlignVCenter, self.text + "…")
        track = QRectF(20 * u, h - 20 * u, w - 40 * u, 4 * u)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(255, 255, 255, 50))
        p.drawRoundedRect(track, 2 * u, 2 * u)
        p.setBrush(ACCENT)
        p.drawRoundedRect(QRectF(track.x(), track.y(), track.width() * self.done / STEPS, track.height()), 2 * u, 2 * u)


def show() -> Splash | None:
    """The splash, shown and painted (None without the image or a screen)."""
    global _current
    screen = QGuiApplication.primaryScreen()
    pix = QPixmap(str(IMAGE))
    if pix.isNull() or screen is None:
        return None
    dpr = screen.devicePixelRatio()
    avail = screen.availableGeometry()
    w, h = pix.width() / dpr, pix.height() / dpr  # points at the image's own pixel size
    k = min(1.0, FIT * avail.width() / w, FIT * avail.height() / h)
    if k < 1.0:
        pix = pix.scaled(round(pix.width() * k), round(pix.height() * k), Qt.KeepAspectRatio, Qt.SmoothTransformation)
    pix.setDevicePixelRatio(dpr)
    _current = Splash(screen, pix)
    _current.show()
    _current.raise_()
    QGuiApplication.processEvents()
    return _current


def step(text: str, last: bool = False) -> None:
    """A startup step: its name over the image, the bar one step further (full at the `last`; nothing without a
    splash)."""
    if _current is not None:
        _current.step(text, last)


def finish(window) -> None:
    global _current
    if _current is not None:
        _current.finish(window)
        _current = None


QUIET_POLLS = 3  # the live refreshes idle this many polls in a row: the first opening is done
POLL_MS = 100
LONGEST_S = 180  # closed anyway after this long


def follow(window, lives, ev) -> None:
    """Kept over the shown `window` while its first opening runs: the `lives` (LiveCook) refreshing the views, the
    evaluator `ev` compiling shaders (new pipelines); the bar goes on with the refreshes' steps and the splash closes
    once they are all idle."""
    s = _current
    if s is None:
        return
    from PySide6.QtCore import QElapsedTimer, QTimer
    clock = QElapsedTimer()
    clock.start()
    state = {"quiet": 0, "pipelines": len(ev.pipelines), "most": 0}
    base = s.done

    def poll():
        busy = any(live.refreshing() for live in lives)
        n = len(ev.pipelines)
        pending = sum(len(live.pending) for live in lives)
        state["most"] = max(state["most"], pending)
        left = 1.0 - pending / state["most"] if state["most"] else 0.0
        s.done = max(s.done, min(base + (STEPS - 1 - base) * left, STEPS - 1))  # never back (a refresh restarts)
        if n > state["pipelines"]:
            s.text = f"Compiling GPU shaders ({n})"
        elif busy:
            s.text = "Building the previews"
        state["pipelines"] = n
        state["quiet"] = 0 if busy else state["quiet"] + 1
        s.repaint()
        if state["quiet"] >= QUIET_POLLS or clock.elapsed() > LONGEST_S * 1000:
            timer.stop()
            s.step("Ready", last=True)
            finish(window)

    timer = QTimer(s, interval=POLL_MS, timeout=poll)
    timer.start()
