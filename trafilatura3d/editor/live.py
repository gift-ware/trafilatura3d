"""The live refresh of what a workspace's viewports show, shared by the Graph and Paint workspaces.

An edit restarts the refresh: the workspace plans it (`plan`: the steps, every target they cook, what an edit of the
selection restarts from) and the steps run one per turn of the event loop, so the interface answers between them and
a new edit starts over. A dragged value is shown at the preview resolution (`previewing`); the full resolution comes
back on release or once the value has been still for PREVIEW_SETTLE_MS. Once a full refresh is done and the
selection has stayed WARM_DELAY_MS, what its next edit restarts from is cooked ahead where the cache lacks it."""

from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Callable, Generator
from typing import Any

from PySide6.QtCore import QObject, QTimer

from trafilatura3d.evaluator import PREVIEW, Evaluator, Job, Wait, restart_points

PREVIEW_SIZE = 1024  # the largest side a dragged value is shown at
PREVIEW_SETTLE_MS = 500
WARM_DELAY_MS = 500  # ours: the selection still this long before the cache is prepared for its edit (`_warm`)
FENCE_POLL_MS = 2  # while the GPU works on a step's job, how often it is asked whether done (the event loop runs)
PAUSE_POLL_MS = 30  # while paused (a view being navigated: the GPU left to its frames), how often to look again


@dataclass
class Plan:
    """A refresh: `steps` run one per turn of the event loop (given to the workspace's `step`); `related` every
    target they cook (the budget keeps their graphs); `below` what an edit of the selection restarts from (held)."""
    steps: list[Any]
    related: list[Job] = field(default_factory=list)
    below: list[Job] = field(default_factory=list)


class LiveCook(QObject):
    """A workspace's live refresh. `plan()` -> Plan or None (nothing to show); `step(item)` cooks and shows one step
    (with `cook`); `paused()`: the steps wait (a stroke being painted: `resume` at its end); `done()`: a refresh's
    last step ran; `settled()`: a dragged value still for PREVIEW_SETTLE_MS (by default `changed`: the workspace's
    own reaction to an edit, when it has one, calls `changed` itself); `owner`: whose hold the selection's
    restart points are (`Evaluator.hold`: the workspaces share the evaluator)."""

    def __init__(self, parent: QObject, ev: Evaluator, plan: Callable[[], Plan | None], step: Callable[[Any], None],
                 paused: Callable[[], bool] = lambda: False, done: Callable[[], None] = lambda: None,
                 settled: Callable[[], None] | None = None, owner: str = "selection"):
        super().__init__(parent)
        self.ev, self._plan, self._step_fn, self._paused, self._done = ev, plan, step, paused, done
        self.owner = owner
        self.previewing = False
        self.pending: list = []  # the steps still to run of the refresh
        self._running: Generator | None = None  # the step being run, when it is a generator (`_next`)
        self._fence: Wait | None = None  # its last part's work (the GPU's, a process's), resumed once done
        self._warm_fence: Wait | None = None  # the same for the cooking ahead
        self.related: list[Job] = []
        self.precious: frozenset = frozenset()  # the refresh's restart points (`restart_points`)
        self.render_timer = QTimer(self, singleShot=True, interval=0, timeout=self._refresh)
        self._step_timer = QTimer(self, singleShot=True, interval=0, timeout=self._next)
        self._settle_timer = QTimer(self, singleShot=True, interval=PREVIEW_SETTLE_MS,
                                    timeout=settled or self.changed)
        # what the next edit restarts from, cooked ahead after a refresh (`_warm`)
        self._warm_below, self._warm_points, self.warm_queue = [], [], []
        self._warming: Generator | None = None  # the cook of `warm_queue`'s first, between its steps
        self._warm_timer = QTimer(self, singleShot=True, interval=0, timeout=self._warm_step)
        self._warm_delay = QTimer(self, singleShot=True, interval=WARM_DELAY_MS, timeout=self._warm)
        # the cooks the interface asked for besides the views (`side`): key -> (steps, done); the one running, its
        # generator, its last fence
        self._sides: dict[str, tuple[Callable[[], Generator], Callable[[Any], None]]] = {}
        self._side_key: str | None = None
        self._side_running: Generator | None = None
        self._side_fence: Wait | None = None
        self._side_timer = QTimer(self, singleShot=True, interval=0, timeout=self._side_step)

    # --- what starts a refresh ------------------------------------------------------------------

    def changed(self) -> None:
        """An edit, or a drag's release: refreshed at full resolution."""
        self.previewing = False
        self._settle_timer.stop()
        self.render_timer.start()

    def previewed(self) -> None:
        """A dragged value: refreshed at the preview resolution."""
        self.previewing = True
        self._settle_timer.stop()  # counted from when the preview is shown (`_next`)
        self.render_timer.start()

    def schedule(self) -> None:
        """Something else shown (a selection, a display mode): refreshed at the current resolution."""
        self.render_timer.start()

    def start(self, preview: bool) -> None:
        """A refresh planned now, its steps on the next turns of the event loop (a document opened: the window
        answers while it cooks)."""
        self.previewing = preview
        self._refresh()

    def refresh_now(self) -> None:
        """The whole refresh, now."""
        self._start()
        while self.busy:
            if self._fence is not None:
                self._fence.wait()
            self._next()

    @property
    def busy(self) -> bool:
        """Steps of the refresh still to run."""
        return bool(self.pending) or self._running is not None

    def refreshing(self) -> bool:
        """A refresh asked for, being run or waiting for the GPU, or a preview waiting for its full resolution (the
        cooking ahead afterwards does not count)."""
        return (self.busy or self._fence is not None or self.render_timer.isActive() or self._step_timer.isActive()
                or self._settle_timer.isActive())

    def resume(self) -> None:
        """After a pause (`paused`): the refresh it held, or the cooking ahead."""
        if self.busy:
            self._step_timer.start()
        elif self.warm_queue or self._warming is not None:
            self._warm_timer.start()

    def cook(self, jobs) -> None:
        """A step's cook, now: the refresh's other targets and restart points kept by the budget."""
        self.ev.cook(list(jobs), related=self.related, precious=self.precious)

    def cook_steps(self, jobs):
        """`cook` for a step that is a generator (`yield from`): a turn of the event loop at each wait of the cook
        for the GPU (`Evaluator.cook_steps`)."""
        return self.ev.cook_steps(list(jobs), related=self.related, precious=self.precious)

    # --- the steps ------------------------------------------------------------------------------

    def _refresh(self):
        self._start()
        if self.busy:
            self._step_timer.start()

    def _start(self):
        self.pending = []
        for running in (self._running, self._warming, self._side_running):  # the refresh before, the cooking ahead,
            if running is not None:   # a side cook: abandoned (what they computed stays cached; the side cook starts
                running.close()       # over after the refresh)
        self._running = self._warming = self._fence = self._warm_fence = None
        self._side_key = self._side_running = self._side_fence = None
        plan = self._plan()
        if plan is None:
            return
        self.ev.hold(self.owner, plan.below)
        self.related = list(plan.related)
        points = restart_points(self.related, {j.key for j in plan.below})
        self.precious = frozenset(points)
        self.warm_queue = []
        self._warm_delay.stop()
        self._warm_below, self._warm_points = plan.below, list(points.values())
        self.pending = list(plan.steps)

    def _next(self):
        if not self.busy:
            return
        if self._paused():  # asked again soon (a view being navigated), or resumed by `resume` (a stroke)
            self._step_timer.start(PAUSE_POLL_MS)
            return
        if self._fence is not None and not self._fence.done():  # the GPU still on the last part: look again soon
            self._step_timer.start(FENCE_POLL_MS)
            return
        self._fence = None
        if self._running is None:
            step = self._step_fn(self.pending.pop(0))
            if isinstance(step, Generator):
                self._running = step
        if self._running is not None:  # a step that is a generator: one of its parts per turn
            try:
                part = next(self._running)
                self._fence = part if isinstance(part, Wait) else None
            except StopIteration:
                self._running = None
        if self.busy:
            self._step_timer.start(0)
            return
        self._done()
        if self._sides:
            self._side_timer.start(0)
        if self.previewing:  # the full resolution once the dragged value has been still this long after it shows
            self._settle_timer.start()
        else:
            self.ev.drop_partition(PREVIEW)  # shown at full resolution: the previews are not needed until a drag
            self._warm_delay.start()

    # --- cooking ahead --------------------------------------------------------------------------

    def _warm(self):
        """Once a refresh is done and the selection has stayed WARM_DELAY_MS (a selection browsed quickly starts
        nothing): what the next edit of the selection restarts from, cooked ahead where the cache lacks it, so the
        first edit does not compute everything. What is below the selection a checkpoint at a time from the bottom,
        then the restart points of what is computed from it; a turn of the event loop between them, a refresh stops
        it."""
        order, seen = [], set()

        def visit(job):  # what is cached is not computed again, nor what is under it
            if job.key in seen or job.key in self.ev.cache:
                return
            seen.add(job.key)
            for src in job.inputs.values():
                if src:
                    visit(src[0])
            if job.keep:
                order.append(job)

        for job in self._warm_below:
            visit(job)
        self.warm_queue = order + [j for j in self._warm_points if j.key not in self.ev.cache]
        if self.warm_queue:
            self._warm_timer.start()

    def _warm_step(self):
        if self._paused():  # as the refresh's steps
            self._warm_timer.start(PAUSE_POLL_MS)
            return
        if self.busy:
            return
        if self._warm_fence is not None and not self._warm_fence.done():  # as the refresh's steps
            self._warm_timer.start(FENCE_POLL_MS)
            return
        self._warm_fence = None
        if self._warming is None:
            if self._sides:  # the side cooks first (`_side_step` resumes this after them)
                return
            while self.warm_queue and self.warm_queue[0].key in self.ev.cache:
                self.warm_queue.pop(0)
            if not self.warm_queue:
                return
            self._warming = self.cook_steps([self.warm_queue.pop(0)])
        try:  # a job of its cook per turn, as the refresh's steps
            part = next(self._warming)
            self._warm_fence = part if isinstance(part, Wait) else None
        except StopIteration:
            self._warming = None
        if self.warm_queue or self._warming is not None:
            self._warm_timer.start(0)

    # --- side cooks -----------------------------------------------------------------------------

    def side(self, key: str, steps: Callable[[], Generator], done: Callable[[Any], None]) -> None:
        """A cook the interface asks for besides what the views show (a Levels' histogram, the 2D view's): `steps()`
        a generator (`Evaluator.histogram_steps`) run a part per turn of the event loop as the refresh's steps, once
        the refresh is done and before the cooking ahead; `done(its return value)` (None if it failed). One per `key`,
        run in the order asked: a new one replaces the one of its key (a panel rebuilt asks again); a refresh puts
        them off and starts the one it stopped over after."""
        if key == self._side_key:
            self._side_running.close()
            self._side_key = self._side_running = self._side_fence = None
        self._sides.pop(key, None)
        self._sides[key] = (steps, done)
        self._side_timer.start(0)

    @property
    def side_busy(self) -> bool:
        """A side cook asked for and not done yet (`side`)."""
        return bool(self._sides)

    def histogram(self, key: str, job: Job, output: str, then: Callable[[tuple], None]) -> None:
        """`then(histogram)` of a job output (`Evaluator.histogram`): now when it is known, else once cooked as the
        side cook `key` (nothing waits for the GPU); never if it fails."""
        known = self.ev.histogram_known(job, output)
        if known is not None:
            then(known)
            return
        self.side(key, lambda: self.ev.histogram_steps(job, output),
                  lambda stats: None if stats is None else then(stats))

    def _side_step(self):
        if not self._sides or self.busy:  # after the refresh (`_next` starts it again)
            return
        if self._paused():  # as the refresh's steps
            self._side_timer.start(PAUSE_POLL_MS)
            return
        if self._side_fence is not None and not self._side_fence.done():  # as the refresh's steps
            self._side_timer.start(FENCE_POLL_MS)
            return
        self._side_fence = None
        if self._side_running is None:
            if self._warming is not None:  # the cooking ahead's job ends first (a cook started now completes it,
                self._side_timer.start(FENCE_POLL_MS)  # waiting)
                return
            self._side_key = next(iter(self._sides))
            self._side_running = self._sides[self._side_key][0]()
        try:
            part = next(self._side_running)
            self._side_fence = part if isinstance(part, Wait) else None
            self._side_timer.start(0)
            return
        except StopIteration as end:
            value = end.value
        except Exception:  # noqa: BLE001 - a failing node upstream must not stop the editor (as the thumbnails)
            value = None
        done = self._sides.pop(self._side_key)[1]
        self._side_key = self._side_running = None
        done(value)
        if self._sides:
            self._side_timer.start(0)
        elif self.warm_queue:
            self._warm_timer.start(0)
