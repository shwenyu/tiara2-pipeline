"""Shared compute helpers: CPU process pools and a reusable GPU scheduler.

The legacy train script embedded a bespoke dynamic-GPU scheduler in bash. That
logic is valuable but was trapped in one script. Here it becomes a reusable,
testable component any stage can call, so "use the GPU / run concurrently"
becomes a one-liner instead of copy-pasted bash.
"""
from __future__ import annotations

import concurrent.futures as cf
import shutil
import subprocess
import time
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence


def cpu_map(func: Callable, items: Sequence, workers: int,
            ordered: bool = False, on_result: Callable | None = None):
    """Run ``func(item)`` across a bounded process pool.

    Bounds in-flight futures to ~2x workers to cap memory (the same pattern the
    chop stage already relied on). Yields results as they complete.
    """
    results = []
    with cf.ProcessPoolExecutor(max_workers=workers) as ex:
        pending = set()
        it = iter(items)
        # prime
        for item in _take(it, workers * 2):
            pending.add(ex.submit(func, item))
        while pending:
            done, pending = cf.wait(pending, return_when=cf.FIRST_COMPLETED)
            for fut in done:
                res = fut.result()
                if on_result:
                    on_result(res)
                results.append(res)
                for item in _take(it, 1):
                    pending.add(ex.submit(func, item))
    return results


def _take(it, n):
    out = []
    for _ in range(n):
        try:
            out.append(next(it))
        except StopIteration:
            break
    return out


@dataclass
class GpuPolicy:
    allowed: Sequence[int]
    min_free_mib: int = 8000
    max_util: int = 30            # a card is "idle enough" to seed a new job
    shared_min_free_mib: int = 6000
    shared_max_util: int = 82     # cap when packing extra jobs onto a busy card
    max_tasks_per_gpu: int = 2
    poll_seconds: int = 10
    launch_delay: int = 30


def query_gpus() -> list[dict]:
    """Return per-GPU {index, free_mib, util}. Empty if nvidia-smi is absent."""
    if not shutil.which("nvidia-smi"):
        return []
    out = subprocess.run(
        ["nvidia-smi",
         "--query-gpu=index,memory.free,utilization.gpu",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=30,
    ).stdout.strip()
    gpus = []
    for line in out.splitlines():
        idx, free, util = (x.strip() for x in line.split(","))
        gpus.append({"index": int(idx), "free_mib": int(free), "util": int(util)})
    return gpus


class GpuScheduler:
    """Dynamic GPU packing scheduler (portable reimplementation of the bash one).

    Feed it a list of jobs (each a callable that accepts a gpu index and returns
    a subprocess handle or a blocking result). It places jobs on genuinely idle
    cards first, then packs additional jobs while staying under the shared caps.
    """

    def __init__(self, policy: GpuPolicy, log=None):
        self.policy = policy
        self.log = log
        self._running: dict[int, int] = {g: 0 for g in policy.allowed}

    def _pick(self, seeding: bool) -> int | None:
        p = self.policy
        gpus = {g["index"]: g for g in query_gpus()}
        min_free = p.min_free_mib if seeding else p.shared_min_free_mib
        max_util = p.max_util if seeding else p.shared_max_util
        best = None
        for idx in p.allowed:
            g = gpus.get(idx)
            if g is None:
                continue
            if self._running[idx] >= p.max_tasks_per_gpu:
                continue
            if g["free_mib"] < min_free or g["util"] > max_util:
                continue
            if best is None or g["free_mib"] > gpus[best]["free_mib"]:
                best = idx
        return best

    def run(self, jobs: Iterable[Callable[[int], object]],
            wait_fn: Callable[[object], None]) -> None:
        """Schedule ``jobs`` across GPUs. ``wait_fn(handle)`` blocks until done.

        Handles are tracked so ``max_tasks_per_gpu`` and the shared caps are
        honored. This is intentionally simple and side-effect free to test.
        """
        p = self.policy
        handles: list[tuple[int, object]] = []
        queue = list(jobs)
        while queue or handles:
            # reap finished
            still = []
            for gpu, h in handles:
                if _finished(h):
                    self._running[gpu] -= 1
                else:
                    still.append((gpu, h))
            handles = still
            if queue:
                seeding = all(v == 0 for v in self._running.values())
                gpu = self._pick(seeding)
                if gpu is not None:
                    job = queue.pop(0)
                    if self.log:
                        self.log.info("launch job on GPU %d (running=%s)",
                                      gpu, self._running)
                    handle = job(gpu)
                    self._running[gpu] += 1
                    handles.append((gpu, handle))
                    wait = wait_fn if False else None  # non-blocking placement
                    time.sleep(p.launch_delay if not seeding else 0)
                    continue
            time.sleep(p.poll_seconds)
        # drain: block on any handle exposing wait
        for _, h in handles:
            wait_fn(h)


def _finished(handle) -> bool:
    poll = getattr(handle, "poll", None)
    if callable(poll):
        return poll() is not None
    done = getattr(handle, "done", None)
    if callable(done):
        return done()
    return True
