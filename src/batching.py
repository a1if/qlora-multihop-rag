"""
src/batching.py
Dynamic request batching for an in-process model.

Callers on many threads submit single requests and block on the result. A
worker thread collects whatever arrives within a short window (up to
max_batch_size), groups requests that can share one generate() call, runs
them together and hands each caller its own output.

This is the same idea inference servers (vLLM, TGI) use: the GPU is far
better utilised by one batch of 16 than by 16 sequential calls.
"""

import queue
import threading
import time
from collections import defaultdict
from collections.abc import Callable, Hashable
from concurrent.futures import Future
from dataclasses import dataclass, field


@dataclass
class Request:
    payload: object                    # what run_batch needs for this item
    group: Hashable = None             # only requests with equal groups share a batch
    future: Future = field(default_factory=Future)


_STOP = object()


class DynamicBatcher:
    """
    Args:
        run_batch:      fn(list of payloads) -> list of results, same order.
        max_batch_size: largest batch handed to run_batch.
        max_wait_ms:    how long to wait for more requests after the first one.
    """

    def __init__(self, run_batch: Callable[[list], list], max_batch_size: int = 16, max_wait_ms: float = 10.0):
        self.run_batch = run_batch
        self.max_batch_size = max_batch_size
        self.max_wait_s = max_wait_ms / 1000
        self.batch_sizes: list[int] = []          # observability: size of every batch run
        self._queue: queue.Queue = queue.Queue()
        self._worker = threading.Thread(target=self._loop, name="dynamic-batcher", daemon=True)
        self._worker.start()

    def submit(self, payload, group: Hashable = None):
        """Blocking call from any thread; returns this payload's result or raises its error."""
        request = Request(payload, group)
        self._queue.put(request)
        return request.future.result()

    def close(self):
        self._queue.put(_STOP)
        self._worker.join(timeout=5)

    # ── Worker ────────────────────────────────────────────────────────────────

    def _collect(self, first: Request) -> tuple[list[Request], bool]:
        batch, stop = [first], False
        deadline = time.monotonic() + self.max_wait_s
        while len(batch) < self.max_batch_size:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                item = self._queue.get(timeout=remaining)
            except queue.Empty:
                break
            if item is _STOP:
                stop = True
                break
            batch.append(item)
        return batch, stop

    def _run(self, requests: list[Request]):
        try:
            results = self.run_batch([r.payload for r in requests])
            if len(results) != len(requests):
                raise RuntimeError(f"run_batch returned {len(results)} results for {len(requests)} requests")
        except Exception as exc:
            for r in requests:
                r.future.set_exception(exc)
            return
        self.batch_sizes.append(len(requests))
        for r, result in zip(requests, results, strict=True):
            r.future.set_result(result)

    def _loop(self):
        while True:
            first = self._queue.get()
            if first is _STOP:
                return
            batch, stop = self._collect(first)
            groups = defaultdict(list)
            for request in batch:
                groups[request.group].append(request)
            for requests in groups.values():
                self._run(requests)
            if stop:
                return
