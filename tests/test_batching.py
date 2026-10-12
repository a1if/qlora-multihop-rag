import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from src.batching import DynamicBatcher


class Recorder:
    def __init__(self, fail_on=None):
        self.batches = []
        self.fail_on = fail_on
        self.lock = threading.Lock()

    def __call__(self, payloads):
        with self.lock:
            self.batches.append(list(payloads))
        if self.fail_on is not None and self.fail_on in payloads:
            raise ValueError("boom")
        return [p * 10 for p in payloads]


def run_concurrently(batcher, payloads, groups=None):
    groups = groups or [None] * len(payloads)
    with ThreadPoolExecutor(max_workers=len(payloads)) as pool:
        return list(pool.map(lambda pg: batcher.submit(*pg), zip(payloads, groups, strict=True)))


def test_concurrent_calls_are_batched_and_routed():
    rec = Recorder()
    batcher = DynamicBatcher(rec, max_batch_size=8, max_wait_ms=200)
    try:
        results = run_concurrently(batcher, list(range(8)))
    finally:
        batcher.close()
    assert results == [p * 10 for p in range(8)]       # each caller gets its own result
    assert len(rec.batches) < 8                         # at least some calls shared a batch
    assert sum(batcher.batch_sizes) == 8


def test_respects_max_batch_size():
    rec = Recorder()
    batcher = DynamicBatcher(rec, max_batch_size=3, max_wait_ms=200)
    try:
        run_concurrently(batcher, list(range(10)))
    finally:
        batcher.close()
    assert max(len(b) for b in rec.batches) <= 3


def test_groups_never_mix():
    rec = Recorder()
    batcher = DynamicBatcher(rec, max_batch_size=16, max_wait_ms=200)
    try:
        payloads = list(range(12))
        groups = ["greedy" if p % 2 else ("sample", 0.7) for p in payloads]
        results = run_concurrently(batcher, payloads, groups)
    finally:
        batcher.close()
    assert results == [p * 10 for p in payloads]
    for batch in rec.batches:
        assert len({p % 2 for p in batch}) == 1


def test_errors_reach_every_caller_in_the_batch():
    batcher = DynamicBatcher(Recorder(fail_on=3), max_batch_size=1, max_wait_ms=1)
    try:
        assert batcher.submit(2) == 20
        with pytest.raises(ValueError, match="boom"):
            batcher.submit(3)
        assert batcher.submit(4) == 40                  # worker survives a failed batch
    finally:
        batcher.close()


def test_wrong_result_count_is_an_error():
    batcher = DynamicBatcher(lambda payloads: [], max_batch_size=1, max_wait_ms=1)
    try:
        with pytest.raises(RuntimeError, match="0 results for 1"):
            batcher.submit(1)
    finally:
        batcher.close()
