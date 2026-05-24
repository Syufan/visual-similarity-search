# Phase 3 — P50/P95/P99 latency profiling + concurrent QPS benchmark

import time
import statistics
import concurrent.futures
from typing import List

import numpy as np


def latency_profile(search_fn, queries: np.ndarray, k: int = 20, n_runs: int = 1000) -> dict:
    latencies = []
    for q in np.tile(queries, (n_runs // len(queries) + 1, 1))[:n_runs]:
        t0 = time.perf_counter()
        search_fn(q[None], k)
        latencies.append((time.perf_counter() - t0) * 1000)

    latencies.sort()
    return {
        "p50_ms":  latencies[int(0.50 * n_runs)],
        "p95_ms":  latencies[int(0.95 * n_runs)],
        "p99_ms":  latencies[int(0.99 * n_runs)],
        "mean_ms": statistics.mean(latencies),
    }


def qps_benchmark(search_fn, queries: np.ndarray, concurrency_levels: List[int] = None) -> List[dict]:
    if concurrency_levels is None:
        concurrency_levels = [1, 4, 8, 16, 32]

    results = []
    for n_workers in concurrency_levels:
        # TODO: submit n_workers concurrent requests, measure wall-clock QPS and P99
        pass
    return results


if __name__ == "__main__":
    # TODO: load index, run both benchmarks, print table
    pass
