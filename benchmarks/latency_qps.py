"""
Phase 3 — FAISS retrieval benchmark
Measures single-threaded P50/P95/P99 latency and concurrent QPS at multiple
concurrency levels, then prints a summary table and saves plots.

Usage:
    python benchmarks/latency_qps.py [--index result/gallery.index] \
                                     [--embeddings result/gallery_embeddings.npy] \
                                     [--k 20] [--n-runs 1000] [--out result/]
"""

import argparse
import os
import time
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import statistics
import concurrent.futures
from pathlib import Path
from typing import Callable, List

import numpy as np
import faiss


# ---------------------------------------------------------------------------
# Core measurement functions
# ---------------------------------------------------------------------------

def latency_profile(
    search_fn: Callable,
    queries: np.ndarray,
    k: int = 20,
    n_runs: int = 1000,
) -> dict:
    """Single-threaded latency profile over n_runs sequential queries."""
    repeated = np.tile(queries, (n_runs // len(queries) + 1, 1))[:n_runs]
    latencies_ms = []

    for q in repeated:
        t0 = time.perf_counter()
        search_fn(q[None], k)
        latencies_ms.append((time.perf_counter() - t0) * 1000)

    latencies_ms.sort()
    n = len(latencies_ms)
    return {
        "n_runs":   n,
        "mean_ms":  round(statistics.mean(latencies_ms), 3),
        "p50_ms":   round(latencies_ms[int(0.50 * n)], 3),
        "p95_ms":   round(latencies_ms[int(0.95 * n)], 3),
        "p99_ms":   round(latencies_ms[int(0.99 * n)], 3),
        "max_ms":   round(latencies_ms[-1], 3),
        "raw":      latencies_ms,
    }


def qps_benchmark(
    search_fn: Callable,
    queries: np.ndarray,
    k: int = 20,
    concurrency_levels: List[int] = None,
    requests_per_level: int = 500,
) -> List[dict]:
    """
    For each concurrency level, submit `requests_per_level` queries using a
    ThreadPoolExecutor and measure wall-clock QPS and per-request P99.
    """
    if concurrency_levels is None:
        concurrency_levels = [1, 4, 8, 16, 32]

    query_pool = np.tile(queries, (requests_per_level // len(queries) + 1, 1))[:requests_per_level]
    results = []

    for n_workers in concurrency_levels:
        latencies_ms = []

        def _run(q):
            t0 = time.perf_counter()
            search_fn(q[None], k)
            return (time.perf_counter() - t0) * 1000

        wall_start = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
            futs = [pool.submit(_run, q) for q in query_pool]
            latencies_ms = [f.result() for f in concurrent.futures.as_completed(futs)]
        wall_elapsed = time.perf_counter() - wall_start

        latencies_ms.sort()
        n = len(latencies_ms)
        qps = round(n / wall_elapsed, 1)
        results.append({
            "concurrency": n_workers,
            "qps":         qps,
            "p50_ms":      round(latencies_ms[int(0.50 * n)], 3),
            "p99_ms":      round(latencies_ms[int(0.99 * n)], 3),
        })
        print(f"  concurrency={n_workers:>2}  QPS={qps:>7.1f}  p99={latencies_ms[int(0.99*n)]:.2f}ms")

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_latency_table(stats: dict) -> None:
    print("\n── Single-threaded latency profile ─────────────────")
    print(f"  Runs : {stats['n_runs']}")
    print(f"  Mean : {stats['mean_ms']:.3f} ms")
    print(f"  P50  : {stats['p50_ms']:.3f} ms")
    print(f"  P95  : {stats['p95_ms']:.3f} ms")
    print(f"  P99  : {stats['p99_ms']:.3f} ms")
    print(f"  Max  : {stats['max_ms']:.3f} ms")


def print_qps_table(rows: List[dict]) -> None:
    print("\n── Concurrent QPS benchmark ─────────────────────────")
    print(f"  {'Concurrency':>11}  {'QPS':>8}  {'P50 (ms)':>8}  {'P99 (ms)':>8}")
    print("  " + "-" * 45)
    for r in rows:
        print(f"  {r['concurrency']:>11}  {r['qps']:>8.1f}  {r['p50_ms']:>8.3f}  {r['p99_ms']:>8.3f}")


def save_plots(latency_stats: dict, qps_rows: List[dict], out_dir: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not installed — skipping plots)")
        return

    os.makedirs(out_dir, exist_ok=True)

    # latency histogram
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(latency_stats["raw"], bins=60, color="#4C72B0", edgecolor="white", linewidth=0.4)
    for label, val, color in [
        ("P50", latency_stats["p50_ms"], "#2ca02c"),
        ("P95", latency_stats["p95_ms"], "#ff7f0e"),
        ("P99", latency_stats["p99_ms"], "#d62728"),
    ]:
        ax.axvline(val, color=color, linestyle="--", linewidth=1.4, label=f"{label}={val:.2f}ms")
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Count")
    ax.set_title("FAISS Search Latency Distribution (single-threaded, 1000 queries)")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(out_dir, "latency_histogram.png")
    fig.savefig(path, dpi=150)
    print(f"\n  Saved: {path}")
    plt.close(fig)

    # throughput-latency curve
    concurrencies = [r["concurrency"] for r in qps_rows]
    qps_vals      = [r["qps"]         for r in qps_rows]
    p99_vals      = [r["p99_ms"]      for r in qps_rows]

    fig, ax1 = plt.subplots(figsize=(7, 4))
    color_qps, color_p99 = "#4C72B0", "#d62728"
    ax1.plot(concurrencies, qps_vals, "o-", color=color_qps, label="QPS")
    ax1.set_xlabel("Concurrency")
    ax1.set_ylabel("QPS", color=color_qps)
    ax1.tick_params(axis="y", labelcolor=color_qps)

    ax2 = ax1.twinx()
    ax2.plot(concurrencies, p99_vals, "s--", color=color_p99, label="P99 latency")
    ax2.set_ylabel("P99 Latency (ms)", color=color_p99)
    ax2.tick_params(axis="y", labelcolor=color_p99)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    ax1.set_title("FAISS Throughput vs Latency")
    fig.tight_layout()
    path = os.path.join(out_dir, "qps_curve.png")
    fig.savefig(path, dpi=150)
    print(f"  Saved: {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index",      default="result/gallery.index")
    parser.add_argument("--embeddings", default="result/gallery_embeddings.npy")
    parser.add_argument("--k",          type=int, default=20)
    parser.add_argument("--n-runs",     type=int, default=1000)
    parser.add_argument("--out",        default="result/")
    args = parser.parse_args()

    print(f"Loading index  : {args.index}")
    index = faiss.read_index(args.index)
    if hasattr(index, "nprobe"):
        index.nprobe = 10
    print(f"  ntotal={index.ntotal}  type={type(index).__name__}")

    print(f"Loading queries: {args.embeddings}")
    embeddings = np.load(args.embeddings).astype("float32")
    faiss.normalize_L2(embeddings)
    print(f"  shape={embeddings.shape}")

    def search_fn(q, k):
        return index.search(q, k)

    # --- latency profile ---
    print(f"\nRunning latency profile ({args.n_runs} sequential queries, k={args.k}) ...")
    lat_stats = latency_profile(search_fn, embeddings, k=args.k, n_runs=args.n_runs)
    print_latency_table(lat_stats)

    # --- QPS benchmark ---
    print("\nRunning QPS benchmark (500 requests per concurrency level) ...")
    qps_rows = qps_benchmark(search_fn, embeddings, k=args.k)
    print_qps_table(qps_rows)

    # --- plots ---
    print("\nSaving plots ...")
    save_plots(lat_stats, qps_rows, args.out)


if __name__ == "__main__":
    main()
