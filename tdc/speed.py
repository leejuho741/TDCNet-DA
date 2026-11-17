"""Latency measurement: batch 1, fp16 autocast, CUDA graph replay (falls back to eager), median over runs."""
import time

import numpy as np
import torch


def timeit(fn, warm=30, n=100):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t) * 1000)
    return float(np.median(ts))


def cuda_graph(fn):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    return g.replay


def latency_ms(fn, repeats=3, n=200):
    """fn: zero-argument callable on fixed input tensors. Returns (median ms, how)."""
    def f16():
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            return fn()
    runs, how = [], "fp16 + CUDA graph"
    for _ in range(repeats):
        try:
            runs.append(timeit(cuda_graph(f16), 30, n))
        except Exception:
            torch.cuda.synchronize()
            how = "fp16 eager"
            runs.append(timeit(f16, 30, n))
    return float(np.median(runs)), how
