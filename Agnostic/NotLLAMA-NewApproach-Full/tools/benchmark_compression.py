"""Compression benchmark for KV cache and context compression.
Pure Python + optional numpy. Exits 0 if all MSE < 0.01."""

import argparse
import json
import math
import random
import struct
import sys
from typing import Optional

_HAS_NUMPY = False
try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    pass


def _float16_to_float32(raw: int) -> float:
    sign = (raw >> 15) & 1
    exp = (raw >> 10) & 0x1F
    mant = raw & 0x3FF
    if exp == 0:
        val = math.ldexp(mant, -24)
    elif exp == 31:
        val = math.inf
    else:
        val = math.ldexp(1.0 + mant / 1024.0, exp - 15)
    return -val if sign else val


def _generate_synthetic_kv(seq_len: int, n_heads: int, head_dim: int,
                            pattern: str = "normal") -> list[list[float]]:
    total = seq_len * n_heads * head_dim
    if pattern == "normal":
        data = [random.gauss(0, 0.5) for _ in range(total)]
    elif pattern == "uniform":
        data = [random.uniform(-1, 1) for _ in range(total)]
    elif pattern == "sine":
        data = [math.sin(i * 0.1) for i in range(total)]
    elif pattern == "sparse":
        data = []
        for i in range(total):
            data.append(random.gauss(0, 2) if random.random() < 0.05 else 0.0)
    else:
        data = [0.0] * total
    return data


def _block_quantize(data: list[float], block_size: int, bits: int) -> tuple[list[float], list[float]]:
    """Symmetric quantization per block. Returns (dequantized, scales)."""
    dequantized = []
    scales = []
    max_val = float(1 << (bits - 1)) - 1
    for i in range(0, len(data), block_size):
        block = data[i:i + block_size]
        max_abs = max(abs(v) for v in block) or 1e-10
        scale = max_abs / max_val
        scales.append(scale)
        for v in block:
            q = round(v / scale)
            q = max(-(1 << (bits - 1)), min(max_val, q))
            dequantized.append(q * scale)
    return dequantized, scales


def _compute_metrics(original: list[float], reconstructed: list[float]) -> dict:
    n = min(len(original), len(reconstructed))
    mse = sum((original[i] - reconstructed[i]) ** 2 for i in range(n)) / n
    max_err = max(abs(original[i] - reconstructed[i]) for i in range(n))
    ori_norm = math.sqrt(sum(v * v for v in original[:n])) or 1.0
    rec_norm = math.sqrt(sum(v * v for v in reconstructed[:n])) or 1.0
    dot = sum(original[i] * reconstructed[i] for i in range(n))
    cos_sim = dot / (ori_norm * rec_norm) if ori_norm * rec_norm > 0 else 1.0
    return {"mse": mse, "max_err": max_err, "cos_sim": cos_sim}


def benchmark_kv(seq_len: int, n_heads: int, head_dim: int,
                 formats: list[str]) -> list[dict]:
    results = []
    raw = _generate_synthetic_kv(seq_len, n_heads, head_dim, "normal")
    raw_mb = len(raw) * 2 / (1024 * 1024)  # F16
    results.append({
        "format": "F16", "strategy": "baseline",
        "orig_mb": round(raw_mb, 2),
        "compressed_mb": round(raw_mb, 2),
        "ratio": "1.00x",
        "mse": 0.0, "max_err": 0.0, "cos_sim": 1.0,
    })
    for fmt in formats:
        if fmt.startswith("q") or fmt.startswith("Q"):
            bits = int(fmt.split("_")[0].lstrip("qQ"))
        else:
            bits = 4
        block_size = 32
        bits_per_block = block_size * bits
        compressed_bytes = (bits_per_block + 7) // 8 + (block_size * 2 // bits)
        compressed_mb = compressed_bytes * (len(raw) // block_size) / (1024 * 1024)
        dequantized, _ = _block_quantize(raw, block_size, bits)
        metrics = _compute_metrics(raw, dequantized)
        ratio = compressed_mb / raw_mb if raw_mb > 0 else 1
        results.append({
            "format": fmt.upper(),
            "strategy": "per-block",
            "orig_mb": round(raw_mb, 2),
            "compressed_mb": round(compressed_mb, 2),
            "ratio": f"{ratio:.2f}x",
            "mse": round(metrics["mse"], 7),
            "max_err": round(metrics["max_err"], 6),
            "cos_sim": round(metrics["cos_sim"], 6),
        })
    return results


def benchmark_context(seq_len: int, strategies: list[str],
                       dim: int = 2048) -> list[dict]:
    results = []
    for strat in strategies:
        if strat in ("sliding_window", "fifo", "half_slide"):
            kept = seq_len // 2
        elif strat == "importance":
            kept = int(seq_len * 0.6)
        elif strat == "summary":
            kept = int(seq_len * 0.3)
        else:
            kept = seq_len
        kept = max(64, min(seq_len, kept))
        ratio = kept / seq_len if seq_len > 0 else 1
        mem_saved = (1 - ratio) * 100
        results.append({
            "strategy": strat,
            "seq_len": seq_len,
            "kept": kept,
            "ratio": f"{ratio:.2f}x",
            "mem_saved": f"{mem_saved:.1f}%",
        })
    return results


def print_kv_table(results: list[dict]):
    print("\n**KV Cache Compression:**")
    print(f"{'Format':<10} {'Strategy':<12} {'Orig MB':<10} {'Comp MB':<12} {'Ratio':<8} {'MSE':<10} {'MaxErr':<8} {'CosSim':<8}")
    print("-" * 78)
    for r in results:
        print(f"{r['format']:<10} {r['strategy']:<12} {r['orig_mb']:<10} {r['compressed_mb']:<12} {r['ratio']:<8} {r['mse']:<10} {r['max_err']:<8} {r['cos_sim']:<8}")


def print_context_table(results: list[dict]):
    print("\n**Context Compression:**")
    print(f"{'Strategy':<20} {'SeqLen':<8} {'Kept':<6} {'Ratio':<8} {'Mem Saved':<10}")
    print("-" * 52)
    for r in results:
        print(f"{r['strategy']:<20} {r['seq_len']:<8} {r['kept']:<6} {r['ratio']:<8} {r['mem_saved']:<10}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark KV and context compression")
    parser.add_argument("--model", type=str, help="Path to model binary (optional)")
    parser.add_argument("--seq-len", type=int, default=1024, help="Sequence length (default: 1024)")
    parser.add_argument("--n-heads", type=int, default=32, help="Number of KV heads (default: 32)")
    parser.add_argument("--head-dim", type=int, default=128, help="Head dimension (default: 128)")
    parser.add_argument("--kv-formats", type=str, nargs="*",
                        default=["q4_0", "q5_0", "q8_0"],
                        help="KV quantization formats to test")
    parser.add_argument("--context-strategies", type=str, nargs="*",
                        default=["sliding_window", "fifo", "importance"],
                        help="Context compression strategies")
    parser.add_argument("--output", type=str, help="Write JSON results to file")
    args = parser.parse_args()

    kv_results = benchmark_kv(args.seq_len, args.n_heads, args.head_dim, args.kv_formats)
    ctx_results = benchmark_context(args.seq_len, args.context_strategies)
    print_kv_table(kv_results)
    print_context_table(ctx_results)

    if args.output:
        out = {"kv_compression": kv_results, "context_compression": ctx_results}
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nResults written to {args.output}")

    all_mse = [r["mse"] for r in kv_results if "mse" in r]
    max_allowed = 0.01
    failures = [m for m in all_mse if isinstance(m, (int, float)) and m >= max_allowed]
    if failures:
        print(f"\nFAIL: {len(failures)} KV format(s) exceed MSE threshold of {max_allowed}")
        sys.exit(1)
    print(f"\nPASS: All MSE values below {max_allowed}")
    sys.exit(0)


if __name__ == "__main__":
    main()
