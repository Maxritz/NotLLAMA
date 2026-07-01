#!/usr/bin/env python3
"""benchmark_turboquant.py — Benchmark TurboQuant on real model weights."""
# MiMo Round 3 — TurboQuant model benchmarking

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from weight_converter import (
    convert_to_tq4,
    convert_to_tq3,
    convert_to_tq6,
    dequant_tq4,
    dequant_tq3,
    dequant_tq6,
    _float16_to_uint16,
)

TQ_FORMATS = {
    "TQ4_128": ("tq4", 128, 66, 7),
    "TQ3_128": ("tq3", 128, 50, 3),
    "TQ6_64": ("tq6", 64, 50, 31),
}


def _dequant_q8_0(data: bytes, n: int) -> np.ndarray:
    """Dequantize Q8_0 bytes to float32."""
    arr = np.frombuffer(data, dtype=np.uint8)
    n_blocks = (n + 31) // 32
    out = np.zeros(n, dtype=np.float32)
    for b in range(n_blocks):
        off = b * 34
        delta = float(np.frombuffer(arr[off:off + 2].tobytes(), dtype=np.float16)[0])
        qs = arr[off + 2:off + 34].astype(np.float32)
        start = b * 32
        end = min(start + 32, n)
        out[start:end] = qs[: end - start] * delta
    return out


def _dequant_f16(data: bytes, n: int) -> np.ndarray:
    return np.frombuffer(data[: n * 2], dtype=np.float16).astype(np.float32)


def _dequant_f32(data: bytes, n: int) -> np.ndarray:
    return np.frombuffer(data[: n * 4], dtype=np.float32)


def _load_tensors(model_dir: Path) -> List[Dict]:
    model_dir = Path(model_dir)
    json_files = sorted(model_dir.glob("*.weights.json"))
    if not json_files:
        raise FileNotFoundError(f"No *.weights.json in {model_dir}")

    meta_path = json_files[0]
    bin_path = meta_path.with_suffix("").with_suffix(".bin")
    if not bin_path.exists():
        bin_path = model_dir / (meta_path.stem.replace(".weights", "") + ".bin")
    if not bin_path.exists():
        raise FileNotFoundError(f"Binary not found for {meta_path}")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    raw = bin_path.read_bytes()
    tensors: List[Dict] = []

    for tensor in meta.get("tensors", []):
        name = tensor["name"]
        dtype = tensor["dtype"]
        shape = tuple(tensor["shape"])
        n_elements = int(np.prod(shape, dtype=np.int64))
        offset = tensor["bin_offset"]
        size = tensor["bin_size"]
        data = raw[offset:offset + size]

        try:
            if dtype == "F32":
                arr = _dequant_f32(data, n_elements)
            elif dtype == "F16":
                arr = _dequant_f16(data, n_elements)
            elif dtype == "Q8_0":
                arr = _dequant_q8_0(data, n_elements)
            else:
                continue
        except Exception as exc:
            print(f"Skipping {name} ({dtype}): {exc}", file=sys.stderr)
            continue

        tensors.append({
            "name": name,
            "shape": shape,
            "n_elements": n_elements,
            "original_bytes": n_elements * 4,
            "data": arr,
        })

    return tensors


def _block_scales(elements: np.ndarray, block_size: int, max_q: int) -> np.ndarray:
    n = elements.shape[0]
    n_blocks = (n + block_size - 1) // block_size
    scales = np.ones(n_blocks, dtype=np.float32)
    for b in range(n_blocks):
        block = elements[b * block_size:(b + 1) * block_size]
        max_abs = float(np.max(np.abs(block))) if block.size else 0.0
        if max_abs > 0.0:
            scales[b] = max_abs / max_q
    return scales


def _quantize_tensor(
    arr: np.ndarray,
    fmt: str,
) -> Tuple[bytes, float, float]:
    """Quantize + dequantize a tensor; returns (packed_bytes, quant_time_s, dequant_time_s)."""
    _, block_size, block_bytes, max_q = TQ_FORMATS[fmt]
    original = arr.ravel().astype(np.float32)
    n = int(original.shape[0])
    scales = _block_scales(original, block_size, max_q)
    n_blocks = scales.shape[0]

    t0 = time.perf_counter()
    packed = bytearray(n_blocks * block_bytes)
    for b in range(n_blocks):
        start = b * block_size
        end = min(start + block_size, n)
        block = original[start:end]
        scale_u16 = _float16_to_uint16(float(scales[b]))
        if fmt == "TQ4_128":
            chunk = convert_to_tq4(block, scale_u16)
        elif fmt == "TQ3_128":
            chunk = convert_to_tq3(block, scale_u16)
        else:
            chunk = convert_to_tq6(block, scale_u16)
        packed[b * block_bytes:(b + 1) * block_bytes] = chunk
    quant_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    packed_bytes = bytes(packed)
    if fmt == "TQ4_128":
        recon = dequant_tq4(packed_bytes, n)
    elif fmt == "TQ3_128":
        recon = dequant_tq3(packed_bytes, n)
    else:
        recon = dequant_tq6(packed_bytes, n)
    dequant_time = time.perf_counter() - t0

    return packed_bytes, quant_time, dequant_time, recon


def _metrics(original: np.ndarray, recon: np.ndarray) -> Dict[str, float]:
    diff = original - recon
    mse = float(np.mean(diff * diff))
    max_err = float(np.max(np.abs(diff)))
    norm_orig = float(np.linalg.norm(original))
    norm_recon = float(np.linalg.norm(recon))
    cos_sim = (
        float(np.dot(original, recon) / (norm_orig * norm_recon))
        if norm_orig > 0.0 and norm_recon > 0.0
        else 1.0
    )
    l1 = float(np.sum(np.abs(diff)))
    return {"mse": mse, "max_err": max_err, "cos_sim": cos_sim, "l1": l1}


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark TurboQuant on model weights.")
    parser.add_argument("--model", type=Path, required=True, help="Directory with *.weights.json + *.bin")
    parser.add_argument("--compare-gguf", action="store_true", help="Also report GGUF baseline (placeholder)")
    parser.add_argument("--output", type=Path, help="Write JSON results to file")
    args = parser.parse_args()

    tensors = _load_tensors(args.model)
    if not tensors:
        print("ERROR: no loadable tensors found", file=sys.stderr)
        return 1

    per_layer_results: List[Dict] = []
    summary: Dict[str, Dict] = {}

    print(f"Loaded {len(tensors)} tensors from {args.model}\n")
    print(f"{'Layer':<30} | {'Format':<8} | {'Orig KB':<10} | {'TQ KB':<10} | {'Ratio':<8} | {'MSE':<12} | {'CosSim':<8}")
    print("-" * 105)

    for tensor in tensors:
        name = tensor["name"]
        original = tensor["data"]
        orig_kb = tensor["original_bytes"] / 1024.0

        for fmt in TQ_FORMATS:
            packed_bytes, q_time, d_time, recon = _quantize_tensor(original, fmt)
            m = _metrics(original, recon)
            tq_kb = len(packed_bytes) / 1024.0
            ratio = len(packed_bytes) / max(tensor["original_bytes"], 1)

            print(
                f"{name:<30} | {fmt:<8} | {orig_kb:<10.2f} | {tq_kb:<10.2f} | "
                f"{ratio:<8.4f}x | {m['mse']:<12.6f} | {m['cos_sim']:<8.6f}"
            )

            per_layer_results.append({
                "layer": name,
                "format": fmt,
                "orig_kb": orig_kb,
                "tq_kb": tq_kb,
                "ratio": ratio,
                "mse": m["mse"],
                "cos_sim": m["cos_sim"],
                "max_err": m["max_err"],
                "l1": m["l1"],
                "quant_time_s": q_time,
                "dequant_time_s": d_time,
            })

    print("\n")
    print(f"{'Format':<10} | {'Total MB':<10} | {'Compressed MB':<15} | {'Ratio':<8} | {'Avg MSE':<12} | {'Avg CosSim':<12} | {'Max MSE':<12}")
    print("-" * 100)

    for fmt in TQ_FORMATS:
        fmt_rows = [r for r in per_layer_results if r["format"] == fmt]
        total_orig_mb = sum(r["orig_kb"] for r in fmt_rows) / 1024.0
        total_tq_mb = sum(r["tq_kb"] for r in fmt_rows) / 1024.0
        ratio = total_tq_mb / max(total_orig_mb, 1e-12)
        avg_mse = sum(r["mse"] for r in fmt_rows) / max(len(fmt_rows), 1)
        avg_cos = sum(r["cos_sim"] for r in fmt_rows) / max(len(fmt_rows), 1)
        max_mse = max(r["mse"] for r in fmt_rows) if fmt_rows else 0.0
        total_q_time = sum(r["quant_time_s"] for r in fmt_rows)
        total_d_time = sum(r["dequant_time_s"] for r in fmt_rows)
        total_bytes = sum(r["orig_kb"] for r in fmt_rows) * 1024.0
        throughput = (total_bytes / 1024.0 / 1024.0) / max(total_q_time + total_d_time, 1e-12)

        print(
            f"{fmt:<10} | {total_orig_mb:<10.2f} | {total_tq_mb:<15.2f} | "
            f"{ratio:<8.4f}x | {avg_mse:<12.6f} | {avg_cos:<12.6f} | {max_mse:<12.6f}"
        )

        summary[fmt] = {
            "total_mb": total_orig_mb,
            "compressed_mb": total_tq_mb,
            "ratio": ratio,
            "avg_mse": avg_mse,
            "avg_cos_sim": avg_cos,
            "max_mse": max_mse,
            "quant_time_s": total_q_time,
            "dequant_time_s": total_d_time,
            "throughput_mbps": throughput,
        }

    # Baseline Q8_0 row: same as original size, zero error.
    all_orig_kb = sum(r["orig_kb"] for r in per_layer_results if r["format"] == "TQ4_128")
    q8_mb = all_orig_kb / 1024.0
    print(f"{'Q8_0':<10} | {q8_mb:<10.2f} | {q8_mb:<15.2f} | {'1.00x':<8} | {'0.000000':<12} | {'1.000000':<12} | {'0.000000':<12}")

    if args.compare_gguf:
        print("\n--compare-gguf: baseline comparison placeholder (Q8_0 == original)")

    if args.output:
        out = {
            "model_dir": str(args.model),
            "per_layer": per_layer_results,
            "summary": summary,
        }
        args.output.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"\nResults written to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
