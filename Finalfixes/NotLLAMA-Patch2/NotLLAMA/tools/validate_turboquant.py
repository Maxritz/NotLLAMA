#!/usr/bin/env python3
"""validate_turboquant.py — Synthetic and optional real-model TurboQuant accuracy check."""
# MiMo Round 2 — TurboQuant accuracy validation

import argparse
import json
import struct
import sys
from pathlib import Path
from typing import Dict, Tuple

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


def _block_scales(elements: np.ndarray, block_size: int, max_q: int) -> np.ndarray:
    """Compute a per-block fp16 scale = max(|x|) / max_q, guarding all-zero blocks."""
    n = elements.shape[0]
    n_blocks = (n + block_size - 1) // block_size
    scales = np.ones(n_blocks, dtype=np.float32)
    for b in range(n_blocks):
        block = elements[b * block_size:(b + 1) * block_size]
        max_abs = float(np.max(np.abs(block))) if block.size else 0.0
        if max_abs > 0.0:
            scales[b] = max_abs / max_q
    return scales


def _quantize_dequant(
    elements: np.ndarray,
    fmt: str,
) -> Tuple[np.ndarray, int]:
    """Quantize then dequantize a float32 array; returns (reconstructed, packed_bytes)."""
    _, block_size, block_bytes, max_q = TQ_FORMATS[fmt]
    n = int(elements.shape[0])
    scales = _block_scales(elements, block_size, max_q)
    n_blocks = scales.shape[0]

    packed = bytearray(n_blocks * block_bytes)
    for b in range(n_blocks):
        start = b * block_size
        end = min(start + block_size, n)
        block = elements[start:end]
        scale_u16 = _float16_to_uint16(float(scales[b]))
        if fmt == "TQ4_128":
            chunk = convert_to_tq4(block, scale_u16)
        elif fmt == "TQ3_128":
            chunk = convert_to_tq3(block, scale_u16)
        else:
            chunk = convert_to_tq6(block, scale_u16)
        packed[b * block_bytes:(b + 1) * block_bytes] = chunk

    packed_bytes = bytes(packed)
    if fmt == "TQ4_128":
        recon = dequant_tq4(packed_bytes, n)
    elif fmt == "TQ3_128":
        recon = dequant_tq3(packed_bytes, n)
    else:
        recon = dequant_tq6(packed_bytes, n)

    return recon, len(packed_bytes)


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
    return {"mse": mse, "max_err": max_err, "cos_sim": cos_sim}


def _synthetic_distribution(name: str, n: int, rng: np.random.Generator) -> np.ndarray:
    if name == "uniform":
        return rng.uniform(-1.0, 1.0, size=n).astype(np.float32)
    if name == "normal":
        return rng.normal(0.0, 0.5, size=n).astype(np.float32)
    if name == "sin":
        x = np.linspace(0.0, 4.0 * np.pi, n, dtype=np.float32)
        return np.sin(x)
    if name == "sparse":
        arr = np.zeros(n, dtype=np.float32)
        idx = rng.choice(n, size=n // 10, replace=False)
        arr[idx] = rng.normal(0.0, 0.5, size=n // 10).astype(np.float32)
        return arr
    raise ValueError(f"Unknown distribution: {name}")


def _load_model_tensors(model_dir: Path) -> Dict[str, np.ndarray]:
    """Load float32 tensors from a weight_converter.py output directory."""
    model_dir = Path(model_dir)
    json_files = sorted(model_dir.glob("*.weights.json"))
    if not json_files:
        raise FileNotFoundError(f"No *.weights.json found in {model_dir}")

    meta_path = json_files[0]
    bin_path = meta_path.with_suffix("").with_suffix(".bin")
    if not bin_path.exists():
        # Try dropping .weights and adding .bin directly.
        bin_path = model_dir / (meta_path.stem.replace(".weights", "") + ".bin")
    if not bin_path.exists():
        raise FileNotFoundError(f"Could not find binary for {meta_path}")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    tensors: Dict[str, np.ndarray] = {}
    raw = bin_path.read_bytes()

    for tensor in meta.get("tensors", []):
        name = tensor["name"]
        dtype = tensor["dtype"]
        shape = tuple(tensor["shape"])
        n_elements = int(np.prod(shape, dtype=np.int64))
        offset = tensor["bin_offset"]
        size = tensor["bin_size"]
        data = raw[offset:offset + size]

        if dtype == "F32":
            arr = np.frombuffer(data[: n_elements * 4], dtype=np.float32).reshape(shape)
        elif dtype == "F16":
            arr = np.frombuffer(data[: n_elements * 2], dtype=np.float16).astype(np.float32).reshape(shape)
        elif dtype == "Q8_0":
            arr = _dequant_q8_0(data, n_elements).reshape(shape)
        else:
            continue
        tensors[name] = arr

    return tensors


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


def _run_synthetic(rng: np.random.Generator) -> Dict[str, Dict[str, float]]:
    results: Dict[str, Dict[str, float]] = {}
    n = 4096
    original = np.concatenate([
        _synthetic_distribution("uniform", n, rng),
        _synthetic_distribution("normal", n, rng),
        _synthetic_distribution("sin", n, rng),
        _synthetic_distribution("sparse", n, rng),
    ])

    for fmt in TQ_FORMATS:
        recon, packed_bytes = _quantize_dequant(original, fmt)
        m = _metrics(original, recon)
        m["size"] = packed_bytes
        m["ratio"] = packed_bytes / (original.nbytes + 1e-12)
        results[fmt] = m
    return results


def _run_model(model_dir: Path, rng: np.random.Generator) -> Dict[str, Dict[str, float]]:
    tensors = _load_model_tensors(model_dir)
    results: Dict[str, Dict[str, float]] = {}

    for fmt in TQ_FORMATS:
        total_mse = 0.0
        total_cos = 0.0
        max_mse = 0.0
        total_packed = 0
        total_orig = 0
        count = 0

        for name, arr in tensors.items():
            original = arr.ravel().astype(np.float32)
            recon, packed_bytes = _quantize_dequant(original, fmt)
            m = _metrics(original, recon)
            total_mse += m["mse"]
            total_cos += m["cos_sim"]
            max_mse = max(max_mse, m["mse"])
            total_packed += packed_bytes
            total_orig += original.nbytes
            count += 1

        results[fmt] = {
            "mse": total_mse / max(count, 1),
            "max_err": 0.0,
            "cos_sim": total_cos / max(count, 1),
            "size": total_packed,
            "ratio": total_packed / max(total_orig, 1),
            "max_mse": max_mse,
        }
    return results


def _print_table(results: Dict[str, Dict[str, float]]) -> None:
    print(f"{'Format':<12} | {'MSE':<12} | {'MaxErr':<12} | {'CosSim':<8} | {'Size':<10} | {'Ratio':<8}")
    print("-" * 75)
    for fmt in TQ_FORMATS:
        m = results[fmt]
        print(
            f"{fmt:<12} | {m['mse']:<12.6f} | {m['max_err']:<12.6f} | "
            f"{m['cos_sim']:<8.6f} | {m['size']:<10} | {m['ratio']:<8.4f}x"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate TurboQuant accuracy.")
    parser.add_argument("--model", type=Path, help="Directory with *.weights.json + *.bin")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    print("Synthetic distribution test:")
    syn_results = _run_synthetic(rng)
    _print_table(syn_results)

    if args.model:
        print("\nReal model test:")
        model_results = _run_model(args.model, rng)
        _print_table(model_results)
        check_results = model_results
    else:
        check_results = syn_results

    ok = all(m["mse"] < 0.01 for m in check_results.values())
    if not ok:
        print("FAILED: MSE threshold exceeded", file=sys.stderr)
        return 1

    print("\nAll TurboQuant accuracy checks PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
