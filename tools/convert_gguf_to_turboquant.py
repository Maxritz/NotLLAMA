#!/usr/bin/env python3
"""convert_gguf_to_turboquant.py — Convert a weight_converter.py GGUF output to TurboQuant."""
# MiMo Round 3 — GGUF to TurboQuant standalone converter

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
    _float16_to_uint16,
)

TQ_FORMATS = {
    "tq4_128": ("TQ4_128", "tq4", 128, 66, 7),
    "tq3_128": ("TQ3_128", "tq3", 128, 50, 3),
    "tq6_64": ("TQ6_64", "tq6", 64, 50, 31),
}


def _dequant_q8_0(data: bytes, n: int) -> np.ndarray:
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
    fmt_key: str,
) -> bytes:
    _, _, block_size, block_bytes, max_q = TQ_FORMATS[fmt_key]
    original = arr.ravel().astype(np.float32)
    n = int(original.shape[0])
    scales = _block_scales(original, block_size, max_q)
    n_blocks = scales.shape[0]

    packed = bytearray(n_blocks * block_bytes)
    for b in range(n_blocks):
        start = b * block_size
        end = min(start + block_size, n)
        block = original[start:end]
        scale_u16 = _float16_to_uint16(float(scales[b]))
        if fmt_key == "tq4_128":
            chunk = convert_to_tq4(block, scale_u16)
        elif fmt_key == "tq3_128":
            chunk = convert_to_tq3(block, scale_u16)
        else:
            chunk = convert_to_tq6(block, scale_u16)
        packed[b * block_bytes:(b + 1) * block_bytes] = chunk

    return bytes(packed)


def _tensor_to_float32(tensor: Dict, raw: bytes) -> np.ndarray:
    dtype = tensor["dtype"]
    n_elements = int(np.prod(tensor["shape"], dtype=np.int64))
    data = raw[tensor["bin_offset"]:tensor["bin_offset"] + tensor["bin_size"]]

    if dtype == "F32":
        return _dequant_f32(data, n_elements)
    if dtype == "F16":
        return _dequant_f16(data, n_elements)
    if dtype == "Q8_0":
        return _dequant_q8_0(data, n_elements)
    raise ValueError(f"Unsupported source dtype for conversion: {dtype}")


def _load_meta(input_path: Path) -> Tuple[Path, Path, Dict]:
    input_path = Path(input_path)
    if input_path.suffix == ".json":
        meta_path = input_path
        bin_path = meta_path.with_suffix("").with_suffix(".bin")
    else:
        bin_path = input_path
        meta_path = bin_path.with_suffix("").with_suffix(".json")
        if not meta_path.exists():
            meta_path = Path(str(bin_path).replace(".bin", ".weights.json"))

    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata not found: {meta_path}")
    if not bin_path.exists():
        raise FileNotFoundError(f"Binary not found: {bin_path}")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return meta_path, bin_path, meta


def _model_name_from_path(path: Path) -> str:
    stem = path.stem
    stem = stem.replace(".weights", "")
    stem = stem.replace("_q8_0", "").replace("_f16", "").replace("_f32", "")
    return stem


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert a GGUF-derived binary to TurboQuant.")
    parser.add_argument("--input", type=Path, required=True, help="Input .bin or .weights.json")
    parser.add_argument("--format", type=str, required=True, choices=list(TQ_FORMATS.keys()), help="Target TurboQuant format")
    parser.add_argument("--output-dir", type=Path, default=Path("."), help="Output directory")
    args = parser.parse_args()

    fmt_key = args.format
    fmt_name, _, block_size, block_bytes, _ = TQ_FORMATS[fmt_key]

    meta_path, bin_path, meta = _load_meta(args.input)
    raw = bin_path.read_bytes()

    model_name = _model_name_from_path(bin_path)
    out_name = f"{model_name}_{fmt_key}"
    out_bin = args.output_dir / f"{out_name}.weights.bin"
    out_json = args.output_dir / f"{out_name}.weights.json"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    arch = meta.get("model", {}).get("architecture", "unknown")
    model_meta = meta.get("model", {})

    output_meta = {
        "model": {
            "name": out_name,
            "architecture": arch,
            "format": fmt_name,
            "num_layers": model_meta.get("block_count", 0),
            "hidden_dim": model_meta.get("embedding_length", 0),
            "num_heads": model_meta.get("attention.head_count", 0),
            "head_dim": (
                model_meta.get("attention.head_count", 0)
                and model_meta.get("embedding_length", 0) // model_meta.get("attention.head_count", 1)
            ),
            "vocab_size": model_meta.get("vocab_size", 0),
            "max_seq_len": model_meta.get("context_length", 0),
        },
        "tensor_offsets": {},
        "block_layout": {
            "format": fmt_name,
            "block_bytes": block_bytes,
            "elements_per_block": block_size,
            "scale_bytes": 2,
            "data_bytes": block_bytes - 2,
        },
        "tensors": [],
    }

    with open(out_bin, "wb") as bin_f:
        current_offset = 0
        for tensor in meta.get("tensors", []):
            name = tensor["name"]
            dtype = tensor["dtype"]

            if dtype == fmt_name:
                print(f"Skipping {name} (already {fmt_name})")
                continue

            try:
                arr = _tensor_to_float32(tensor, raw)
            except Exception as exc:
                print(f"Skipping {name}: {exc}", file=sys.stderr)
                continue

            packed = _quantize_tensor(arr, fmt_key)
            expected = tensor["n_dims"]
            _ = expected  # unused, but kept for future validation
            padding = (256 - (len(packed) % 256)) % 256

            tensor_meta = {
                "name": name,
                "dtype": fmt_name,
                "dtype_id": -1,
                "shape": list(tensor["shape"]),
                "n_dims": tensor["n_dims"],
                "size_bytes": len(packed),
                "bin_offset": current_offset,
                "bin_size": len(packed) + padding,
                "quant_block_size": block_bytes,
                "quant_block_elements": block_size,
            }
            output_meta["tensors"].append(tensor_meta)
            output_meta["tensor_offsets"][name] = {
                "offset": current_offset,
                "size": len(packed),
                "dtype": fmt_name,
            }

            bin_f.write(packed)
            bin_f.write(b"\x00" * padding)
            current_offset += len(packed) + padding

            print(f"Converted {name}: {len(packed)} bytes ({fmt_name})")

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(output_meta, f, indent=2)

    total = out_bin.stat().st_size
    print(f"\nWrote {out_bin} ({total / 1024 / 1024:.2f} MB)")
    print(f"Wrote {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
