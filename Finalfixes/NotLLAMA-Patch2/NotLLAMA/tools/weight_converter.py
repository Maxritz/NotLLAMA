#!/usr/bin/env python3
"""weight_converter.py — Converts .gguf → GPU-ready binary + metadata JSON."""
import json
import struct
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent))
from gguf_loader import GGUFLoader, GGMLType

class WeightConverter:
    """Convert GGUF to a flat binary + JSON metadata for fast GPU upload."""

    def __init__(self, gguf_path: str):
        self.loader = GGUFLoader(gguf_path)

    def validate_q6_k(self):
        """Scan Q6_K tensors — d-last layout (d at offset 208 in 210-byte block)."""
        for name, info in self.loader.tensors.items():
            if info.dtype != GGMLType.Q6_K:
                continue
            raw = self.loader.get_tensor_raw(name)
            num_elements = 1
            for d in info.shape: num_elements *= d
            QK_K = 256
            BLOCK_SIZE = 210
            n_blocks = (num_elements + QK_K - 1) // QK_K
            nan_count = 0
            for b in range(n_blocks):
                pos = b * BLOCK_SIZE
                # Q6_K is d-last: ql[128]@0, qh[64]@128, scales[16]@192, d@208
                d = struct.unpack("<e", raw[pos+208:pos+210])[0]
                if d != d: nan_count += 1
            pct = 100.0 * nan_count / max(n_blocks, 1)
            status = "OK" if pct < 5.0 else "SUSPICIOUS"
            print(f"{name}: {n_blocks} blocks, NaN={nan_count} ({pct:.1f}%) [{status}]")
        return True

    def convert(self, out_prefix: str):
        bin_path = f"{out_prefix}.weights.bin"
        json_path = f"{out_prefix}.weights.json"

        arch = self.loader.metadata.get("general.architecture", "llama")
        prefix = arch

        metadata = {
            "model": {
                "architecture": arch,
                "block_count": self.loader.metadata.get(f"{prefix}.block_count", 0),
                "embedding_length": self.loader.metadata.get(f"{prefix}.embedding_length", 0),
                "feed_forward_length": self.loader.metadata.get(f"{prefix}.feed_forward_length", 0),
                "attention.head_count": self.loader.metadata.get(f"{prefix}.attention.head_count", 0),
                "attention.head_count_kv": self.loader.metadata.get(f"{prefix}.attention.head_count_kv", 0),
                "vocab_size": self.loader.metadata.get(f"{prefix}.vocab_size", len(self.loader.get_vocab())),
                "context_length": self.loader.metadata.get(f"{prefix}.context_length", 0),
            },
            "tokenizer": {
                "vocab": self.loader.get_vocab(),
                "merges": self.loader.get_merges(),
                "special_tokens": self.loader.get_special_tokens(),
            },
            "tensors": []
        }

        with open(bin_path, "wb") as bin_f:
            current_offset = 0

            for name, info in self.loader.tensors.items():
                raw = self.loader.get_tensor_raw(name)
                padding = (256 - (len(raw) % 256)) % 256

                tensor_meta = {
                    "name": name,
                    "dtype": info.dtype.name,
                    "dtype_id": int(info.dtype),
                    "shape": list(info.shape),
                    "n_dims": info.n_dims,
                    "size_bytes": info.size,
                    "bin_offset": current_offset,
                    "bin_size": len(raw) + padding,
                    "quant_block_size": self.loader.TYPE_BLOCK_SIZES.get(info.dtype, 1),
                    "quant_block_elements": self.loader.TYPE_BLOCK_ELEMENTS.get(info.dtype, 1),
                }
                metadata["tensors"].append(tensor_meta)

                bin_f.write(raw)
                bin_f.write(b'\x00' * padding)
                current_offset += len(raw) + padding

        with open(json_path, "w") as f:
            json.dump(metadata, f, indent=2)

        total_size = Path(bin_path).stat().st_size
        print(f"Converted to:")
        print(f"  Binary: {bin_path} ({total_size / 1024 / 1024:.2f} MB)")
        print(f"  JSON:   {json_path}")
        print(f"  Vocab:  {len(metadata['tokenizer']['vocab'])} tokens")
        return bin_path, json_path


def main():
    if len(sys.argv) < 3:
        print("Usage: python weight_converter.py <model.gguf> <output_prefix>")
        sys.exit(1)

    gguf_path = sys.argv[1]
    out_prefix = sys.argv[2]
    conv = WeightConverter(gguf_path)
    if not conv.validate_q6_k():
        print("ERROR: model validation failed — corrupted Q6_K data detected, aborting")
        sys.exit(1)
    conv.convert(out_prefix)


if __name__ == "__main__":
    main()


# MiMo Round 2 — TurboQuant conversion helpers
# These functions are appended after the main converter so they can be imported
# by validation / benchmark scripts without changing the original GGUF workflow.

import numpy as np


def _float16_to_uint16(scale_f16: float) -> int:
    """Pack a Python float into a little-endian uint16 float16 bit pattern."""
    return int(np.asarray(scale_f16, dtype=np.float16).view(np.uint16))


def _uint16_to_float16(scale_u16: int) -> float:
    """Unpack a little-endian uint16 float16 bit pattern into a Python float."""
    return float(np.asarray(scale_u16, dtype=np.uint16).view(np.float16))


def _pad_block(elements: np.ndarray, target: int) -> np.ndarray:
    """Pad a 1-D float32 slice to exactly target elements with zeros."""
    n = elements.shape[0]
    if n == target:
        return elements
    padded = np.zeros(target, dtype=np.float32)
    padded[:n] = elements
    return padded


def convert_to_tq4(elements: np.ndarray, scale: int) -> bytes:
    """Quantize a float32 array to TQ4_128 (66 bytes per 128 elements).

    Args:
        elements: 1-D (or flattenable) float32 numpy array.
        scale: little-endian uint16 bit pattern of the fp16 block scale.

    Returns:
        Packed bytes: 2-byte scale + 64 bytes of low-nibble-first nibbles per block.
    """
    elements = np.asarray(elements, dtype=np.float32).ravel()
    scale_f = _uint16_to_float16(scale)
    n = int(elements.shape[0])
    n_blocks = (n + 127) // 128
    out = np.zeros(n_blocks * 66, dtype=np.uint8)

    for b in range(n_blocks):
        start = b * 128
        end = min(start + 128, n)
        block = _pad_block(elements[start:end], 128)

        q = np.clip(np.round(block / scale_f), -8.0, 7.0).astype(np.int16)
        q = ((q + 8) & 0xF).astype(np.uint8)

        # Pack low nibble first: byte[j] = v[2j] | (v[2j+1] << 4)
        byte_vals = q[0::2] | (q[1::2] << 4)

        off = b * 66
        out[off:off + 2] = np.frombuffer(struct.pack("<H", scale), dtype=np.uint8)
        out[off + 2:off + 66] = byte_vals

    return out.tobytes()


def convert_to_tq3(elements: np.ndarray, scale: int) -> bytes:
    """Quantize a float32 array to TQ3_128 (50 bytes per 128 elements).

    Args:
        elements: 1-D (or flattenable) float32 numpy array.
        scale: little-endian uint16 bit pattern of the fp16 block scale.

    Returns:
        Packed bytes: 2-byte scale + 48 bytes of sequential 3-bit weights per block.
    """
    elements = np.asarray(elements, dtype=np.float32).ravel()
    scale_f = _uint16_to_float16(scale)
    n = int(elements.shape[0])
    n_blocks = (n + 127) // 128
    out = np.zeros(n_blocks * 50, dtype=np.uint8)

    for b in range(n_blocks):
        start = b * 128
        end = min(start + 128, n)
        block = _pad_block(elements[start:end], 128)

        q = np.clip(np.round(block / scale_f), -4.0, 3.0).astype(np.int16)
        q = ((q + 4) & 0x7).astype(np.int64)

        # Sequential 3-bit packing: 128 values = 384 bits = 48 bytes, little-endian.
        # Use Python int for arbitrary-precision bit placement.
        bits = sum(int(q[i]) << (3 * i) for i in range(128))
        packed = bits.to_bytes(48, "little")

        off = b * 50
        out[off:off + 2] = np.frombuffer(struct.pack("<H", scale), dtype=np.uint8)
        out[off + 2:off + 50] = np.frombuffer(packed, dtype=np.uint8)

    return out.tobytes()


def convert_to_tq6(elements: np.ndarray, scale: int) -> bytes:
    """Quantize a float32 array to TQ6_64 (50 bytes per 64 elements).

    Args:
        elements: 1-D (or flattenable) float32 numpy array.
        scale: little-endian uint16 bit pattern of the fp16 block scale.

    Returns:
        Packed bytes: 2-byte scale + 48 bytes of sequential 6-bit weights per block.
    """
    elements = np.asarray(elements, dtype=np.float32).ravel()
    scale_f = _uint16_to_float16(scale)
    n = int(elements.shape[0])
    n_blocks = (n + 63) // 64
    out = np.zeros(n_blocks * 50, dtype=np.uint8)

    for b in range(n_blocks):
        start = b * 64
        end = min(start + 64, n)
        block = _pad_block(elements[start:end], 64)

        q = np.clip(np.round(block / scale_f), -32.0, 31.0).astype(np.int16)
        q = ((q + 32) & 0x3F).astype(np.int64)

        # Sequential 6-bit packing: 64 values = 384 bits = 48 bytes, little-endian.
        # Use Python int for arbitrary-precision bit placement.
        bits = sum(int(q[i]) << (6 * i) for i in range(64))
        packed = bits.to_bytes(48, "little")

        off = b * 50
        out[off:off + 2] = np.frombuffer(struct.pack("<H", scale), dtype=np.uint8)
        out[off + 2:off + 50] = np.frombuffer(packed, dtype=np.uint8)

    return out.tobytes()


def dequant_tq4(data: bytes, n: int) -> np.ndarray:
    """Dequantize TQ4_128 bytes back to float32 (first n elements)."""
    arr = np.frombuffer(data, dtype=np.uint8)
    n_blocks = (n + 127) // 128
    out = np.zeros(n, dtype=np.float32)

    for b in range(n_blocks):
        off = b * 66
        scale = _uint16_to_float16(int(struct.unpack("<H", arr[off:off + 2].tobytes())[0]))
        weight_bytes = arr[off + 2:off + 66]

        nibbles = np.zeros(128, dtype=np.uint8)
        nibbles[0::2] = weight_bytes & 0x0F
        nibbles[1::2] = (weight_bytes >> 4) & 0x0F

        vals = (nibbles.astype(np.float32) - 8.0) * scale
        start = b * 128
        end = min(start + 128, n)
        out[start:end] = vals[:end - start]

    return out


def dequant_tq3(data: bytes, n: int) -> np.ndarray:
    """Dequantize TQ3_128 bytes back to float32 (first n elements)."""
    arr = np.frombuffer(data, dtype=np.uint8)
    n_blocks = (n + 127) // 128
    out = np.zeros(n, dtype=np.float32)

    for b in range(n_blocks):
        off = b * 50
        scale = _uint16_to_float16(int(struct.unpack("<H", arr[off:off + 2].tobytes())[0]))
        bits = int.from_bytes(arr[off + 2:off + 50].tobytes(), "little")
        count = min(128, n - b * 128)

        for i in range(count):
            q = (bits >> (3 * i)) & 0x7
            out[b * 128 + i] = (float(q) - 4.0) * scale

    return out


def dequant_tq6(data: bytes, n: int) -> np.ndarray:
    """Dequantize TQ6_64 bytes back to float32 (first n elements)."""
    arr = np.frombuffer(data, dtype=np.uint8)
    n_blocks = (n + 63) // 64
    out = np.zeros(n, dtype=np.float32)

    for b in range(n_blocks):
        off = b * 50
        scale = _uint16_to_float16(int(struct.unpack("<H", arr[off:off + 2].tobytes())[0]))
        bits = int.from_bytes(arr[off + 2:off + 50].tobytes(), "little")
        count = min(64, n - b * 64)

        for i in range(count):
            q = (bits >> (6 * i)) & 0x3F
            out[b * 64 + i] = (float(q) - 32.0) * scale

    return out


# TurboQuant conversion test (intended CLI extension point):
# python weight_converter.py --tq-test <input.bin> <format: tq4|tq3|tq6>
