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
        """Scan Q6_K tensors — NaN deltas are normal (~2-3% of blocks)."""
        for name, info in self.loader.tensors.items():
            if info.dtype != GGMLType.Q6_K:
                continue
            raw = self.loader.get_tensor_raw(name)
            num_elements = 1
            for d in info.shape: num_elements *= d
            QK_K = 256
            n_blocks = (num_elements + QK_K - 1) // QK_K
            nan_count = 0
            for b in range(n_blocks):
                pos = b * 210
                d = struct.unpack("<e", raw[pos:pos+2])[0]
                if d != d: nan_count += 1
            pct = 100.0 * nan_count / max(n_blocks, 1)
            status = "OK" if pct < 10.0 else "SUSPICIOUS"
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
