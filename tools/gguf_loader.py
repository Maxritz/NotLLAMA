#!/usr/bin/env python3
"""gguf_loader.py — Full GGUF v3 parser with quantized tensor support."""
import struct
from typing import Dict, List, Tuple, Optional, BinaryIO, Any
from dataclasses import dataclass, field
from enum import IntEnum

class GGUFValueType(IntEnum):
    UINT8 = 0; INT8 = 1; UINT16 = 2; INT16 = 3; UINT32 = 4; INT32 = 5
    FLOAT32 = 6; BOOL = 7; STRING = 8; ARRAY = 9; UINT64 = 10; INT64 = 11
    FLOAT64 = 12

class GGMLType(IntEnum):
    F32 = 0; F16 = 1; Q4_0 = 2; Q4_1 = 3; Q5_0 = 6; Q5_1 = 7; Q8_0 = 8
    Q8_1 = 9; Q2_K = 10; Q3_K = 11; Q4_K = 12; Q5_K = 13; Q6_K = 14; Q8_K = 15
    IQ2_XXS = 16; IQ2_XS = 17; IQ3_XXS = 18; IQ1_S = 19; IQ4_NL = 20; IQ3_S = 21
    IQ2_S = 22; IQ4_XS = 23; I8 = 24; I16 = 25; I32 = 26; I64 = 27; F64 = 28

@dataclass
class TensorInfo:
    name: str
    n_dims: int
    shape: Tuple[int, ...]
    dtype: GGMLType
    offset: int
    size: int = 0

@dataclass
class BufferDesc:
    name: str
    size_bytes: int
    quant_format: GGMLType
    shape: Tuple[int, ...]
    scale: float = 1.0
    gpu_address: int = 0

class GGUFParseError(Exception):
    pass

class GGUFLoader:
    TYPE_BLOCK_SIZES = {
        GGMLType.F32: 4, GGMLType.F16: 2,
        GGMLType.Q4_0: 18, GGMLType.Q4_1: 20,
        GGMLType.Q5_0: 22, GGMLType.Q5_1: 24,
        GGMLType.Q8_0: 34, GGMLType.Q8_1: 36,
        GGMLType.Q2_K: 84, GGMLType.Q3_K: 110,
        GGMLType.Q4_K: 144, GGMLType.Q5_K: 176,
        GGMLType.Q6_K: 210, GGMLType.Q8_K: 292,
        GGMLType.IQ2_XXS: 256, GGMLType.IQ2_XS: 256,
        GGMLType.IQ3_XXS: 256, GGMLType.IQ1_S: 256,
        GGMLType.IQ4_NL: 256, GGMLType.IQ3_S: 256,
        GGMLType.IQ2_S: 256, GGMLType.IQ4_XS: 256,
        GGMLType.I8: 1, GGMLType.I16: 2, GGMLType.I32: 4, GGMLType.I64: 8,
        GGMLType.F64: 8,
    }

    TYPE_BLOCK_ELEMENTS = {
        GGMLType.F32: 1, GGMLType.F16: 1,
        GGMLType.Q4_0: 32, GGMLType.Q4_1: 32,
        GGMLType.Q5_0: 32, GGMLType.Q5_1: 32,
        GGMLType.Q8_0: 32, GGMLType.Q8_1: 32,
        GGMLType.Q2_K: 256, GGMLType.Q3_K: 256,
        GGMLType.Q4_K: 256, GGMLType.Q5_K: 256,
        GGMLType.Q6_K: 256, GGMLType.Q8_K: 256,
        GGMLType.IQ2_XXS: 256, GGMLType.IQ2_XS: 256,
        GGMLType.IQ3_XXS: 256, GGMLType.IQ1_S: 256,
        GGMLType.IQ4_NL: 256, GGMLType.IQ3_S: 256,
        GGMLType.IQ2_S: 256, GGMLType.IQ4_XS: 256,
        GGMLType.I8: 1, GGMLType.I16: 1, GGMLType.I32: 1, GGMLType.I64: 1,
        GGMLType.F64: 1,
    }

    def __init__(self, path: str):
        self.path = path
        self.metadata: Dict[str, Any] = {}
        self.tensors: Dict[str, TensorInfo] = {}
        self.tensor_data_offset: int = 0
        self._parse()

    def _read_str(self, f: BinaryIO) -> str:
        length = struct.unpack("<Q", f.read(8))[0]
        return f.read(length).decode("utf-8")

    def _read_val(self, f: BinaryIO, vtype: GGUFValueType) -> Any:
        if vtype == GGUFValueType.UINT8: return struct.unpack("<B", f.read(1))[0]
        elif vtype == GGUFValueType.INT8: return struct.unpack("<b", f.read(1))[0]
        elif vtype == GGUFValueType.UINT16: return struct.unpack("<H", f.read(2))[0]
        elif vtype == GGUFValueType.INT16: return struct.unpack("<h", f.read(2))[0]
        elif vtype == GGUFValueType.UINT32: return struct.unpack("<I", f.read(4))[0]
        elif vtype == GGUFValueType.INT32: return struct.unpack("<i", f.read(4))[0]
        elif vtype == GGUFValueType.FLOAT32: return struct.unpack("<f", f.read(4))[0]
        elif vtype == GGUFValueType.UINT64: return struct.unpack("<Q", f.read(8))[0]
        elif vtype == GGUFValueType.INT64: return struct.unpack("<q", f.read(8))[0]
        elif vtype == GGUFValueType.FLOAT64: return struct.unpack("<d", f.read(8))[0]
        elif vtype == GGUFValueType.BOOL: return struct.unpack("<?", f.read(1))[0]
        elif vtype == GGUFValueType.STRING: return self._read_str(f)
        elif vtype == GGUFValueType.ARRAY:
            arr_type = GGUFValueType(struct.unpack("<I", f.read(4))[0])
            count = struct.unpack("<Q", f.read(8))[0]
            return [self._read_val(f, arr_type) for _ in range(count)]
        else:
            raise GGUFParseError(f"Unknown value type: {vtype}")

    def _parse(self):
        with open(self.path, "rb") as f:
            magic = f.read(4)
            if magic != b"GGUF":
                raise GGUFParseError(f"Invalid magic: {magic!r}")

            version = struct.unpack("<I", f.read(4))[0]
            if version not in (2, 3):
                raise GGUFParseError(f"Unsupported GGUF version: {version}")

            tensor_count = struct.unpack("<Q", f.read(8))[0]
            metadata_kv_count = struct.unpack("<Q", f.read(8))[0]

            for _ in range(metadata_kv_count):
                key = self._read_str(f)
                vtype = GGUFValueType(struct.unpack("<I", f.read(4))[0])
                self.metadata[key] = self._read_val(f, vtype)

            for _ in range(tensor_count):
                name = self._read_str(f)
                n_dims = struct.unpack("<I", f.read(4))[0]
                shape = struct.unpack(f"<{n_dims}Q", f.read(8 * n_dims))
                dtype = GGMLType(struct.unpack("<I", f.read(4))[0])
                offset = struct.unpack("<Q", f.read(8))[0]

                num_elements = 1
                for d in shape: num_elements *= d

                block_size = self.TYPE_BLOCK_SIZES.get(dtype, 1)
                block_elems = self.TYPE_BLOCK_ELEMENTS.get(dtype, 1)
                n_blocks = (num_elements + block_elems - 1) // block_elems
                size = n_blocks * block_size

                self.tensors[name] = TensorInfo(
                    name=name, n_dims=n_dims, shape=shape,
                    dtype=dtype, offset=offset, size=size
                )

            self.tensor_data_offset = f.tell()
            align = 32
            padding = (align - (self.tensor_data_offset % align)) % align
            self.tensor_data_offset += padding

    def get_tensor_raw(self, name: str) -> bytes:
        info = self.tensors[name]
        with open(self.path, "rb") as f:
            f.seek(self.tensor_data_offset + info.offset)
            return f.read(info.size)

    def dequantize_q4_0(self, name: str) -> Tuple[List[float], Tuple[int, ...]]:
        info = self.tensors[name]
        raw = self.get_tensor_raw(name)
        shape = info.shape
        num_elements = 1
        for d in shape: num_elements *= d

        result = []
        pos = 0
        for _ in range((num_elements + 31) // 32):
            delta = struct.unpack("<e", raw[pos:pos+2])[0]
            pos += 2
            qs = raw[pos:pos+16]
            pos += 16
            for i in range(32):
                byte = qs[i // 2]
                nibble = (byte >> 4) if (i % 2) else (byte & 0x0F)
                result.append(delta * (nibble - 8))
        return result[:num_elements], shape

    def dequantize_q8_0(self, name: str) -> Tuple[List[float], Tuple[int, ...]]:
        info = self.tensors[name]
        raw = self.get_tensor_raw(name)
        shape = info.shape
        num_elements = 1
        for d in shape: num_elements *= d

        result = []
        pos = 0
        for _ in range((num_elements + 31) // 32):
            delta = struct.unpack("<e", raw[pos:pos+2])[0]
            pos += 2
            qs = raw[pos:pos+32]
            pos += 32
            for q in qs:
                result.append(delta * q)
        return result[:num_elements], shape

    def dequantize_f16(self, name: str) -> Tuple[List[float], Tuple[int, ...]]:
        info = self.tensors[name]
        raw = self.get_tensor_raw(name)
        shape = info.shape
        num_elements = len(raw) // 2
        floats = [struct.unpack("<e", raw[i*2:(i+1)*2])[0] for i in range(num_elements)]
        return floats, shape

    def dequantize_q6_k(self, name: str) -> Tuple[List[float], Tuple[int, ...]]:
        info = self.tensors[name]
        raw = self.get_tensor_raw(name)
        shape = info.shape
        num_elements = 1
        for d in shape: num_elements *= d

        result = []
        pos = 0
        QK_K = 256
        for _ in range((num_elements + QK_K - 1) // QK_K):
            d = struct.unpack("<e", raw[pos:pos+2])[0]
            pos += 2
            scales = list(raw[pos:pos+16])
            pos += 16
            ql = raw[pos:pos+128]
            pos += 128
            qh = raw[pos:pos+64]
            pos += 64

            for j in range(16):
                sc = scales[j]
                for l in range(16):
                    idx = j * 16 + l
                    if idx >= QK_K:
                        break
                    q4_low = ql[(j * 16 + l) // 2]
                    qh_byte = qh[(j * 16 + l) // 4]
                    shift = (l % 4) * 2
                    val = ((q4_low & 0xF) | (((qh_byte >> shift) & 3) << 4)) - 32
                    result.append(d * sc * val)
        return result[:num_elements], shape

    def dequantize(self, name: str) -> Tuple[List[float], Tuple[int, ...]]:
        dtype = self.tensors[name].dtype
        if dtype == GGMLType.Q4_0:
            return self.dequantize_q4_0(name)
        elif dtype == GGMLType.Q8_0:
            return self.dequantize_q8_0(name)
        elif dtype == GGMLType.Q6_K:
            return self.dequantize_q6_k(name)
        elif dtype == GGMLType.F16:
            return self.dequantize_f16(name)
        elif dtype == GGMLType.F32:
            info = self.tensors[name]
            raw = self.get_tensor_raw(name)
            shape = info.shape
            num_elements = len(raw) // 4
            floats = [struct.unpack("<f", raw[i*4:(i+1)*4])[0] for i in range(num_elements)]
            return floats, shape
        else:
            raise NotImplementedError(f"Dequantization for {dtype.name} not yet implemented")

    def build_buffer_descs(self) -> Dict[str, BufferDesc]:
        descs = {}
        for name, info in self.tensors.items():
            descs[name] = BufferDesc(
                name=name,
                size_bytes=info.size,
                quant_format=info.dtype,
                shape=info.shape,
                scale=1.0
            )
        return descs

    def get_vocab(self) -> List[str]:
        """Extract vocabulary tokens from GGUF metadata."""
        # Common GGUF metadata keys for vocab
        for key in ["tokenizer.ggml.tokens", "tokenizer.tokens", "vocab"]:
            if key in self.metadata:
                return self.metadata[key]
        return []

    def get_merges(self) -> List[str]:
        """Extract BPE merge rules from GGUF metadata."""
        for key in ["tokenizer.ggml.merges", "tokenizer.merges", "merges"]:
            if key in self.metadata:
                return self.metadata[key]
        return []

    def get_special_tokens(self) -> Dict[str, int]:
        """Get special token IDs."""
        tokens = {}
        for key, mapped in [
            ("tokenizer.ggml.bos_token_id", "bos"),
            ("tokenizer.ggml.eos_token_id", "eos"),
            ("tokenizer.ggml.pad_token_id", "pad"),
            ("tokenizer.ggml.unknown_token_id", "unk"),
        ]:
            if key in self.metadata:
                tokens[mapped] = self.metadata[key]
        return tokens

    def summary(self) -> str:
        lines = [
            f"GGUF: {self.path}",
            f"Metadata: {len(self.metadata)} KV pairs",
            f"Tensors: {len(self.tensors)}",
            f"Data offset: {self.tensor_data_offset}",
            f"Vocab size: {len(self.get_vocab())}",
            f"Merges: {len(self.get_merges())}",
            "",
            "Key tensors:",
        ]
        for name in list(self.tensors.keys())[:10]:
            t = self.tensors[name]
            size_mb = t.size / (1024 * 1024)
            lines.append(f"  {name}: {t.dtype.name} {t.shape} ({size_mb:.2f} MB)")
        if len(self.tensors) > 10:
            lines.append(f"  ... and {len(self.tensors) - 10} more")
        return "\n".join(lines)


def main():
    import sys
    if len(sys.argv) < 2:
        print("Usage: python gguf_loader.py <model.gguf>")
        print("       python gguf_loader.py <model.gguf> --info")
        print("       python gguf_loader.py <model.gguf> --dump <tensor_name>")
        print("       python gguf_loader.py <model.gguf> --vocab")
        sys.exit(1)

    path = sys.argv[1]
    loader = GGUFLoader(path)

    if "--info" in sys.argv:
        print(loader.summary())
        print("\nMetadata:")
        for k, v in loader.metadata.items():
            print(f"  {k}: {v}")
    elif "--dump" in sys.argv:
        idx = sys.argv.index("--dump")
        name = sys.argv[idx + 1]
        data, shape = loader.dequantize(name)
        print(f"Tensor '{name}' shape={shape}, len={len(data)}")
        print(f"First 10 values: {data[:10]}")
    elif "--vocab" in sys.argv:
        vocab = loader.get_vocab()
        print(f"Vocabulary size: {len(vocab)}")
        print(f"First 20 tokens: {vocab[:20]}")
        merges = loader.get_merges()
        print(f"Merges: {len(merges)}")
        print(f"First 10 merges: {merges[:10]}")
        special = loader.get_special_tokens()
        print(f"Special tokens: {special}")
    else:
        print(loader.summary())


if __name__ == "__main__":
    main()
