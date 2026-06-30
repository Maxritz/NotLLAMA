#!/usr/bin/env python3
"""Debug Q6_K block layout — compare our reader vs GGML source."""
import struct
import sys

path = r"E:\OLLAMA-Models\GGUF\acrux-500m-o1-journey-q6_k.gguf"

# Read raw GGUF to get tensor offsets
sys.path.insert(0, r"C:\Users\rr\Desktop\Notllama-loc\tools")
from gguf_loader import GGUFLoader, GGMLType

loader = GGUFLoader(path)
name = "blk.0.ffn_down.weight"
info = loader.tensors[name]
raw = loader.get_tensor_raw(name)

print(f"Tensor: {name}, shape={info.shape}, size={info.size}")
print(f"Q6_K block size: 210 bytes")
print()

# Block 0 — known good
b = 0
block = raw[b*210:(b+1)*210]
d = struct.unpack("<e", block[0:2])[0]
print(f"=== Block {b} (good) ===")
print(f"  d (fp16): {d}")
print(f"  scales: {list(block[2:18])}")
print(f"  ql[0:32]: {list(block[18:50])}")
print(f"  qh[0:16]: {list(block[146:162])}")

# Block 53 — NaN
b = 53
block = raw[b*210:(b+1)*210]
d = struct.unpack("<e", block[0:2])[0]
print(f"\n=== Block {b} (NaN) ===")
print(f"  d (fp16): {d} (nan={d!=d})")
print(f"  raw bytes[0:32]: {block[0:32].hex()}")
print(f"  scales: {list(block[2:18])}")
print(f"  ql[0:32]: {list(block[18:50])}")
print(f"  qh[0:16]: {list(block[146:162])}")

# Check: is this block layout correct?
# GGML Q6_K layout:
#   d:      fp16 (2 bytes)         offset 0
#   scales: int8[16] (16 bytes)    offset 2
#   ql:     uint8[128] (128 bytes) offset 18
#   qh:     uint8[64] (64 bytes)   offset 146
# Total: 210 bytes

# Our reader uses: block[2:18] for scales, block[18:146] for ql, block[146:210] for qh
# Let's check if this matches GGML's ggml_dequantize_row_q6_K

# From GGML source (ggml.c):
# static void ggml_dequantize_row_q6_k(const void * vx, float * y, int k) {
#     const int nb = 32;
#     ...
#     for (int i = 0; i < nc; ++i) {
#         const float d = GGML_HALF_TO_FLOAT(((const ggml_half *)vx)[0]);
#         vx = (const char *)vx + sizeof(ggml_half);
#         const int8_t * restrict scales = (const int8_t *)vx;
#         vx = (const char *)vx + 16 * sizeof(int8_t);
#         const uint8_t * restrict ql = (const uint8_t *)vx;
#         vx = (const char *)vx + 128;
#         const uint8_t * restrict qh = (const uint8_t *)vx;
#         vx = (const char *)vx + 64;
#
#         for (int j = 0; j < nb; ++j) {
#             const uint8_t q1 = ql[j*nb/4 + 0] & 0xF;
#             const uint8_t q2 = ql[j*nb/4 + 1] & 0xF;
#             const uint8_t q3 = ql[j*nb/4 + 2] & 0xF;
#             const uint8_t q4 = ql[j*nb/4 + 3] & 0xF;
#             const uint8_t q5 = qh[j] & 0xF;
#             const uint8_t q6 = (qh[j] >> 4) & 0xF;
#             const int8_t  s1 = scales[j/2 + 0];
#             const int8_t  s2 = scales[j/2 + 8];
#             ...
#         }
#     }
# }
#
# Wait — GGML iterates j from 0 to 31 (nb=32), not 0 to 15!
# And the indexing is DIFFERENT:
#   ql: j*nb/4 = j*8
#   qh: j (not j/4!)
#   scales: j/2 and j/2+8

print("\n=== GGML Q6_K layout analysis ===")
print("GGML: nc=256 elements per block, nb=32 sub-blocks of 8")
print("GGML loops j=0..31:")
print("  ql index: j*8 + (0..3)  — 4 bytes per sub-block")
print("  qh index: j             — 1 byte per sub-block")
print("  scales index: j/2 and j/2+8")
print()
print("Our reader loops j=0..15:")
print("  ql index: (j*16+l)//2   — l=0..15")
print("  qh index: (j*16+l)//4")
print("  scales index: j")
print()

# Let's try decoding with GGML's indexing
print("=== Decode block 0 with GGML indexing ===")
block = raw[0:210]
d_val = struct.unpack("<e", block[0:2])[0]
scales = block[2:18]
ql = block[18:146]
qh = block[146:210]

for j in range(32):
    q1 = ql[j*8 + 0] & 0xF
    q2 = ql[j*8 + 1] & 0xF
    q3 = ql[j*8 + 2] & 0xF
    q4 = ql[j*8 + 3] & 0xF
    q5 = qh[j] & 0xF
    q6 = (qh[j] >> 4) & 0xF
    s1 = scales[j // 2]
    s2 = scales[j // 2 + 8]
    
    v1 = (q1 | (q5 << 4)) - 32
    v2 = (q2 | (q6 << 4)) - 32
    v3 = (q3 | (q5 << 4)) - 32
    v4 = (q4 | (q6 << 4)) - 32
    
    o1 = d_val * s1 * v1
    o2 = d_val * s1 * v2
    o3 = d_val * s2 * v3
    o4 = d_val * s2 * v4
    
    if j < 4:
        print(f"  j={j}: q1={q1} q2={q2} q3={q3} q4={q4} q5={q5} q6={q6} s1={s1} s2={s2}")
        print(f"       v1={v1} v2={v2} v3={v3} v4={v4}")
        print(f"       o1={o1} o2={o2} o3={o3} o4={o4}")

# Now check: does block 53 have the same ql[0] value whether we read at offset 18 or at j*8=0?
print("\n=== Block 53 raw analysis ===")
block53 = raw[53*210:54*210]
d53 = struct.unpack("<e", block53[0:2])[0]
print(f"  d={d53} nan={d53!=d53}")
print(f"  bytes[0:4]: {block53[0:4].hex()}")

# Check if maybe our block size is wrong — what if it's NOT 210 bytes?
# Try reading at different block sizes
for bsize in [200, 208, 210, 212, 214, 220, 224]:
    pos = 53 * bsize
    if pos + 2 <= len(raw):
        d_test = struct.unpack("<e", raw[pos:pos+2])[0]
        print(f"  block_size={bsize}: d={d_test} nan={d_test!=d_test}")
