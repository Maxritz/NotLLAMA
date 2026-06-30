import struct, json, sys, numpy as np

# Load model metadata
with open("build/model/VibeThinker-3B.Q6_K.weights.json") as f:
    meta = json.load(f)

# Find token_embd.weight tensor
emb = None
for t in meta["tensors"]:
    if t["name"] == "token_embd.weight":
        emb = t
        break

print(f"token_embd.weight: shape={emb['shape']} dtype={emb['dtype_id']} bin_offset={emb['bin_offset']} size_bytes={emb['size_bytes']}")

# Load raw bin data
with open("build/model/VibeThinker-3B.Q6_K.weights.bin", "rb") as f:
    bin_data = f.read()

bin_offset = emb["bin_offset"]
# The shader reads from gpuAddress which is the start of the per-tensor buffer
# In the uploader, the data is: binData[bin_offset .. bin_offset+size_bytes]
tensor_data = bin_data[bin_offset : bin_offset + emb["size_bytes"]]
print(f"Tensor data loaded: {len(tensor_data)} bytes")

# Q6_K: 256 elements per block, 210 bytes per block
# Block layout: d(f16)@0, scales(int8)@2, ql(128B)@18, qh(64B)@146

def dequant_q6k_element(tensor_data, elem_idx):
    """Dequantize a single element from Q6_K data"""
    block_idx = elem_idx // 256
    ele_in_block = elem_idx % 256
    bs = block_idx * 210

    # Read d (float16 scale)
    d_bytes = tensor_data[bs:bs+2]
    d = np.frombuffer(d_bytes, dtype=np.float16)[0]

    # Read scale
    sub_block = ele_in_block // 16
    sc_raw = tensor_data[bs + 2 + sub_block]
    sc = sc_raw if sc_raw < 128 else sc_raw - 256

    # Read ql (low bits)
    ql_byte_idx = ele_in_block // 2
    q4_raw = tensor_data[bs + 18 + ql_byte_idx]
    q4 = (q4_raw & 0xF) if (ele_in_block & 1) == 0 else (q4_raw >> 4)

    # Read qh (high bits)
    qh_byte_idx = ele_in_block // 4
    qh_byte = tensor_data[bs + 146 + qh_byte_idx]
    qh_shift = (ele_in_block & 3) * 2

    val = (q4 | (((qh_byte >> qh_shift) & 3) << 4)) - 32

    result = d * sc * val
    return d, sc, val, result

# Check element 0 (token 0, dim 0)
d0, sc0, val0, r0 = dequant_q6k_element(tensor_data, 0)
print(f"\nelem 0: d={float(d0):.6f} sc={sc0} val={val0} result={float(r0):.6f}")

# Check element 310542336 (token 151643, dim 0)
idx = 151643 * 2048
d1, sc1, val1, r1 = dequant_q6k_element(tensor_data, idx)
print(f"elem {idx}: d={float(d1):.6f} sc={sc1} val={val1} result={float(r1):.6f}")

# Check a few more near token 151643
for off in range(8):
    idx2 = idx + off
    d2, sc2, val2, r2 = dequant_q6k_element(tensor_data, idx2)
    f16_bytes = np.float16(r2).tobytes()
    f16_val = struct.unpack('<H', f16_bytes)[0]
    print(f"  elem {idx2}: d={float(d2):.6f} sc={sc2} val={val2} result={float(r2):.6f} f16=0x{f16_val:04x}")

# Now check what the GPU output says
print(f"\nExpected for token 151643 [0..7]:")
for off in range(8):
    idx2 = idx + off
    d2, sc2, val2, r2 = dequant_q6k_element(tensor_data, idx2)
    print(f"  elem {idx2}: {float(r2):.6f}")

# Also verify the block index calculations are correct
block_idx = idx // 256
ele_in_block = idx % 256
bs = block_idx * 210
print(f"\nBlock idx: {block_idx}, ele in block: {ele_in_block}, byte offset: {bs}")
print(f"Max byte offset needed: {bs + 210 - 1}, tensor size: {len(tensor_data)}")
