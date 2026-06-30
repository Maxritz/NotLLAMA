import struct

binData = open('model/llamamobile.weights.bin', 'rb').read()
kOff = 248123648

def fp16(raw):
    sign = (raw >> 15) & 1
    exp = (raw >> 10) & 0x1F
    man = raw & 0x3FF
    if exp == 0 and man == 0: return 0.0
    if exp == 31:
        if man == 0: return float('-inf') if sign else float('inf')
        return float('nan')
    if exp == 0:
        while not (man & 0x400): man <<= 1; exp -= 1
        exp += 1; man &= 0x3FF
    exp += 127 - 15
    return struct.unpack('f', struct.pack('I', (sign<<31)|(exp<<23)|(man<<13)))[0]

# Q4_K layout per ggml-quants.h:
# block_q4_k {
#     uint8_t qs[QK_K/2];      // 128 bytes of 4-bit values
#     uint8_t scales[QK_K/32]; // 8 bytes of int8 scales  
#     ggml_half d;              // 2 bytes fp16
#     ggml_half dmin;           // 2 bytes fp16
# };
# = 128 + 8 + 2 + 2 = 140 bytes
#
# But GGUF says block_size=144 (4 bytes padding at end)
# Layout: qs[128] @0, scales[8] @128, d(fp16) @136, dmin(fp16) @138, pad[6] @140
#
# Dequant: val = d * scales[sub_block] * (nibble - 8) + dmin
# where nibble is 4-bit value (0-15) centered at 8

print("=== Testing layout: qs[128] @0, scales[8] @128, d(fp16) @136, dmin(fp16) @138 ===")
for b in range(5):
    bs = kOff + b * 144
    d = fp16(struct.unpack_from('<H', binData, bs+136)[0])
    dmin = fp16(struct.unpack_from('<H', binData, bs+138)[0])
    sc = list(binData[bs+128:bs+136])  # 8 int8 scales
    qs = binData[bs:bs+128]
    
    print(f"Block {b}: d={d:.8f} dmin={dmin:.8f} scales={sc}")
    
    vals = []
    for j in range(32):  # first sub-block (32 elements)
        byte_idx = j // 2
        nib = (qs[byte_idx] >> (4 * (j & 1))) & 0x0F
        val = d * sc[0] * (nib - 8) + dmin
        vals.append(val)
    print(f"  First 32 values: min={min(vals):.6f} max={max(vals):.6f} avg={sum(vals)/len(vals):.6f}")
    print(f"  First 8: {vals[:8]}")
    
# Also test: maybe it's d * (sc * (nibble - 8) + dmin) or d * sc * (nibble - 16)
print("\n=== Testing nibble-16 variant ===")
for b in range(3):
    bs = kOff + b * 144
    d = fp16(struct.unpack_from('<H', binData, bs+136)[0])
    dmin = fp16(struct.unpack_from('<H', binData, bs+138)[0])
    sc = list(binData[bs+128:bs+136])
    qs = binData[bs:bs+128]
    
    vals = []
    for j in range(32):
        byte_idx = j // 2
        nib = (qs[byte_idx] >> (4 * (j & 1))) & 0x0F
        val = d * sc[0] * (nib - 16) + dmin
        vals.append(val)
    print(f"Block {b}: nib-16: min={min(vals):.6f} max={max(vals):.6f}")
    
# Test: maybe d is at end (offset 140) with 144-byte block
print("\n=== Testing: d @ 140, dmin @ 142, scales @ 128, qs @ 0 ===")
for b in range(5):
    bs = kOff + b * 144
    d = fp16(struct.unpack_from('<H', binData, bs+140)[0])
    dmin = fp16(struct.unpack_from('<H', binData, bs+142)[0])
    sc = list(binData[bs+128:bs+140])  # 12 bytes
    qs = binData[bs:bs+128]
    
    print(f"Block {b}: d={d:.8f} dmin={dmin:.8f}")
    # Try scales as 8 int8 values (first 8 of 12)
    vals = []
    for j in range(32):
        byte_idx = j // 2
        nib = (qs[byte_idx] >> (4 * (j & 1))) & 0x0F
        val = d * sc[0] * (nib - 8) + dmin
        vals.append(val)
    print(f"  First 32: min={min(vals):.6f} max={max(vals):.6f}")
