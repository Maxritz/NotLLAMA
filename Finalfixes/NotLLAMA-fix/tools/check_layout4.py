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

# Maybe the 12 scale bytes are actually: 8 int8 scales + 4 bytes padding
# Or maybe it's: scales_lo[8] + scales_hi[8] where the real scale is (scales_lo[i] | (scales_hi[i] << 4))
# Let me check ALL 4096 blocks to find which layout gives sane d values for ALL blocks

def find_d_positions(kOff, nBlocks):
    """Scan all blocks and find positions where fp16 values look like small scales"""
    candidates = {}
    for b in range(nBlocks):
        bs = kOff + b * 144
        for pos in range(0, 144, 2):
            raw = struct.unpack_from('<H', binData, bs + pos)[0]
            val = fp16(raw)
            if abs(val) > 1000 or abs(val) < 1e-8:
                continue
            # Count how many blocks have a reasonable fp16 at this position
            if pos not in candidates:
                candidates[pos] = 0
            candidates[pos] += 1
    
    print("Positions where fp16 reads are small/sane (out of %d blocks):" % nBlocks)
    for pos in sorted(candidates, key=candidates.get, reverse=True)[:10]:
        print(f"  offset {pos}: {candidates[pos]}/{nBlocks} blocks have sane d")

nBlocks = 589824 // 144
find_d_positions(kOff, nBlocks)

# Now try the best candidate positions
# Also check: what if Q4_K has a totally different structure?
# From ggml-quants.h (llama.cpp):
# The ACTUAL struct is sometimes: d at position 140, dmin at 142
# scales_lo at 128, scales_hi at 136
# qs at 0

# Let me try: qs[128] @0, scales_packed[16] @128, d(fp16) @140, dmin(fp16) @142
# With scales_packed decoded as 8 pairs of (lo4, hi4) nibbles -> 8 int8 values
print("\n=== Trying qs[128]@0, scales_packed[16]@128, d@140, dmin@142 ===")
for b in range(5):
    bs = kOff + b * 144
    d = fp16(struct.unpack_from('<H', binData, bs+140)[0])
    dmin = fp16(struct.unpack_from('<H', binData, bs+142)[0])
    packed = binData[bs+128:bs+144]
    
    # Decode scales: each byte has two 4-bit values
    scales = []
    for i in range(8):
        lo = packed[i] & 0x0F
        hi = (packed[i] >> 4) & 0x0F
        scales.append(lo)
        scales.append(hi)
    
    print(f"Block {b}: d={d:.8f} dmin={dmin:.8f}")
    print(f"  packed scales bytes: {list(packed[:8])}")
    print(f"  decoded scales: {scales[:8]}")
    
    qs = binData[bs:bs+128]
    vals = []
    for j in range(32):
        byte_idx = j // 2
        nib = (qs[byte_idx] >> (4 * (j & 1))) & 0x0F
        sc = scales[j // 16]  # 16 elements per sub-block -> 2 sub-blocks per 32
        val = d * (sc - 8) * (nib - 8) + dmin
        vals.append(val)
    print(f"  First 32: min={min(vals):.6f} max={max(vals):.6f} avg={sum(vals)/len(vals):.6f}")
