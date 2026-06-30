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

# Correct layout per tools/AGENTS.md:
# qs[128] @ 0, scales[12] @ 128, d(fp16) @ 140, dmin(fp16) @ 142
# Scales: 8 sub-blocks, stored as int8, remaining 4 bytes unused?

for b in range(5):
    bs = kOff + b * 144
    d_raw = struct.unpack_from('<H', binData, bs+140)[0]
    dmin_raw = struct.unpack_from('<H', binData, bs+142)[0]
    d_val = fp16(d_raw)
    dmin_val = fp16(dmin_raw)
    
    # Read 12 scale bytes at offset 128
    scale_bytes = list(binData[bs+128:bs+140])
    
    print(f"Block {b}: d=0x{d_raw:04x}={d_val:.8f} dmin=0x{dmin_raw:04x}={dmin_val:.8f}")
    print(f"  scales[0..11] @ 128: {scale_bytes}")
    
    # Decode first 8 elements (2 nibbles per byte, sub-block 0 = first 16 elements)
    qs = binData[bs:bs+128]
    for j in range(8):
        bval = qs[j]
        n_lo = bval & 0x0F
        n_hi = (bval >> 4) & 0x0F
        sb = j * 2 // 32  # sub-block (0 for first 16 elements)
        # Wait: 32 elements per sub-block, each byte has 2 elements
        # So elements 0-31 are sub-block 0, elements 32-63 are sub-block 1, etc.
        elem_lo = j * 2
        elem_hi = j * 2 + 1
        sb_lo = elem_lo // 32
        sb_hi = elem_hi // 32
        sc_lo = scale_bytes[sb_lo] if sb_lo < 8 else 0
        sc_hi = scale_bytes[sb_hi] if sb_hi < 8 else 0
        val_lo = (sc_lo / 255.0) * ((n_lo - 8) * d_val + dmin_val) if False else d_val * sc_lo * (n_lo - 8) + dmin_val
        val_hi = d_val * sc_hi * (n_hi - 8) + dmin_val
        print(f"    elem[{elem_lo}] = d*sc*(n-8)+dmin = {d_val:.6f}*{sc_lo}*({n_lo}-8)+{dmin_val:.6f} = {val_lo:.6f}")
        print(f"    elem[{elem_hi}] = d*sc*(n-8)+dmin = {d_val:.6f}*{sc_hi}*({n_hi}-8)+{dmin_val:.6f} = {val_hi:.6f}")
    print()
