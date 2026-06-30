import struct, sys

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

# The GGUF converter defines Q4_K block_size=144, block_elements=256
# Standard ggml Q4_K layout (from ggml-quants.h):
#   qs[128] + scales_lo[32] + scales_hi[32] -- no wait
# Let me just try the standard GGML layout and see which gives sane values

# Layout A: d(2) + dmin(2) + scales_int8(8) + qs(128) = 140 (padded to 144?)
# Layout B: qs(128) + scales_int8(8) + d(2) + dmin(2) = 140
# Layout C: d(2) + dmin(2) + scales(16) + qs(128) = 148 (too big)

# Actually the REAL ggml block_q4_k layout is:
# typedef struct {
#     uint8_t qs[256/2];     // 128 bytes
#     uint8_t scales[256/32]; // 8 bytes  
#     ggml_half d;            // 2 bytes
#     ggml_half dmin;         // 2 bytes
# } block_q4_k;  // = 140 bytes
#
# But the GGUF file says block_size=144. So there must be padding or different version.

# Let me check ALL possible positions for d/dmin/scales and see what gives reasonable values

for b in range(3):
    bs = kOff + b * 144
    print(f"\n=== Block {b} at offset {bs} ===")
    
    # Try layout: qs(128) at 0, scales(8 int8) at 128, d(2) at 136, dmin(2) at 138, pad(6) to 144
    qs = binData[bs:bs+128]
    sc = [int(x) for x in binData[bs+128:bs+136]]
    d_raw = struct.unpack_from('<H', binData, bs+136)[0]
    dmin_raw = struct.unpack_from('<H', binData, bs+138)[0]
    d_val = fp16(d_raw)
    dmin_val = fp16(dmin_raw)
    print(f"  Layout qs(128)+sc(8)+d(2)+dmin(2):")
    print(f"    d=0x{d_raw:04x}={d_val:.8f} dmin=0x{dmin_raw:04x}={dmin_val:.8f}")
    print(f"    scales={sc}")
    # Decode first 4 values
    for j in range(4):
        n_lo = qs[j] & 0x0F
        n_hi = (qs[j] >> 4) & 0x0F
        sb = j // 16  # which 32-element sub-block
        if sb < 8:
            val = sc[sb] * ((n_lo - 8) * d_val + dmin_val)
            print(f"    elem {j}: nibble={n_lo} sc={sc[sb]} val={val:.6f}")

# Also try: d(2) + dmin(2) + scales(8) + qs(128) + pad(4) = 144
print("\n--- Alt layout: d(2)+dmin(2)+sc(8)+qs(128)+pad(4) ---")
for b in range(3):
    bs = kOff + b * 144
    d_raw = struct.unpack_from('<H', binData, bs)[0]
    dmin_raw = struct.unpack_from('<H', binData, bs+2)[0]
    sc = [int(x) for x in binData[bs+4:bs+12]]
    qs = binData[bs+12:bs+140]
    d_val = fp16(d_raw)
    dmin_val = fp16(dmin_raw)
    print(f"  Block {b}: d=0x{d_raw:04x}={d_val:.8f} dmin=0x{dmin_raw:04x}={dmin_val:.8f}")
    print(f"    scales={sc}")
    for j in range(4):
        n_lo = qs[j] & 0x0F
        n_hi = (qs[j] >> 4) & 0x0F
        sb = j // 16
        if sb < 8:
            val = sc[sb] * ((n_lo - 8) * d_val + dmin_val)
            print(f"    elem {j}: nibble={n_lo} sc={sc[sb]} val={val:.6f}")

# Try: d(2) + dmin(2) + scales_int16(16) + qs(128) = 148 ... but block is 144
# Try: d(2) + scales_int8(8) + dmin(2) + qs(128) + pad(4) = 144
print("\n--- Alt layout: d(2)+sc(8)+dmin(2)+qs(128)+pad(4) ---")
for b in range(3):
    bs = kOff + b * 144
    d_raw = struct.unpack_from('<H', binData, bs)[0]
    sc = [int(x) for x in binData[bs+2:bs+10]]
    dmin_raw = struct.unpack_from('<H', binData, bs+10)[0]
    qs = binData[bs+12:bs+140]
    d_val = fp16(d_raw)
    dmin_val = fp16(dmin_raw)
    print(f"  Block {b}: d=0x{d_raw:04x}={d_val:.8f} dmin=0x{dmin_raw:04x}={dmin_val:.8f}")
    print(f"    scales={sc}")
    for j in range(4):
        n_lo = qs[j] & 0x0F
        sb = j // 16
        if sb < 8:
            val = sc[sb] * ((n_lo - 8) * d_val + dmin_val)
            print(f"    elem {j}: nibble={n_lo} sc={sc[sb]} val={val:.6f}")
