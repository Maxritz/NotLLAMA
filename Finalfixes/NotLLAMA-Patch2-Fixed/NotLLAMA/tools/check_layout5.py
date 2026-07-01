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

# Layout: d(2) @ 0, dmin(2) @ 2, scales(12) @ 4, qs(128) @ 16
# Total: 2+2+12+128 = 144 ✓
print("=== Layout: d@0, dmin@2, scales_int8(12)@4, qs(128)@16 ===")
for b in range(5):
    bs = kOff + b * 144
    d = fp16(struct.unpack_from('<H', binData, bs)[0])
    dmin = fp16(struct.unpack_from('<H', binData, bs+2)[0])
    sc_bytes = list(binData[bs+4:bs+16])  # 12 bytes
    qs = binData[bs+16:bs+144]  # 128 bytes
    
    print(f"Block {b}: d={d:.8f} dmin={dmin:.8f}")
    print(f"  scales bytes: {sc_bytes}")
    
    # First 32 elements: sub-block 0
    vals = []
    for j in range(32):
        byte_idx = j // 2
        nib = (qs[byte_idx] >> (4 * (j & 1))) & 0x0F
        # 8 sub-blocks of 32 elements, scale is int8
        sb = j // 32
        sc = sc_bytes[sb]
        val = d * sc * (nib - 8) + dmin
        vals.append(val)
    print(f"  First 32: min={min(vals):.6f} max={max(vals):.6f} avg={sum(vals)/len(vals):.6f}")
    
# Also check: maybe scales are packed nibbles (not raw int8)
# Each scale byte has two nibbles, 6 bytes * 2 = 12 scales
# Or 12 bytes as raw int8 = 12 sub-blocks? No, 256/32=8 sub-blocks
print("\n=== Maybe scales are 12 bytes with 8 used, first 8 are int8 ===")
for b in range(3):
    bs = kOff + b * 144
    d = fp16(struct.unpack_from('<H', binData, bs)[0])
    dmin = fp16(struct.unpack_from('<H', binData, bs+2)[0])
    sc_bytes = list(binData[bs+4:bs+16])
    qs = binData[bs+16:bs+144]
    
    # Try decoding all 128 elements
    all_vals = []
    for j in range(256):
        byte_idx = j // 2
        nib = (qs[byte_idx] >> (4 * (j & 1))) & 0x0F
        sb = j // 32  # 8 sub-blocks
        sc = sc_bytes[sb] if sb < 8 else 0
        val = d * sc * (nib - 8) + dmin
        all_vals.append(val)
    
    finite = [v for v in all_vals if abs(v) < 100]
    print(f"Block {b}: {len(finite)}/{len(all_vals)} values in [-100,100], min={min(all_vals):.4f} max={max(all_vals):.4f}")

# What if qs is at offset 0 and scales are packed at the end?
# Layout: qs(128) @ 0, scales(12) @ 128, d(2) @ 140, dmin(2) @ 142
# But we already know d is at offset 0... unless d and qs overlap somehow?
# That can't be right.

# Let me check the GENUINE ggml source. The struct is:
# block_q4_k { qs[128], scales[8], d(2), dmin(2) } = 140, padded to 144
# In memory layout: qs @ 0, scales @ 128, d @ 136, dmin @ 138, pad @ 140

# But our scan says d is at offset 0 with 100% sanity. So maybe the
# converter reorganized the data?

# Let me check: is this GGUF file created by our converter or by llama.cpp?
# If our converter, it may re-pack differently.
print("\n=== Check Q8_0 K-like tensor for comparison ===")
q8k = [x for x in json.load(open('model/llamamobile.weights.json'))['tensors'] if x['name']=='blk.0.attn_q.weight'][0]
qOff = q8k['bin_offset']
qSz = q8k['bin_size']
qFmt = q8k['dtype_id']
qElems = 1
for d in q8k['shape']: qElems *= d
print(f"Q: off={qOff} sz={qSz} fmt={qFmt} elems={qElems} block_size={q8k['quant_block_size']} block_elems={q8k['quant_block_elements']}")

# Q4_0 block_size = 18 (d(2) + qs(16)), block_elems = 32
# Q8_0 block_size = 34 (d(2) + qs(32)), block_elems = 32
# So Q8_0 format ID = 8 per our enum, but what does the GGUF say?
print(f"Q dtype is {q8k['dtype']} (id={qFmt})")
# dtype 12 = Q4_K
# Wait, Q is also Q4_K? Let me check...
import json
j = json.load(open('model/llamamobile.weights.json'))
for t in j['tensors']:
    if 'attn' in t['name'] and 'blk.0' in t['name']:
        print(f"  {t['name']}: fmt={t['dtype_id']} ({t['dtype']}) bin_size={t['bin_size']} elems={sum([1]+t['shape'][1:]) if t['shape'] else 0}")
