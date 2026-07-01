import struct

# Read the weight_uploader Q4_K output from the binary
# We know the GGUF K weight at offset 248123648, 589824 bytes Q4_K
# After pre-dequant, it should be 1048576 F32 values

# Actually, let me just check the uploaded F32 data directly
# The GPU reads from the uploaded buffer. Let me check if the CPU dequant
# produces the same values as expected.

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

# Correct Q4_K dequant: d@0, dmin@2, scales(8 int8)@4, qs(128)@16
# Dequant: val = d * (float)sc * ((float)nibble - 8.0f) + dmin
print("=== Verifying Q4_K dequant on K weight (first 16 values of block 0) ===")
bs = kOff
d = fp16(struct.unpack_from('<H', binData, bs)[0])
dmin = fp16(struct.unpack_from('<H', binData, bs+2)[0])
sc_bytes = list(binData[bs+4:bs+12])  # 8 int8 scales
qs = binData[bs+16:bs+144]

print(f"d={d:.8f} dmin={dmin:.8f}")
print(f"scales={sc_bytes}")

for j in range(16):
    byte_idx = j // 2
    nib = (qs[byte_idx] >> (4 * (j & 1))) & 0x0F
    sb = j // 32
    sc = sc_bytes[sb]
    val = d * float(sc) * (float(nib) - 8.0) + dmin
    print(f"  K[{j}] = d*sc*(nib-8)+dmin = {d:.6f}*{sc}*({nib}-8)+{dmin:.6f} = {val:.6f}")

# Now check Q weight (also Q4_K format? No, let me check)
j = json.load(open('model/llamamobile.weights.json'))
q = [x for x in j['tensors'] if x['name']=='blk.0.attn_q.weight'][0]
print(f"\nQ weight: fmt={q['dtype_id']} ({q['dtype']}) block_size={q['quant_block_size']} block_elems={q['quant_block_elements']}")
k = [x for x in j['tensors'] if x['name']=='blk.0.attn_k.weight'][0]
print(f"K weight: fmt={k['dtype_id']} ({k['dtype']}) block_size={k['quant_block_size']} block_elems={k['quant_block_elements']}")

# Check: are Q and K the same format?
import json
