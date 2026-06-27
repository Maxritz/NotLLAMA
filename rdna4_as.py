#!/usr/bin/env python3
"""rdna4_as.py — Minimal SPIR-V assembler for RDNA4 compute kernels."""
import argparse, struct

# SPIR-V opcodes (subset)
OP_NAME, OP_EXT_INST_IMPORT, OP_MEMORY_MODEL, OP_ENTRY_POINT, OP_EXECUTION_MODE, OP_CAPABILITY = 5, 11, 14, 15, 17, 17
OP_TYPE_VOID, OP_TYPE_INT, OP_TYPE_FLOAT, OP_TYPE_VECTOR, OP_TYPE_STRUCT, OP_TYPE_POINTER, OP_TYPE_FUNCTION = 19, 21, 22, 23, 30, 32, 33
OP_CONSTANT, OP_CONSTANT_COMPOSITE, OP_FUNCTION, OP_FUNCTION_END, OP_VARIABLE, OP_LOAD, OP_STORE = 43, 44, 54, 56, 59, 61, 62
OP_ACCESS_CHAIN, OP_BITCAST, OP_CONVERT_FTOF, OP_FMUL = 65, 124, 112, 130
OP_LABEL, OP_RETURN, OP_UADD = 248, 253, 128

# Capabilities / enums
CAP_SHADER, CAP_FLOAT16, CAP_INT64, CAP_PHYSICAL_STORAGE_BUFFER = 1, 9, 11, 5347
EXEC_GL_COMPUTE, MODE_LOCAL_SIZE = 5, 17
STORAGE_INPUT, STORAGE_PUSH_CONSTANT, STORAGE_PHYSICAL_STORAGE_BUFFER = 1, 9, 5349
MEMORY_PHYSICAL_STORAGE_BUFFER_64, ADDRESS_PHYSICAL_STORAGE_BUFFER_64 = 5348, 5348

class Builder:
    def __init__(self):
        self.words, self.bound, self.ids = [], 1, {}
    def id(self, name=None):
        i = self.bound; self.bound += 1
        if name: self.ids[name] = i
        return i
    def get(self, n): return self.ids[n]
    def emit(self, op, ops):
        self.words.append(((1+len(ops))<<16)|op); self.words.extend(ops)
    def emit_str(self, op, ops, s):
        b = s.encode()+b'\x00'; b += b'\x00'*((4-len(b)%4)%4)
        self.words.append(((1+len(ops)+len(b)//4)<<16)|op)
        self.words.extend(ops)
        for i in range(0,len(b),4): self.words.append(struct.unpack("<I",b[i:i+4])[0])
    def build(self):
        h = [0x07230203, 0x00010300, 0, self.bound, 0]
        return b''.join(struct.pack("<I",w) for w in h+self.words)

def assemble():
    b = Builder()
    # Capabilities
    for c in [CAP_SHADER, CAP_PHYSICAL_STORAGE_BUFFER, CAP_FLOAT16, CAP_INT64]:
        b.emit(OP_CAPABILITY, [c])
    ext = b.id("ext")
    b.emit_str(OP_EXT_INST_IMPORT, [ext], "SPV_KHR_physical_storage_buffer")
    b.emit(OP_MEMORY_MODEL, [ADDRESS_PHYSICAL_STORAGE_BUFFER_64, 1])
    # Types
    void_t = b.id("void"); b.emit(OP_TYPE_VOID, [void_t])
    fn_t = b.id("fn"); b.emit(OP_TYPE_FUNCTION, [fn_t, void_t])
    u32 = b.id("u32"); b.emit(OP_TYPE_INT, [u32, 32, 0])
    u64 = b.id("u64"); b.emit(OP_TYPE_INT, [u64, 64, 0])
    f32 = b.id("f32"); b.emit(OP_TYPE_FLOAT, [f32, 32])
    f16 = b.id("f16"); b.emit(OP_TYPE_FLOAT, [f16, 16])
    v3u32 = b.id("v3u32"); b.emit(OP_TYPE_VECTOR, [v3u32, u32, 3])
    ptr_f16 = b.id("ptr_f16"); b.emit(OP_TYPE_POINTER, [ptr_f16, STORAGE_PHYSICAL_STORAGE_BUFFER, f16])
    ptr_f32 = b.id("ptr_f32"); b.emit(OP_TYPE_POINTER, [ptr_f32, STORAGE_PHYSICAL_STORAGE_BUFFER, f32])
    push_t = b.id("Push"); b.emit(OP_TYPE_STRUCT, [push_t, u64, u64, u32, u32, f32])
    b.emit_str(OP_NAME, [push_t], "PushConstants")
    ptr_push = b.id("ptr_push"); b.emit(OP_TYPE_POINTER, [ptr_push, STORAGE_PUSH_CONSTANT, push_t])
    # Constants
    c0 = b.id("c0"); b.emit(OP_CONSTANT, [u32, c0, 0])
    c1 = b.id("c1"); b.emit(OP_CONSTANT, [u32, c1, 1])
    c2 = b.id("c2"); b.emit(OP_CONSTANT, [u32, c2, 2])
    c4 = b.id("c4"); b.emit(OP_CONSTANT, [u32, c4, 4])
    c0_64 = b.id("c0_64"); b.emit(OP_CONSTANT, [u64, c0_64, 0, 0])
    c256 = b.id("c256"); b.emit(OP_CONSTANT, [u32, c256, 256])
    wgs = b.id("wgs"); b.emit(OP_CONSTANT_COMPOSITE, [v3u32, wgs, c256, c1, c1])
    # Variables
    ptr_in_u32 = b.id("ptr_in_u32"); b.emit(OP_TYPE_POINTER, [ptr_in_u32, STORAGE_INPUT, u32])
    tid_var = b.id("tid"); b.emit(OP_VARIABLE, [ptr_in_u32, tid_var, STORAGE_INPUT])
    push_var = b.id("push"); b.emit(OP_VARIABLE, [ptr_push, push_var, STORAGE_PUSH_CONSTANT])
    # Entry point
    main = b.id("main"); b.emit_str(OP_ENTRY_POINT, [EXEC_GL_COMPUTE, main], "main")
    b.emit(OP_EXECUTION_MODE, [main, MODE_LOCAL_SIZE, 256, 1, 1])
    # Function
    b.emit(OP_FUNCTION, [void_t, main, 0, fn_t])
    entry = b.id("entry"); b.emit(OP_LABEL, [entry])
    # tid = load tid_var
    tid_val = b.id("tid_val"); b.emit(OP_LOAD, [u32, tid_val, tid_var, 0x2, 0x4])
    # load count (member 2)
    ptr_u32 = b.id("ptr_u32"); b.emit(OP_TYPE_POINTER, [ptr_u32, STORAGE_PUSH_CONSTANT, u32])
    cnt_ptr = b.id("cnt_ptr"); b.emit(OP_ACCESS_CHAIN, [ptr_u32, cnt_ptr, push_var, c2])
    cnt = b.id("cnt"); b.emit(OP_LOAD, [u32, cnt, cnt_ptr, 0x2, 0x4])
    # load addr_in (member 0, u64)
    ptr_u64 = b.id("ptr_u64"); b.emit(OP_TYPE_POINTER, [ptr_u64, STORAGE_PUSH_CONSTANT, u64])
    ain_ptr = b.id("ain_ptr"); b.emit(OP_ACCESS_CHAIN, [ptr_u64, ain_ptr, push_var, c0])
    ain = b.id("ain"); b.emit(OP_LOAD, [u64, ain, ain_ptr, 0x2, 0x8])
    # load addr_out (member 1, u64)
    aout_ptr = b.id("aout_ptr"); b.emit(OP_ACCESS_CHAIN, [ptr_u64, aout_ptr, push_var, c1])
    aout = b.id("aout"); b.emit(OP_LOAD, [u64, aout, aout_ptr, 0x2, 0x8])
    # load scale (member 4, f32)
    ptr_f32_pc = b.id("ptr_f32_pc"); b.emit(OP_TYPE_POINTER, [ptr_f32_pc, STORAGE_PUSH_CONSTANT, f32])
    scl_ptr = b.id("scl_ptr"); b.emit(OP_ACCESS_CHAIN, [ptr_f32_pc, scl_ptr, push_var, c4])
    scl = b.id("scl"); b.emit(OP_LOAD, [f32, scl, scl_ptr, 0x2, 0x4])
    # in_ptr = bitcast addr_in -> f16*
    in_ptr = b.id("in_ptr"); b.emit(OP_BITCAST, [ptr_f16, in_ptr, ain])
    # val_f16 = in_ptr[tid]
    ep_f16 = b.id("ep_f16"); b.emit(OP_ACCESS_CHAIN, [ptr_f16, ep_f16, in_ptr, tid_val])
    v16 = b.id("v16"); b.emit(OP_LOAD, [f16, v16, ep_f16, 0x2, 0x2])
    # f16 -> f32
    v32 = b.id("v32"); b.emit(OP_CONVERT_FTOF, [f32, v32, v16])
    # scaled = v32 * scale
    scaled = b.id("scaled"); b.emit(OP_FMUL, [f32, scaled, v32, scl])
    # out_ptr = bitcast addr_out -> f32*
    out_ptr = b.id("out_ptr"); b.emit(OP_BITCAST, [ptr_f32, out_ptr, aout])
    # out_ptr[tid] = scaled
    ep_f32 = b.id("ep_f32"); b.emit(OP_ACCESS_CHAIN, [ptr_f32, ep_f32, out_ptr, tid_val])
    b.emit(OP_STORE, [ep_f32, scaled, 0x2, 0x4])
    b.emit(OP_RETURN, [])
    b.emit(OP_FUNCTION_END, [])
    return b.build()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("source", nargs="?", help="Ignored for now")
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--arch", default="gfx1200")
    args = p.parse_args()
    spirv = assemble()
    with open(args.output, "wb") as f: f.write(spirv)
    print(f"Emitted {len(spirv)} bytes -> {args.output} (bound={46})")

if __name__ == "__main__": main()