#!/usr/bin/env python3
"""rdna4_as.py — Minimal working SPIR-V assembler for RDNA4 compute.

Generates valid SPIR-V 1.3 compute shaders with:
  - PhysicalStorageBuffer (buffer_device_address)
  - Push constants
  - Subgroup shuffle operations (for DPP-style reductions)
  - No external compiler needed (glslang-free)

Usage:
    python rdna4_as.py --test -o kernel.spv
"""
import struct
import sys
import argparse
from typing import List, Dict, Tuple
from dataclasses import dataclass, field
from enum import IntEnum

SPIRV_MAGIC = 0x07230203
SPIRV_VERSION_1_3 = 0x00010300

class Op(IntEnum):
    Name = 5; MemberName = 6; Decorate = 71; MemberDecorate = 72
    ExtInstImport = 11; MemoryModel = 14; EntryPoint = 15
    ExecutionMode = 16; Capability = 17; TypeVoid = 19
    TypeBool = 20; TypeInt = 21; TypeFloat = 22; TypeVector = 23
    TypeRuntimeArray = 29; TypeStruct = 30; TypePointer = 32; TypeFunction = 33
    Constant = 43; ConstantNull = 46; Function = 54
    FunctionParameter = 55; FunctionEnd = 56; Variable = 59
    Load = 61; Store = 62; AccessChain = 65; Bitcast = 124
    IAdd = 128; FAdd = 129; ISub = 130; FSub = 131
    IMul = 132; FMul = 133; SDiv = 134; FDiv = 135
    UMod = 137; SNegate = 126; FNegate = 127
    ShiftLeftLogical = 139; ShiftRightLogical = 140
    BitwiseAnd = 163; BitwiseOr = 161; BitwiseXor = 162
    Not = 164; IEqual = 170; INotEqual = 171
    UGreaterThan = 172; SGreaterThan = 173; SLessThan = 177
    FOrdEqual = 180; FOrdLessThan = 184; FOrdGreaterThan = 186
    FOrdLessThanEqual = 188; FOrdGreaterThanEqual = 190
    Label = 248; Branch = 249; BranchConditional = 250
    Return = 253; Phi = 245; LoopMerge = 246; SelectionMerge = 247
    ConvertFToU = 109; ConvertUToF = 112; ConvertFToS = 110
    ConvertSToF = 111; ConvertUToPtr = 118; ConvertPtrToU = 117; UConvert = 113; SConvert = 114; FConvert = 115
    ControlBarrier = 224; MemoryBarrier = 225
    GroupNonUniformBroadcastFirst = 337; GroupNonUniformBallot = 338
    GroupNonUniformShuffle = 344; GroupNonUniformShuffleXor = 345
    GroupNonUniformShuffleUp = 346; GroupNonUniformShuffleDown = 347
    GroupNonUniformIAdd = 348; GroupNonUniformFAdd = 349
    GroupNonUniformFMin = 354; GroupNonUniformFMax = 356
    AtomicLoad = 227; AtomicStore = 228; AtomicExchange = 229
    AtomicIAdd = 233; AtomicAnd = 239; AtomicOr = 240; AtomicXor = 241

class Capability(IntEnum):
    Shader = 1; Float16 = 9; Int16 = 22; Int64 = 11
    Float64 = 10; GroupNonUniform = 61; GroupNonUniformVote = 62
    GroupNonUniformArithmetic = 63; GroupNonUniformBallot = 64
    GroupNonUniformShuffle = 65; GroupNonUniformShuffleRelative = 66
    VulkanMemoryModel = 4425; PhysicalStorageBufferAddresses = 5347

class Decoration(IntEnum):
    Block = 2; BufferBlock = 3; ArrayStride = 6; BuiltIn = 11
    Flat = 14; Location = 30; Binding = 33; DescriptorSet = 34
    Offset = 35; NonWritable = 24; NonReadable = 25
    Restrict = 19; Aliased = 20; Volatile = 21

class BuiltIn(IntEnum):
    NumWorkgroups = 24; WorkgroupSize = 25; WorkgroupId = 26
    LocalInvocationId = 27; GlobalInvocationId = 28; LocalInvocationIndex = 29
    SubgroupSize = 36; SubgroupLocalInvocationId = 41; SubgroupId = 40
    SubgroupEqMask = 4416; SubgroupGeMask = 4417; SubgroupGtMask = 4418
    SubgroupLeMask = 4419; SubgroupLtMask = 4420

class ExecutionModel(IntEnum):
    GLCompute = 5

class ExecutionMode(IntEnum):
    LocalSize = 17

class StorageClass(IntEnum):
    UniformConstant = 0; Input = 1; Uniform = 2; Output = 3
    Workgroup = 4; CrossWorkgroup = 5; Private = 6; Function = 7
    PushConstant = 9; StorageBuffer = 12; PhysicalStorageBuffer = 5348

class MemoryModel(IntEnum):
    GLSL450 = 1; Vulkan = 3

class AddressingModel(IntEnum):
    Logical = 0; PhysicalStorageBuffer64 = 5348

class FunctionControl(IntEnum):
    None_ = 0

class SelectionControl(IntEnum):
    None_ = 0

class LoopControl(IntEnum):
    None_ = 0

class Scope(IntEnum):
    CrossDevice = 0; Device = 1; Workgroup = 2; Subgroup = 3; Invocation = 4

class MemorySemantics(IntEnum):
    Relaxed = 0; Acquire = 2; Release = 4; AcquireRelease = 8
    UniformMemory = 0x40; WorkgroupMemory = 0x100; ImageMemory = 0x800

@dataclass
class SPIRVModule:
    bound: int = 1
    instructions: List[int] = field(default_factory=list)

    def _id(self) -> int:
        self.bound += 1
        return self.bound - 1

    def emit(self, opcode: int, *operands: int):
        word_count = 1 + len(operands)
        self.instructions.append((word_count << 16) | opcode)
        self.instructions.extend(operands)

    def to_bytes(self) -> bytes:
        header = [SPIRV_MAGIC, SPIRV_VERSION_1_3, 0, self.bound, 0]
        words = header + list(self.instructions)
        return struct.pack(f"<{len(words)}I", *words)

class Assembler:
    """Minimal assembler: emits SPIR-V compute shaders with buffer_device_address."""

    def __init__(self, local_size: Tuple[int, int, int] = (32, 1, 1)):
        self.m = SPIRVModule()
        self.local_size = local_size
        self.constants: Dict[Tuple, int] = {}
        self._setup_base()

    def _id(self) -> int:
        return self.m._id()

    def _setup_base(self):
        m = self.m
        m.emit(Op.Capability, Capability.Shader)
        m.emit(Op.Capability, Capability.Int64)
        m.emit(Op.Capability, Capability.Float16)
        m.emit(Op.Capability, Capability.PhysicalStorageBufferAddresses)
        m.emit(Op.Capability, Capability.GroupNonUniform)
        m.emit(Op.Capability, Capability.GroupNonUniformShuffle)
        m.emit(Op.Capability, Capability.GroupNonUniformArithmetic)
        m.emit(Op.MemoryModel, AddressingModel.PhysicalStorageBuffer64, MemoryModel.Vulkan)

        self.entry_id = self._id()
        m.emit(Op.EntryPoint, ExecutionModel.GLCompute, self.entry_id, *self._encode_string("main"))
        m.emit(Op.ExecutionMode, self.entry_id, ExecutionMode.LocalSize,
               self.local_size[0], self.local_size[1], self.local_size[2])

        self.glsl_ext = self._id()
        m.emit(Op.ExtInstImport, self.glsl_ext, *self._encode_string("GLSL.std.450"))

        self.void_t = self._type_void()
        self.bool_t = self._type_bool()
        self.i8_t = self._type_int(8, 1)
        self.i16_t = self._type_int(16, 1)
        self.i32_t = self._type_int(32, 1)
        self.i64_t = self._type_int(64, 1)
        self.u32_t = self._type_int(32, 0)
        self.u64_t = self._type_int(64, 0)
        self.f16_t = self._type_float(16)
        self.f32_t = self._type_float(32)
        self.f64_t = self._type_float(64)
        self.v2f32_t = self._type_vector(self.f32_t, 2)
        self.v4f32_t = self._type_vector(self.f32_t, 4)
        self.v2u64_t = self._type_vector(self.u64_t, 2)

        self.void_fn_t = self._id()
        m.emit(Op.TypeFunction, self.void_fn_t, self.void_t)

    def _encode_string(self, s: str) -> List[int]:
        b = s.encode("utf-8") + b"\x00"
        while len(b) % 4 != 0:
            b += b"\x00"
        return [int.from_bytes(b[i:i+4], "little") for i in range(0, len(b), 4)]

    def _type_void(self) -> int:
        tid = self._id(); self.m.emit(Op.TypeVoid, tid); return tid
    def _type_bool(self) -> int:
        tid = self._id(); self.m.emit(Op.TypeBool, tid); return tid
    def _type_int(self, width: int, signedness: int) -> int:
        tid = self._id(); self.m.emit(Op.TypeInt, tid, width, signedness); return tid
    def _type_float(self, width: int) -> int:
        tid = self._id(); self.m.emit(Op.TypeFloat, tid, width); return tid
    def _type_vector(self, elem_type: int, count: int) -> int:
        tid = self._id(); self.m.emit(Op.TypeVector, tid, elem_type, count); return tid
    def _type_runtime_array(self, elem_type: int) -> int:
        tid = self._id(); self.m.emit(Op.TypeRuntimeArray, tid, elem_type); return tid
    def _type_struct(self, *member_types: int) -> int:
        tid = self._id(); self.m.emit(Op.TypeStruct, tid, *member_types); return tid
    def _type_pointer(self, storage_class: StorageClass, pointee_type: int) -> int:
        tid = self._id(); self.m.emit(Op.TypePointer, tid, storage_class, pointee_type); return tid

    def _constant(self, type_id: int, value: int) -> int:
        key = (type_id, value)
        if key in self.constants: return self.constants[key]
        cid = self._id()
        self.m.emit(Op.Constant, type_id, cid, value)
        self.constants[key] = cid
        return cid

    def _constant_u64(self, value: int) -> int:
        key = (self.u64_t, value)
        if key in self.constants: return self.constants[key]
        cid = self._id()
        low = value & 0xFFFFFFFF
        high = (value >> 32) & 0xFFFFFFFF
        self.m.emit(Op.Constant, self.u64_t, cid, low, high)
        self.constants[key] = cid
        return cid

    def define_push_constant_struct(self, fields: List[Tuple[int, int]]) -> int:
        """fields: [(type_id, offset), ...]"""
        member_types = [f[0] for f in fields]
        struct_t = self._type_struct(*member_types)
        self.m.emit(Op.Decorate, struct_t, Decoration.Block)
        for i, (ftype, offset) in enumerate(fields):
            self.m.emit(Op.MemberDecorate, struct_t, i, Decoration.Offset, offset)
        return struct_t

    def create_push_constant(self, struct_t: int) -> int:
        ptr_t = self._type_pointer(StorageClass.PushConstant, struct_t)
        var_id = self._id()
        self.m.emit(Op.Variable, ptr_t, var_id, StorageClass.PushConstant)
        return var_id

    def create_function(self, name: str = "main") -> Tuple[int, int]:
        m = self.m
        fid = self.entry_id
        m.emit(Op.Function, self.void_t, fid, FunctionControl.None_, self.void_fn_t)
        m.emit(Op.Name, fid, *self._encode_string(name))
        bid = self._id()
        m.emit(Op.Label, bid)
        return fid, bid

    def end_function(self):
        self.m.emit(Op.Return)
        self.m.emit(Op.FunctionEnd)

    def load(self, result_type: int, ptr: int, memory_access: int = 0) -> int:
        rid = self._id()
        if memory_access:
            self.m.emit(Op.Load, result_type, rid, ptr, memory_access)
        else:
            self.m.emit(Op.Load, result_type, rid, ptr)
        return rid

    def store(self, ptr: int, value: int, memory_access: int = 0):
        if memory_access:
            self.m.emit(Op.Store, ptr, value, memory_access)
        else:
            self.m.emit(Op.Store, ptr, value)

    def access_chain(self, result_type: int, base: int, *indices: int) -> int:
        rid = self._id()
        self.m.emit(Op.AccessChain, result_type, rid, base, *indices)
        return rid

    def iadd(self, a: int, b: int) -> int:
        rid = self._id(); self.m.emit(Op.IAdd, self.i32_t, rid, a, b); return rid
    def fadd(self, a: int, b: int) -> int:
        rid = self._id(); self.m.emit(Op.FAdd, self.f32_t, rid, a, b); return rid
    def fmul(self, a: int, b: int) -> int:
        rid = self._id(); self.m.emit(Op.FMul, self.f32_t, rid, a, b); return rid
    def fdiv(self, a: int, b: int) -> int:
        rid = self._id(); self.m.emit(Op.FDiv, self.f32_t, rid, a, b); return rid
    def fneg(self, a: int) -> int:
        rid = self._id(); self.m.emit(Op.FNegate, self.f32_t, rid, a); return rid

    def uconvert(self, val: int, to_bits: int) -> int:
        if to_bits == 64:
            rid = self._id(); self.m.emit(Op.UConvert, self.u64_t, rid, val); return rid
        rid = self._id(); self.m.emit(Op.UConvert, self.u32_t, rid, val); return rid

    def convert_u_to_ptr(self, val: int, ptr_type: int) -> int:
        rid = self._id(); self.m.emit(Op.ConvertUToPtr, ptr_type, rid, val); return rid

    def bitcast(self, result_type: int, val: int) -> int:
        rid = self._id(); self.m.emit(Op.Bitcast, result_type, rid, val); return rid

    def shift_left(self, val: int, shift: int) -> int:
        rid = self._id(); self.m.emit(Op.ShiftLeftLogical, self.i32_t, rid, val, shift); return rid
    def shift_right_logical(self, val: int, shift: int) -> int:
        rid = self._id(); self.m.emit(Op.ShiftRightLogical, self.i32_t, rid, val, shift); return rid

    def subgroup_shuffle_down(self, val: int, delta: int) -> int:
        rid = self._id()
        scope = self._constant(self.i32_t, Scope.Subgroup)
        delta_c = self._constant(self.i32_t, delta)
        self.m.emit(Op.GroupNonUniformShuffleDown, self.f32_t, rid, scope, val, delta_c)
        return rid

    def subgroup_broadcast_first(self, val: int) -> int:
        rid = self._id()
        scope = self._constant(self.i32_t, Scope.Subgroup)
        self.m.emit(Op.GroupNonUniformBroadcastFirst, self.f32_t, rid, scope, val)
        return rid

    def subgroup_fadd(self, val: int) -> int:
        rid = self._id()
        scope = self._constant(self.i32_t, Scope.Subgroup)
        self.m.emit(Op.GroupNonUniformFAdd, self.f32_t, rid, scope, 0, val)
        return rid

    def barrier(self, execution_scope: int = Scope.Workgroup, memory_scope: int = Scope.Workgroup,
                semantics: int = MemorySemantics.AcquireRelease | MemorySemantics.WorkgroupMemory):
        exec_scope = self._constant(self.i32_t, execution_scope)
        mem_scope = self._constant(self.i32_t, memory_scope)
        sem = self._constant(self.i32_t, semantics)
        self.m.emit(Op.ControlBarrier, exec_scope, mem_scope, sem)

    def memory_barrier(self, scope: int = Scope.Workgroup, semantics: int = MemorySemantics.AcquireRelease | MemorySemantics.WorkgroupMemory):
        mem_scope = self._constant(self.i32_t, scope)
        sem = self._constant(self.i32_t, semantics)
        self.m.emit(Op.MemoryBarrier, mem_scope, sem)

    def save(self, path: str):
        with open(path, "wb") as f:
            f.write(self.m.to_bytes())
        print(f"SPIR-V saved: {path} ({len(self.m.to_bytes())} bytes, bound={self.m.bound})")


def make_test_shader(output_path: str):
    """Generate a test compute shader: scale buffer via buffer reference."""
    a = Assembler(local_size=(32, 1, 1))
    m = a.m

    # Push constants: { u64 addrIn, u64 addrOut, u32 count, f32 scale }
    pc_struct = a.define_push_constant_struct([
        (a.u64_t, 0), (a.u64_t, 8), (a.u32_t, 16), (a.f32_t, 20)
    ])
    pc_var = a.create_push_constant(pc_struct)

    # Buffer reference type: struct { float data[]; }
    buf_arr_t = a._type_runtime_array(a.f32_t)
    buf_struct_t = a._type_struct(buf_arr_t)
    m.emit(Op.Decorate, buf_struct_t, Decoration.Block)
    m.emit(Op.MemberDecorate, buf_struct_t, 0, Decoration.Offset, 0)
    buf_ptr_t = a._type_pointer(StorageClass.PhysicalStorageBuffer, buf_struct_t)
    m.emit(Op.Decorate, buf_ptr_t, Decoration.Aliased)

    # Built-in: gl_GlobalInvocationID
    v3u32_t = a._type_vector(a.u32_t, 3)
    in_struct_t = a._type_struct(v3u32_t)
    m.emit(Op.Decorate, in_struct_t, Decoration.Block)
    m.emit(Op.MemberDecorate, in_struct_t, 0, Decoration.Offset, 0)
    in_ptr_t = a._type_pointer(StorageClass.Input, in_struct_t)
    in_var = a._id()
    m.emit(Op.Variable, in_ptr_t, in_var, StorageClass.Input)
    m.emit(Op.Decorate, in_var, Decoration.BuiltIn, BuiltIn.GlobalInvocationId)
    m.emit(Op.Name, in_var, *a._encode_string("gl_GlobalInvocationID"))
    m.emit(Op.MemberName, in_struct_t, 0, *a._encode_string("gl_GlobalInvocationID"))

    fid, bid = a.create_function("main")

    # idx = gl_GlobalInvocationID.x
    idx_ptr = a.access_chain(a._type_pointer(StorageClass.Input, a.u32_t), in_var,
                             a._constant(a.i32_t, 0), a._constant(a.i32_t, 0))
    idx = a.load(a.u32_t, idx_ptr)

    # Load push constants
    addr_in_ptr = a.access_chain(a._type_pointer(StorageClass.PushConstant, a.u64_t),
                                  pc_var, a._constant(a.i32_t, 0))
    addr_in = a.load(a.u64_t, addr_in_ptr)
    addr_out_ptr = a.access_chain(a._type_pointer(StorageClass.PushConstant, a.u64_t),
                                   pc_var, a._constant(a.i32_t, 1))
    addr_out = a.load(a.u64_t, addr_out_ptr)
    scale_ptr = a.access_chain(a._type_pointer(StorageClass.PushConstant, a.f32_t),
                               pc_var, a._constant(a.i32_t, 3))
    scale = a.load(a.f32_t, scale_ptr)

    # Convert to buffer pointers
    buf_in_ptr = a.convert_u_to_ptr(addr_in, buf_ptr_t)
    buf_out_ptr = a.convert_u_to_ptr(addr_out, buf_ptr_t)

    # Element access
    elem_ptr_in = a.access_chain(a._type_pointer(StorageClass.PhysicalStorageBuffer, a.f32_t),
                                  buf_in_ptr, a._constant(a.i32_t, 0), idx)
    elem_ptr_out = a.access_chain(a._type_pointer(StorageClass.PhysicalStorageBuffer, a.f32_t),
                                   buf_out_ptr, a._constant(a.i32_t, 0), idx)

    # Load, scale, store
    val = a.load(a.f32_t, elem_ptr_in, memory_access=0x2)
    scaled = a.fmul(val, scale)
    a.store(elem_ptr_out, scaled, memory_access=0x2)

    a.end_function()
    a.save(output_path)
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RDNA4 SPIR-V Assembler")
    parser.add_argument("-o", "--output", default="test.spv", help="Output SPIR-V file")
    parser.add_argument("--test", action="store_true", help="Generate test shader")
    args = parser.parse_args()
    if args.test:
        make_test_shader(args.output)
    else:
        print("Usage: python rdna4_as.py --test -o kernel.spv")
