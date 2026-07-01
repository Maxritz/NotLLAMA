# VALIDATED FLOW — Inference Engine Decision Tree
# Based on actual code verification 2026-06-29

## FLOW 1: What Happens When You Run Inference

```
START: main.cpp → engine.generate(prompt, 32)
│
├── tokenizer.encode(prompt) → tokens[]
│
└── FOR EACH token:
    │
    └── engine.forward(tokenId, seqPos)
        │
        └── forwardPartial(tokenId, seqPos, blockCount)
            │
            ├── allocator->reset()                          ✅ EXISTS (allocator.cpp:60)
            │
            ├── Alloc ring buffers:                         ✅ EXISTS (inference_engine.cpp:58-65)
            │   hidden, Q, K, V, attnOut, mlpOut, logits, sampleOut
            │
            ├── EMBED: findTensorAddr("token_embd.weight")  ✅ EXISTS (inference_engine.cpp:83)
            │   └── dispatchInBatch(embed pipeline)         ✅ WORKS (if weight is F32)
            │
            └── FOR EACH LAYER (0..blockCount):
                │
                ├── RMS_NORM: hidden + attn_norm.weight     ✅ EXISTS (line 100-103)
                │
                ├── QKV GEMM:                               ❌ BUG
                │   findTensorAddr("blk.X.attn_q.weight")
                │   └── gemm.comp reads B.data[k * N + col] as float
                │       ├── If weight IS F32 → CORRECT      ✅ stories260K
                │       └── If weight is Q4_0 → GARBAGE     ❌ ALL OTHERS
                │
                ├── RoPE                                    ✅ EXISTS (line 137-140)
                ├── KV_CACHE_WRITE                          ✅ EXISTS (line 143-153)
                ├── ATTENTION (per-head loop)               ✅ EXISTS (line 156-161)
                ├── ATTN OUTPUT GEMM                        ❌ SAME BUG
                ├── RESIDUAL ADD                            ✅ EXISTS (line 180-182)
                ├── FFN NORM                                ✅ EXISTS (line 190-193)
                ├── GATE GEMM                               ❌ SAME BUG
                ├── UP GEMM                                 ❌ SAME BUG
                ├── SILU+MUL                                ✅ EXISTS (line 221-223)
                ├── DOWN GEMM                               ❌ SAME BUG
                └── RESIDUAL ADD                            ✅ EXISTS (line 242-244)
```

## FLOW 2: GEMM Weight Source (THE BUG)

```
GEMM called with: GemmPushConstants { addrA, addrB=weight_addr, addrC, M, N, K, alpha, transB }
│
├── gemm.comp reads: float b = B.data[col * K + k]    (line 46 of gemm.comp)
│
├── addrB points to GPU buffer containing...
│
├── F32 bytes?  → reinterpret_cast<float> → correct value    ✅ WORKS
├── F16 bytes?  → reinterpret_cast<float> → GARBAGE          ❌ FAILS
├── Q4_0 bytes? → reinterpret_cast<float> → GARBAGE          ❌ FAILS
├── Q6_K bytes? → reinterpret_cast<float> → GARBAGE          ❌ FAILS
└── Q8_0 bytes? → reinterpret_cast<float> → GARBAGE          ❌ FAILS
```

## FLOW 3: What Needs to Happen (FIX)

```
FOR EACH weight tensor in forwardPartial():
│
├── step 1: Get tensor metadata
│   tensor = findTensor(model, "blk.X.attn_q.weight")
│   │
│   └── tensor->format tells us the quantization format
│
├── step 2: Decision branch
│   │
│   ├── tensor->format == QuantFormat::F32?
│   │   └── YES → Use tensor->gpuAddress directly for GEMM
│   │             No dequant needed
│   │
│   └── tensor->format != QuantFormat::F32?
│       └── YES → Need GPU dequant
│           │
│           ├── step 2a: Calculate output size
│           │   nElements = tensor->sizeBytes / bytesPerElement(format)
│           │
│           ├── step 2b: Allocate staging buffer
│           │   stagingAddr = allocator->alloc(nElements * sizeof(float))
│           │
│           ├── step 2c: Dispatch dequant shader
│           │   DequantizePushConstants = {
│           │       .addrQuant = tensor->gpuAddress,
│           │       .addrOut = stagingAddr,
│           │       .nElements = nElements,
│           │       .quantFormat = format,
│           │       .totalThreads = nElements,
│           │       .elementOffset = 0
│           │   }
│           │   scheduler->dispatchInBatch(dequantPipeline, ...)
│           │
│           ├── step 2d: Pipeline barrier
│           │   scheduler->barrierBetweenGroups(CS, CS, WRITE, READ)
│           │
│           └── step 2e: Use stagingAddr for GEMM
│               GemmPushConstants = { ..., stagingAddr, ... }
```

## FLOW 4: Pipeline Loading (CURRENT vs NEEDED)

```
CURRENT main.cpp loads:                    NEEDED:
─────────────────────                      ──────
gemm         ✅                            gemm         ✅
attention    ✅                            attention    ✅
rope         ✅                            rope         ✅
add          ✅                            add          ✅
silu_mul     ✅                            silu_mul     ✅
rms_norm     ✅                            rms_norm     ✅
embed        ✅                            embed        ✅
kv_cache_write ✅                          kv_cache_write ✅
                                            dequantize   ❌ MISSING
                                            topk         (optional)
                                            flash_attention (optional)
```

## FLOW 5: Shader Compilation (CURRENT vs NEEDED)

```
CMakeLists.txt KERNELS list:
─────────────────────────────
CURRENT:                                  NEEDED:
gemm.comp           ✅                   gemm.comp           ✅
attention.comp      ✅                   attention.comp      ✅
rope.comp           ✅                   rope.comp           ✅
add.comp            ✅                   add.comp            ✅
silu_mul.comp       ✅                   silu_mul.comp       ✅
rms_norm.comp       ✅                   rms_norm.comp       ✅
embed.comp          ✅                   embed.comp          ✅
kv_cache_write.comp ✅                   kv_cache_write.comp ✅
                                          dequantize.comp     ❌ MISSING
```

## FLOW 6: What Actually Exists in src/kernels/

```
21 .comp files:                           8 compiled:
──────────────                            ──────────
add.comp              ✅ compiled         add.comp ✅
attention.comp        ✅ compiled         attention.comp ✅
bda_test.comp         ❌ NOT compiled
compress_context.comp ❌ NOT compiled
cooperative_gemm.comp ❌ NOT compiled
dequantize_test.comp  ❌ NOT compiled     (writes 1234.0, debug only)
embed.comp            ✅ compiled         embed.comp ✅
flash_attention.comp  ❌ NOT compiled
gemm.comp             ✅ compiled         gemm.comp ✅
gemm_coopmat.comp     ❌ NOT compiled
kernel_entry.comp     ❌ NOT compiled
kv_cache_dequant.comp ❌ NOT compiled
kv_cache_quantize.comp❌ NOT compiled
kv_cache_write.comp   ✅ compiled         kv_cache_write.comp ✅
mlp_fused_gateup.comp ❌ NOT compiled
rms_norm.comp         ✅ compiled         rms_norm.comp ✅
rope.comp             ✅ compiled         rope.comp ✅
silu_mul.comp         ✅ compiled         silu_mul.comp ✅
topk.comp             ❌ NOT compiled
test_shader.spv       ❌ NOT compiled     (binary, not .comp)
```

## FLOW 7: Decision Matrix — Does My Model Work?

```
Model Format │ Weight on GPU │ GEMM Expects │ Match? │ Result
─────────────┼───────────────┼──────────────┼────────┼───────
F32          │ F32 bytes     │ float        │ ✅ YES │ WORKS
F16          │ F16 bytes     │ float        │ ❌ NO  │ GARBAGE
Q4_0         │ Q4_0 bytes    │ float        │ ❌ NO  │ GARBAGE
Q4_1         │ Q4_1 bytes    │ float        │ ❌ NO  │ GARBAGE
Q5_0         │ Q5_0 bytes    │ float        │ ❌ NO  │ GARBAGE
Q5_1         │ Q5_1 bytes    │ float        │ ❌ NO  │ GARBAGE
Q8_0         │ Q8_0 bytes    │ float        │ ❌ NO  │ GARBAGE
Q8_1         │ Q8_1 bytes    │ float        │ ❌ NO  │ GARBAGE
Q2_K         │ Q2_K bytes    │ float        │ ❌ NO  │ GARBAGE
Q3_K         │ Q3_K bytes    │ float        │ ❌ NO  │ GARBAGE
Q4_K         │ Q4_K bytes    │ float        │ ❌ NO  │ GARBAGE
Q5_K         │ Q5_K bytes    │ float        │ ❌ NO  │ GARBAGE
Q6_K         │ Q6_K bytes    │ float        │ ❌ NO  │ GARBAGE
Q8_K         │ Q8_K bytes    │ float        │ ❌ NO  │ GARBAGE
```

## FLOW 8: Minimum Viable Fix (4 Files)

```
FILE 1: src/kernels/dequantize.comp     (CREATE)
├── Real dequant shader for Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, F16
├── Push constants match DequantizePushConstants struct
└── One thread per element, stride loop

FILE 2: CMakeLists.txt                  (EDIT line ~50)
├── Add: src/kernels/dequantize.comp
└── to: set(KERNELS ... )

FILE 3: main.cpp                        (EDIT line ~174)
├── Add: loadPipe("dequantize", sizeof(DequantizePushConstants));
└── After: loadPipe("kv_cache_write", ...)

FILE 4: src/host/inference_engine.cpp   (EDIT forwardPartial)
├── Before each GEMM call:
│   tensor = findTensor(model, prefix + ".attn_q.weight")
│   if (tensor && tensor->format != QuantFormat::F32) {
│       // allocate staging, dispatch dequant, use staging for GEMM
│   }
└── Repeat for: attn_q, attn_k, attn_v, attn_output, ffn_gate, ffn_up, ffn_down
```
