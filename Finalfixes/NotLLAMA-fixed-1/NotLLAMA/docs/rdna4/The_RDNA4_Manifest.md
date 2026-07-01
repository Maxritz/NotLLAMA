# The RDNA4 Manifesto: A Llama Architecture That Doesn't Give a Shit About CUDA

## The Core Heresy

**NVIDIA built a self-driving car. We keep trying to build a self-driving car with bicycle parts.**

RDNA4 is not a broken NVIDIA GPU. It is a **different species**. It is a graphics predator that happens to be excellent at math. Every time we say "CUDA does it this way," we put a muzzle on that predator and teach it to fetch.

Here is what llama.cpp becomes when we stop porting and start architecting.

---

## Philosophy 0: The Compiler Is the Operating System

**CUDA assumption:** The hardware is smart. The programmer writes high-level kernels. The chip schedules, caches, and hides latency automatically.

**RDNA4 truth:** The hardware is an idiot savant. It executes exactly what you tell it, exactly when you tell it, at terrifying speed. The **compiler is the OS, the scheduler, the memory manager, and the prefetcher.**

This means:
- `S_WAITCNT` is our scheduler, not a hardware warp switcher.
- `S_CLAUSE` is our critical section lock, not a dynamic priority boost.
- `S_DELAY_ALU` is our pipeline interlock, not a hardware scoreboard.
- `S_ALLOC_VGPR` is our memory oversubscription, not a context switch.

**The architecture shift:** llama.cpp stops being a collection of kernels that launch and die. It becomes a **single persistent program** that runs on the GPU forever, managing its own execution state in device memory, using the host only to drop tokens into a mailbox.

---

## Philosophy 1: One Wave Is One Unit (32 Threads, Not 128)

**CUDA assumption:** A block is 128 or 256 threads. Shared memory is sized for this. Barriers assume this. Tensor Cores consume this.

**RDNA4 truth:** A SIMD is 32 lanes. A wave is 32 threads. This is not "half a warp." This is the **native organism** of the chip. A 128-thread block on RDNA4 is four waves pretending to be one creature, communicating through LDS (the conveyor belt) when they could just be four separate creatures doing four separate jobs.

**The architecture shift:**
- **One tile = one wave.** A 16×16 WMMA tile fits perfectly in one wave's register file. No inter-wave coordination for GEMM.
- **No LDS for GEMM.** The four waves in a CUDA block use LDS to share data. On RDNA4, each wave is independent. LDS is not a cache; it is a scratchpad for cross-wave reduction only.
- **Occupancy through count, not size.** We launch 8, 12, 16 waves per CU, each 32 threads. Not 2 waves of 128 threads. The scheduler (our `S_WAITCNT` chains) hides latency by having many small workers, not few big ones.

---

## Philosophy 2: The Memory Pipeline Is the Dequantizer

**CUDA assumption:** Load bytes from global memory. Write to shared memory. Swizzle. Convert. Then feed Tensor Core.

**RDNA4 truth:** The memory pipeline (VBUFFER/VIMAGE) has a **format converter built into the load path.** It can unpack 4-bit integers, convert SNORM to FP32, apply channel masks, and swizzle AOS→SOA **before the data reaches a VGPR.**

**The architecture shift:**
- **Weights are not arrays.** They are **buffer resources** (V# descriptors). The descriptor encodes the quantization format (Q4_K, Q6_K, Q8_0).
- **There is no dequantization kernel.** The `BUFFER_LOAD_FORMAT_*` instruction performs the unpack and conversion during the load. The ALU receives FP16/FP32 directly.
- **No shared memory staging.** NVIDIA needs LDS to stage unpacked weights because the load path is dumb. RDNA4's load path is smart. We delete the staging step entirely.

---

## Philosophy 3: Registers Talk to Registers (DPP), Not Through Memory

**CUDA assumption:** To find the max value across a warp, use `__shfl_sync` or write to shared memory, reduce, read back.

**RDNA4 truth:** `DPP_ROW_SL`, `DPP_ROW_BCAST`, and `DS_BPERMUTE` allow **cross-lane data movement within the register file.** No LDS. No memory traffic. No barrier.

**The architecture shift:**
- **Softmax is a register shuffle.** Max reduction: 5 `DPP` shifts + compares. Sum reduction: broadcast + multiply-add. All in VGPRs.
- **Attention scores never touch LDS.** The entire attention block (Q×K, softmax, ×V) can be a single wave that shuffles data internally.
- **LDS is only for cross-wave reduction.** LayerNorm across a workgroup uses LDS. Everything inside the wave uses shoulder-taps.

---

## Philosophy 4: The Kernel Never Dies (Persistent Execution)

**CUDA assumption:** One kernel per layer. Launch, execute, retire. CUDA Graphs amortizes launch cost by recording a sequence.

**RDNA4 truth:** Graphics shaders are persistent. They loop forever. The driver doesn't kill them between draw calls. Compute kernels can do the same.

**The architecture shift:**
- **One kernel, 80 layers, infinite tokens.** The kernel loops: `while (true) { read mailbox; for (layer = 0; layer < 80; layer++) { ... } write output; }`
- **Zero launch overhead.** The host writes a token ID to device memory. The kernel polls and processes it. For token generation, where each token is microseconds, eliminating 80 kernel launches per token is a massive win.
- **Pipelined prefetching.** `S_WAITCNT` chains overlap Layer N's math with Layer N+1's weight fetch. The memory manager (our compiler) knows the latency and fills the gaps.

---

## Philosophy 5: Four Kitchens, Four Meals (Multi-ACE)

**CUDA assumption:** One compute stream. Async copy engine moves data while compute runs. But only one kernel executes at a time per stream.

**RDNA4 truth:** Multiple ACEs (Asynchronous Compute Engines) execute **completely independent command queues** on different CUs simultaneously.

**The architecture shift:**
- **Speculative decoding is true parallelism, not time-slicing.**
  - ACE 0: Main model (target).
  - ACE 1: Draft model (small, running simultaneously).
  - ACE 2: N-gram cache lookup and draft token selection.
  - ACE 3: KV-cache management and verification.
- **No CPU involvement.** The CPU drops a token into the mailbox. The four ACEs negotiate through device-side atomics and fine-grain memory. The draft model runs on ACE 1 while the main model is still processing the previous token on ACE 0.
- **This is a graphics architecture feature.** Games do physics on ACE 1 while rendering on ACE 0. We do inference the same way.

---

## Philosophy 6: Shrinking and Growing (Dynamic VGPR)

**CUDA assumption:** Register allocation is static. A kernel requests 64 registers per thread. Those registers are reserved for the entire execution, even if the thread is waiting for memory.

**RDNA4 truth:** `S_ALLOC_VGPR` allows a wave to **return registers to the pool** when memory-bound, and **reclaim them** when compute-bound.

**The architecture shift:**
- **Memory phases shrink.** When fetching weights, the wave gives away 48 of its 64 VGPRs. Another wave launches and hides latency.
- **Compute phases grow.** When the WMMA chain starts, the wave expands back to full register count.
- **The CU is always full.** NVIDIA's static allocation leaves empty register slots during memory waits. RDNA4's dynamic packing keeps the CU at maximum occupancy.

---

## Philosophy 7: Two Hands, One Beat (VOPD)

**CUDA assumption:** One instruction, one operation. Simple. Clean.

**RDNA4 truth:** `VOPD` packs two independent VALU operations into one 64-bit instruction, issued in one cycle. But the compiler must ensure the sources come from different VGPR banks and the destinations are even/odd paired.

**The architecture shift:**
- **The register allocator is bank-aware.** We don't just allocate registers; we allocate them into alternating banks to satisfy dual-issue constraints.
- **Dequantization + FMA in one cycle.** While the memory pipeline feeds FP16 values, the ALU dual-issues "load next address" with "multiply current value."
- **The inner loop is VLIW.** Very Long Instruction Word. One fetch, two operations. Double the throughput without double the instruction cache pressure.

---

## Philosophy 8: Free Seasoning (OMOD/CLAMP)

**CUDA assumption:** To scale by 0.5 or clamp to [0,1], you issue separate `MUL` and `MAX` instructions.

**RDNA4 truth:** VOP3 encoding has **free modifier bits:** `OMOD` (×2, ×4, ×0.5) and `CLAMP` ([0,1] or INT_MIN/MAX). Applied in the output stage at zero cost.

**The architecture shift:**
- **GELU/ReLU/Sigmoid are free.** Activation functions that require scaling and clamping consume no extra instructions. They are bitfields in the FMA opcode.
- **Quantization scales are folded into the math.** The per-block scale factor is applied via `OMOD` during the final accumulation store, not as a separate multiplication kernel.

---

## Philosophy 9: The Mailroom Knows Everything (VBUFFER/VIMAGE)

**CUDA assumption:** Memory is dumb bytes. The ALU interprets them.

**RDNA4 truth:** The memory pipeline is a **co-processor** with its own format conversion, bounds checking, and compression logic.

**The architecture shift:**
- **DCC (Delta Color Compression) for KV cache.** Allocate KV cache as image objects, not buffers. The memory controller compresses low-entropy regions automatically. Free bandwidth.
- **PRT (Partially Resident Textures) for sparse models.** If a weight page is missing, the hardware returns zero instead of crashing. Enables sparse MoE routing without software page tables.
- **A16/D16 addressing.** 16-bit addresses and 16-bit data returns save VGPRs. For small tensors (attention heads), use 16-bit indices to reduce register pressure.

---

## What llama.cpp Becomes

| Old llama.cpp (CUDA-ism) | New llama.cpp (RDNA4-native) |
|---|---|
| 80 kernel launches per token | 1 persistent kernel, host drops tokens in mailbox |
| 128-thread blocks with LDS swizzling | 32-thread waves, register-to-register data movement |
| Separate dequantization kernels | VBUFFER descriptors with built-in format conversion |
| Shared memory for softmax reduction | DPP shoulder-taps in register file |
| Single compute stream | 4 ACEs running draft, target, cache, and KV in parallel |
| Static register allocation | Dynamic VGPR shrink/grow per phase |
| One instruction, one operation | VOPD dual-issue with bank-aware allocation |
| Plain buffer loads for weights | Image objects with DCC compression |
| Hardware warp scheduler | Compiler-managed `S_WAITCNT`/`S_CLAUSE`/`S_DELAY_ALU` chains |

---

## The Final Truth

**We don't need to beat NVIDIA at NVIDIA's game.** We need to stop playing their game.

RDNA4 is not a worse Tensor Core. It is a **better graphics processor that happens to do matrix math.** Its strengths are tiny teams, explicit scheduling, smart memory, and true parallelism. Its weakness is pretending to be a CDNA chip.

The alternative architecture is not "llama.cpp with AMD optimizations." It is **"llama.cpp reimagined as a game engine shader graph."** Persistent kernels, async compute queues, image-based memory, and register-level communication.

Vulkan is winning because it treats RDNA4 like a graphics chip. ROCm loses because it treats RDNA4 like a broken MI300.

**Fuck CUDA. Build for the kitchen, not the factory.**