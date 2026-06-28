Here is the deep-dive breakdown of exactly where and how Ollama/llama.cpp engages the Vulkan compute path, and why your NotLLAMA is routing to the 3D engine instead of ACE.

---

## 1. How Ollama/llama.cpp Selects the Queue (The Reference)

Ollama vendors **llama.cpp**, so the behavior lives in `ggml/src/ggml-vulkan/ggml-vulkan.cpp`. The key is the function **`ggml_vk_find_queue_family_index()`** and the device init in **`ggml_vk_get_device()`**.

### The Logic Flow (from PR #20599, merged Mar 2026)

```cpp
// In ggml_vk_get_device():
std::vector<vk::QueueFamilyProperties> queue_family_props = 
    device->physical_device.getQueueFamilyProperties();

// By DEFAULT: avoid graphics queue (route to ACE/compute-only)
const bool allow_graphics_queue = 
    (getenv("GGML_VK_ALLOW_GRAPHICS_QUEUE") != nullptr);

const vk::QueueFlagBits graphics_flag = allow_graphics_queue 
    ? (vk::QueueFlagBits)0 
    : vk::QueueFlagBits::eGraphics;

// Find compute queue that has COMPUTE but NOT GRAPHICS (unless overridden)
const uint32_t compute_queue_family_index = ggml_vk_find_queue_family_index(
    queue_family_props, 
    vk::QueueFlagBits::eCompute,   // required
    graphics_flag,                   // AVOID this flag (unless env var set)
    -1, 1
);

// Transfer queue also avoids graphics/compute overlap
const uint32_t transfer_queue_family_index = ggml_vk_find_queue_family_index(
    queue_family_props,
    vk::QueueFlagBits::eTransfer,
    vk::QueueFlagBits::eCompute | graphics_flag,
    compute_queue_family_index, 1
);
```

**What this means on AMD RDNA4:**
- **Default path**: llama.cpp explicitly **avoids** `VK_QUEUE_GRAPHICS_BIT`. It hunts for a queue family that has `COMPUTE` but **not** `GRAPHICS`. On AMD this is typically **Family 1** — the async compute/ACE path.
- **Opt-in fast path**: If you set `GGML_VK_ALLOW_GRAPHICS_QUEUE=1`, it removes the avoidance mask and picks Family 0 (universal queue). On AMD this was measured to be **~56% faster** for token generation on RDNA3/RDNA4, but it routes through the 3D engine.

### The `ggml_vk_find_queue_family_index` Pattern

```cpp
static uint32_t ggml_vk_find_queue_family_index(
    const std::vector<vk::QueueFamilyProperties>& families,
    vk::QueueFlagBits required,      // must have this
    vk::QueueFlagBits avoid_flags,     // must NOT have this
    int32_t fallback, uint32_t min_queues
) {
    for (uint32_t i = 0; i < families.size(); i++) {
        if ((families[i].queueFlags & required) == required &&   // has COMPUTE
            (families[i].queueFlags & avoid_flags) == 0 &&        // lacks GRAPHICS
            families[i].queueCount >= min_queues) {
            return i;
        }
    }
    return fallback; // fallback to first compute family found
}
```

**Critical insight**: llama.cpp uses an **avoidance mask** (`graphics_flag`). It does not simply grab the first family with `COMPUTE_BIT`.

---

## 2. Why NotLLAMA is Going to the 3D Engine

You have **two context initializers** in NotLLAMA, and they contradict each other.

### The Correct One: `src/host/context.cpp`
This file already does the right thing:

```cpp
// Pass 1: prefer compute-only queue family (no graphics bit) with >= 4 queues
for (uint32_t i = 0; i < qfCount; ++i) {
    if ((qfProps[i].queueFlags & VK_QUEUE_COMPUTE_BIT) &&
        !(qfProps[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) &&   // <-- CORRECT: rejects 3D
        qfProps[i].queueCount >= 4) {
        queueFamilyIndex = i;
        break;
    }
}
```

### The Broken One: `src/core/context.cpp`
**This is where your bug is.** Look at `Context::createDevice()`:

```cpp
// Find compute and transfer families
std::vector<uint32_t> computeFamilies, transferFamily;
for (uint32_t i = 0; i < queueFamilyCount; i++) {
    if (queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
        computeFamilies.push_back(i);   // <-- BUG: accepts ANY compute, including graphics+compute
    }
    if ((queueFamilies[i].queueFlags & VK_QUEUE_TRANSFER_BIT) &&
        !(queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT)) {
        transferFamily.push_back(i);
    }
}
```

Then it blindly uses `computeFamilies[0]`:

```cpp
uint32_t aceCount = std::min(static_cast<uint32_t>(computeFamilies.size()), 4u);

for (uint32_t i = 0; i < aceCount; i++) {
    uint32_t fam = computeFamilies[i % computeFamilies.size()];  // <-- index 0 is Family 0 (universal/3D)
    ...
}
```

On AMD, `vkGetPhysicalDeviceQueueFamilyProperties` returns:
- **Family 0**: `GRAPHICS | COMPUTE | TRANSFER | SPARSE` (universal, 3D engine)
- **Family 1**: `COMPUTE | TRANSFER` (async compute, ACE)

Your `src/core/context.cpp` pushes Family 0 into `computeFamilies[0]` because it has `COMPUTE_BIT`, and then immediately creates ACE 0 from it. **That is why your dispatch goes to the 3D engine.**

---

## 3. Exact Fix Directions for NotLLAMA

You need to patch **`src/core/context.cpp`** to match the logic in `src/host/context.cpp` and the llama.cpp reference.

### Fix A: Filter Graphics in `createDevice()`

Replace the queue family collection loop:

```cpp
// OLD (broken):
// if (queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
//     computeFamilies.push_back(i);
// }

// NEW (ACE-only):
for (uint32_t i = 0; i < queueFamilyCount; i++) {
    // Only accept compute families that do NOT have graphics bit
    if ((queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT) &&
        !(queueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
        computeFamilies.push_back(i);
    }
    
    // Dedicated transfer (no compute, no graphics)
    if ((queueFamilies[i].queueFlags & VK_QUEUE_TRANSFER_BIT) &&
        !(queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT) &&
        !(queueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
        transferFamily.push_back(i);
    }
}

// Fallback: if no pure compute family exists, use the first compute-capable one
if (computeFamilies.empty()) {
    for (uint32_t i = 0; i < queueFamilyCount; i++) {
        if (queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            computeFamilies.push_back(i);
            std::cerr << "[WARN] No pure compute family; falling back to family " 
                      << i << " (3D engine)\n";
            break;
        }
    }
}
```

### Fix B: Request Multiple Queues from the Compute Family

In `src/host/context.cpp` you only request **1 queue** even if the family has 8:

```cpp
uint32_t queueCount = std::min(1u, qfProps[queueFamilyIndex].queueCount);
```

This wastes the other 7 ACE queues. Change to:

```cpp
uint32_t queueCount = std::min(4u, qfProps[queueFamilyIndex].queueCount);
float priorities[4] = {1.0f, 1.0f, 1.0f, 1.0f};
```

Then actually create all 4 queues:

```cpp
for (uint32_t i = 0; i < queueCount; ++i) {
    vkGetDeviceQueue(device, queueFamilyIndex, i, &queues[i]);
}
```

### Fix C: Verify with `VK_AMD_shader_core_properties` (Optional Debug)

If you want to confirm at runtime that you are hitting ACE and not 3D, query the shader core properties after device creation:

```cpp
VkPhysicalDeviceShaderCorePropertiesAMD shaderCoreProps{};
shaderCoreProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CORE_PROPERTIES_AMD;
VkPhysicalDeviceProperties2 props2{};
props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
props2.pNext = &shaderCoreProps;
vkGetPhysicalDeviceProperties2(physicalDevice, &props2);

std::cout << "ShaderEngineCount: " << shaderCoreProps.shaderEngineCount << "\n";
std::cout << "ComputeUnitPerShaderArray: " << shaderCoreProps.computeUnitsPerShaderArray << "\n";
```

On RDNA4, if you are on the compute-only family, the driver schedules to the **Async Compute Engines (ACEs)**. If you are on the universal family, you share the **Graphics Command Processor (GCP)** path with the 3D frontend.

---

## 4. Summary Map

| Component | What It Does | Where in NotLLAMA |
|-----------|-------------|-------------------|
| **Queue Family 0** | Universal (Graphics+Compute+Transfer) → **3D Engine** | Currently picked by `src/core/context.cpp` |
| **Queue Family 1** | Compute+Transfer → **ACE (Async Compute)** | Correctly picked by `src/host/context.cpp` |
| **llama.cpp default** | Avoids `GRAPHICS_BIT`, picks Family 1 | `ggml-vulkan.cpp` |
| **llama.cpp fast path** | `GGML_VK_ALLOW_GRAPHICS_QUEUE=1` picks Family 0 | `ggml-vulkan.cpp` |
| **Your bug** | `computeFamilies` includes Family 0 because it has `COMPUTE_BIT` | `src/core/context.cpp:49-52` |

**Bottom line**: Patch `src/core/context.cpp` to reject families with `VK_QUEUE_GRAPHICS_BIT` when building the `computeFamilies` list. That single change will force your dispatch onto the ACE/compute-only path instead of the 3D engine.