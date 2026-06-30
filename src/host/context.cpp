#include "rdna4_vulkan.hpp"
#include <iostream>
#include <vector>
#include <cstring>
#include <string>

namespace rdna4 {

static VKAPI_ATTR VkBool32 VKAPI_CALL DebugUtilsCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT /*type*/,
    const VkDebugUtilsMessengerCallbackDataEXT* cb,
    void* /*user*/) {
    if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
        std::cerr << "[VK Validation] " << cb->pMessage << "\n";
    }
    return VK_FALSE;
}

static bool CheckValidationLayerSupport() {
    uint32_t count = 0;
    vkEnumerateInstanceLayerProperties(&count, nullptr);
    std::vector<VkLayerProperties> layers(count);
    vkEnumerateInstanceLayerProperties(&count, layers.data());
    for (auto& l : layers) {
        if (std::strcmp(l.layerName, "VK_LAYER_KHRONOS_validation") == 0) return true;
    }
    return false;
}

static bool CheckInstanceExtension(const char* name) {
    uint32_t count = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &count, nullptr);
    std::vector<VkExtensionProperties> exts(count);
    vkEnumerateInstanceExtensionProperties(nullptr, &count, exts.data());
    for (auto& e : exts) {
        if (std::strcmp(e.extensionName, name) == 0) return true;
    }
    return false;
}

bool VulkanContext::init() {
    VkApplicationInfo appInfo = {};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "RDNA4 LLaMA";
    appInfo.apiVersion = VK_API_VERSION_1_4;

    VkInstanceCreateInfo instInfo = {};
    instInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instInfo.pApplicationInfo = &appInfo;

    // Validation layers + debug utils messenger (debug builds only)
    std::vector<const char*> layers;
    std::vector<const char*> extensions;
    VkDebugUtilsMessengerCreateInfoEXT dbgCreateInfo{};
    bool enableDebugUtils = false;
#ifndef NDEBUG
    bool haveValidation = CheckValidationLayerSupport();
    bool haveDebugUtils = CheckInstanceExtension(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    if (haveValidation) {
        layers.push_back("VK_LAYER_KHRONOS_validation");
    } else {
        std::cerr << "[VK] Validation layer not available\n";
    }
    if (haveDebugUtils) {
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        enableDebugUtils = true;
    } else {
        std::cerr << "[VK] VK_EXT_debug_utils not available\n";
    }

    if (enableDebugUtils) {
        dbgCreateInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        dbgCreateInfo.messageSeverity =
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        dbgCreateInfo.messageType =
            VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        dbgCreateInfo.pfnUserCallback = DebugUtilsCallback;
        instInfo.pNext = &dbgCreateInfo;
    }
#else
    // Functions only used in debug builds; mark unused in release.
    (void)&CheckValidationLayerSupport;
    (void)&CheckInstanceExtension;
#endif

    instInfo.enabledLayerCount = static_cast<uint32_t>(layers.size());
    instInfo.ppEnabledLayerNames = layers.empty() ? nullptr : layers.data();
    instInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    instInfo.ppEnabledExtensionNames = extensions.empty() ? nullptr : extensions.data();

    VkResult r = vkCreateInstance(&instInfo, nullptr, &instance);
    if (r != VK_SUCCESS) {
        std::cerr << "vkCreateInstance failed: " << r << "\n";
        return false;
    }

    // Create debug messenger after instance is created
#ifndef NDEBUG
    if (enableDebugUtils) {
        auto createFn = (PFN_vkCreateDebugUtilsMessengerEXT)
            vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
        if (createFn) {
            createFn(instance, &dbgCreateInfo, nullptr, &debugMessenger_);
        }
    }
#endif

    uint32_t pdevCount = 0;
    vkEnumeratePhysicalDevices(instance, &pdevCount, nullptr);
    if (pdevCount == 0) {
        std::cerr << "No Vulkan physical devices found\n";
        return false;
    }

    std::vector<VkPhysicalDevice> pdevs(pdevCount);
    vkEnumeratePhysicalDevices(instance, &pdevCount, pdevs.data());

    for (auto& pdev : pdevs) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(pdev, &props);
        if (props.vendorID == 0x1002) {
            physicalDevice = pdev;
            std::cout << "Selected AMD GPU: " << props.deviceName << "\n";
            break;
        }
    }
    if (!physicalDevice) {
        physicalDevice = pdevs[0];
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(physicalDevice, &props);
        vendorID = props.vendorID;
        deviceApiVersion = props.apiVersion;
        std::strncpy(deviceName, props.deviceName, VK_MAX_PHYSICAL_DEVICE_NAME_SIZE - 1);
        std::cout << "Selected GPU: " << deviceName << " (vendor=" << vendorID << ")\n";
    } else {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(physicalDevice, &props);
        vendorID = props.vendorID;
        deviceApiVersion = props.apiVersion;
        std::strncpy(deviceName, props.deviceName, VK_MAX_PHYSICAL_DEVICE_NAME_SIZE - 1);
        std::cout << "Selected GPU: " << deviceName << " (vendor=" << vendorID << ")\n";
        std::cout << "Device Vulkan version: " << VK_VERSION_MAJOR(deviceApiVersion)
                  << "." << VK_VERSION_MINOR(deviceApiVersion)
                  << "." << VK_VERSION_PATCH(deviceApiVersion) << "\n";
    }

    // Query subgroup/wave size + AMD shader core properties for RDNA detection.
    // On RDNA4 (gfx1201) the default is wave64, but wave32 is ~2x faster for compute
    // because it uses 1 cycle instead of 2 (wave64 folds onto 32 lanes × 2 cycles).
    // llama.cpp PR 19625 proved 193% prefill uplift from switching to wave32 on RDNA4.
    {
        VkPhysicalDeviceSubgroupProperties subgroupProps = {};
        subgroupProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;

        VkPhysicalDeviceSubgroupSizeControlProperties subgroupSizeControlProps = {};
        subgroupSizeControlProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES;

        VkPhysicalDeviceShaderCorePropertiesAMD amdCoreProps = {};
        amdCoreProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CORE_PROPERTIES_AMD;

        VkPhysicalDeviceShaderCoreProperties2AMD amdCoreProps2 = {};
        amdCoreProps2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CORE_PROPERTIES_2_AMD;

        VkPhysicalDeviceProperties2 props2 = {};
        props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        props2.pNext = &subgroupProps;
        subgroupProps.pNext = &subgroupSizeControlProps;
        subgroupSizeControlProps.pNext = &amdCoreProps;
        amdCoreProps.pNext = &amdCoreProps2;
        vkGetPhysicalDeviceProperties2(physicalDevice, &props2);

        subgroupSize = subgroupProps.subgroupSize;
        std::cout << "Subgroup size (wave width): " << subgroupSize << "\n";
        std::cout << "Subgroup min/max via size control: "
                  << subgroupSizeControlProps.minSubgroupSize << "/"
                  << subgroupSizeControlProps.maxSubgroupSize << "\n";
        std::cout << "Subgroup supported stages: 0x" << std::hex
                  << subgroupProps.supportedStages << std::dec << "\n";

        if (isAmd()) {
            std::cout << "AMD shader engines: " << amdCoreProps.shaderEngineCount
                      << "  arrays/engine: " << amdCoreProps.shaderArraysPerEngineCount
                      << "  CUs/array: " << amdCoreProps.computeUnitsPerShaderArray
                      << "  SIMD/CU: " << amdCoreProps.simdPerComputeUnit
                      << "  wavefrontSize: " << amdCoreProps.wavefrontSize
                      << "  wavefronts/SIMD: " << amdCoreProps.wavefrontsPerSimd
                      << "  active CUs: " << amdCoreProps2.activeComputeUnitCount << "\n";

            bool wave32Available = subgroupSizeControlProps.minSubgroupSize <= 32 &&
                                   subgroupSizeControlProps.maxSubgroupSize >= 32;
            forceWave32 = wave32Available;

            // Detect RDNA4 vs RDNA3 vs GCN:
            // RDNA4 (gfx1201) defaults to wave64 but supports wave32 via pipeline control
            // RDNA3 (gfx115x/gfx110x) defaults to wave32 natively
            // GCN (gfx8xx/gfx9xx) is wave64-only
            if (wave32Available && subgroupSize == 64) {
                // Default is wave64 but wave32 is available = RDNA4
                amdGeneration = 4;
                forceWave32 = true;
                subgroupSize = 32;
                std::cout << "AMD RDNA4 detected — forcing WAVE32 via pipeline subgroup size control\n";
                std::cout << "Matches llama.cpp PR 19625 pattern (193% prefill uplift)\n";
            } else if (subgroupSize <= 32) {
                amdGeneration = 3;
                std::cout << "AMD RDNA3 detected (native wave32)\n";
            } else {
                amdGeneration = 1;
                std::cout << "AMD GCN detected (wave64 only — using subgroup fallbacks)\n";
            }
        } else {
            std::cout << "Non-AMD GPU, using default wave size\n";
        }
    }

    // Find compute queue family with at least 4 queues
    uint32_t qfCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &qfCount, nullptr);
    std::vector<VkQueueFamilyProperties> qfProps(qfCount);
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &qfCount, qfProps.data());

    // Log all queue families
    for (uint32_t i = 0; i < qfCount; ++i) {
        std::cout << "Queue family " << i << ": flags=0x" << std::hex << qfProps[i].queueFlags
                  << std::dec << " queues=" << qfProps[i].queueCount
                  << " (graphics=" << !!(qfProps[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)
                  << " compute=" << !!(qfProps[i].queueFlags & VK_QUEUE_COMPUTE_BIT)
                  << " transfer=" << !!(qfProps[i].queueFlags & VK_QUEUE_TRANSFER_BIT)
                  << " sparse=" << !!(qfProps[i].queueFlags & VK_QUEUE_SPARSE_BINDING_BIT)
                  << ")\n";
    }

    // Pass 1: prefer compute-only queue family (no graphics bit) with >= 4 queues
    for (uint32_t i = 0; i < qfCount; ++i) {
        bool hasCompute = !!(qfProps[i].queueFlags & VK_QUEUE_COMPUTE_BIT);
        bool hasGraphics = !!(qfProps[i].queueFlags & VK_QUEUE_GRAPHICS_BIT);
        if (hasCompute && !hasGraphics && qfProps[i].queueCount >= 4) {
            queueFamilyIndex = i;
            std::cout << "[Router] SELECTED family " << i << ": COMPUTE-ONLY (no graphics), "
                      << qfProps[i].queueCount << " queues, flags=0x" << std::hex << qfProps[i].queueFlags
                      << std::dec << "\n";
            break;
        }
    }

    // Pass 2: compute-only with fewer queues
    if (queueFamilyIndex == 0xFFFFFFFF) {
        for (uint32_t i = 0; i < qfCount; ++i) {
            if ((qfProps[i].queueFlags & VK_QUEUE_COMPUTE_BIT) &&
                !(qfProps[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
                queueFamilyIndex = i;
                std::cout << "[Router] SELECTED family " << i << ": COMPUTE-ONLY (fewer queues), flags=0x"
                          << std::hex << qfProps[i].queueFlags << std::dec << "\n";
                break;
            }
        }
    }

    // Pass 3: graphics+compute
    if (queueFamilyIndex == 0xFFFFFFFF) {
        for (uint32_t i = 0; i < qfCount; ++i) {
            if ((qfProps[i].queueFlags & VK_QUEUE_COMPUTE_BIT) && qfProps[i].queueCount >= 4) {
                queueFamilyIndex = i;
                std::cout << "[Router] WARNING: family " << i << " has GRAPHICS+COMPUTE, flags=0x"
                          << std::hex << qfProps[i].queueFlags << std::dec
                          << " — will route to 3D engine, not Compute_0!\n";
                break;
            }
        }
    }

    // Pass 4: any compute queue family
    if (queueFamilyIndex == 0xFFFFFFFF) {
        for (uint32_t i = 0; i < qfCount; ++i) {
            if (qfProps[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                queueFamilyIndex = i;
                std::cout << "[Router] WARNING: family " << i << " fallback, flags=0x"
                          << std::hex << qfProps[i].queueFlags << std::dec
                          << " — may route to 3D engine!\n";
                break;
            }
        }
    }

    if (queueFamilyIndex == 0xFFFFFFFF) {
        std::cerr << "No compute queue family found\n";
        return false;
    }

    // Find dedicated transfer family (no compute, no graphics) — like llama.cpp
    uint32_t transferFamilyIndex = 0xFFFFFFFF;
    for (uint32_t i = 0; i < qfCount; ++i) {
        if ((qfProps[i].queueFlags & VK_QUEUE_TRANSFER_BIT) &&
            !(qfProps[i].queueFlags & VK_QUEUE_COMPUTE_BIT) &&
            !(qfProps[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
            transferFamilyIndex = i;
            std::cout << "[Router] Transfer family " << i << ": flags=0x" << std::hex
                      << qfProps[i].queueFlags << std::dec << "\n";
            break;
        }
    }

    uint32_t queueCount = std::min(4u, qfProps[queueFamilyIndex].queueCount);
    float priorities[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    float transferPriority = 1.0f;

    // Build queue create infos — compute + transfer (like llama.cpp)
    std::vector<VkDeviceQueueCreateInfo> queueInfos;
    {
        VkDeviceQueueCreateInfo qci = {};
        qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        qci.queueFamilyIndex = queueFamilyIndex;
        qci.queueCount = queueCount;
        qci.pQueuePriorities = priorities;
        queueInfos.push_back(qci);
    }
    if (transferFamilyIndex != 0xFFFFFFFF && transferFamilyIndex != queueFamilyIndex) {
        VkDeviceQueueCreateInfo qci = {};
        qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        qci.queueFamilyIndex = transferFamilyIndex;
        qci.queueCount = 1;
        qci.pQueuePriorities = &transferPriority;
        queueInfos.push_back(qci);
        std::cout << "Requesting " << queueCount << " compute queues from family "
                  << queueFamilyIndex << " + 1 transfer queue from family "
                  << transferFamilyIndex << "\n";
    } else {
        std::cout << "Requesting " << queueCount << " queues from family "
                  << queueFamilyIndex << " (no separate transfer family)\n";
    }

    // Enumerate device extensions — check for AMD-specific extensions and cooperative matrix
    uint32_t extCount = 0;
    vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &extCount, nullptr);
    std::vector<VkExtensionProperties> exts(extCount);
    vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &extCount, exts.data());

    bool hasCoopMat = false;
    for (auto& e : exts) {
        if (strcmp(e.extensionName, VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME) == 0) {
            hasCoopMat = true;
        }
        if (strcmp(e.extensionName, VK_AMD_GPU_SHADER_HALF_FLOAT_EXTENSION_NAME) == 0) {
            hasAMDShaderHalfFloat = true;
        }
        if (strcmp(e.extensionName, VK_AMD_SHADER_BALLOT_EXTENSION_NAME) == 0) {
            hasAMDShaderBallot = true;
        }
        if (strcmp(e.extensionName, VK_KHR_SHADER_SUBGROUP_ROTATE_EXTENSION_NAME) == 0) {
            hasShaderSubgroupRotate = true;
        }
    }
    std::cout << "VK_KHR_cooperative_matrix: " << (hasCoopMat ? "AVAILABLE" : "not found") << "\n";
    std::cout << "VK_AMD_gpu_shader_half_float: " << (hasAMDShaderHalfFloat ? "AVAILABLE" : "not found") << "\n";
    std::cout << "VK_AMD_shader_ballot: " << (hasAMDShaderBallot ? "AVAILABLE" : "not found") << "\n";
    std::cout << "VK_KHR_shader_subgroup_rotate: " << (hasShaderSubgroupRotate ? "AVAILABLE" : "not found") << "\n";

    // Query cooperative matrix properties (load function dynamically — not in vulkan-1.lib)
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR coopFeat{};
    coopFeat.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR;
    bool enableCoopMat = false;
    if (hasCoopMat) {
        auto vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR =
            (PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR)
            vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR");
        if (vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR) {
            VkPhysicalDeviceCooperativeMatrixPropertiesKHR coopProps = {};
            coopProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_PROPERTIES_KHR;

            VkPhysicalDeviceProperties2 props2 = {};
            props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
            props2.pNext = &coopProps;
            vkGetPhysicalDeviceProperties2(physicalDevice, &props2);

            std::cout << "Cooperative matrix cooperativeMatrixSupportedStages: "
                      << coopProps.cooperativeMatrixSupportedStages << "\n";

            uint32_t coopCount = 0;
            vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR(physicalDevice, &coopCount, nullptr);
            std::vector<VkCooperativeMatrixPropertiesKHR> coopPropsList(coopCount);
            for (auto& cp : coopPropsList) {
                cp.sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
            }
            vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR(physicalDevice, &coopCount, coopPropsList.data());

            for (auto& cp : coopPropsList) {
                std::cout << "  CoopMatrix: " << cp.MSize << "x" << cp.NSize << "x" << cp.KSize
                          << " AType=" << cp.AType << " BType=" << cp.BType
                          << " CType=" << cp.CType << " ResultType=" << cp.ResultType
                          << " saturatingAccumulation=" << cp.saturatingAccumulation
                          << " scope=" << cp.scope << "\n";
            }

            // Enable KHR cooperative matrix if a usable fp16xfp16+fp32 16x16x16 config exists.
            for (auto& cp : coopPropsList) {
                if (cp.MSize == 16 && cp.NSize == 16 && cp.KSize == 16 &&
                    cp.AType == VK_COMPONENT_TYPE_FLOAT16_KHR &&
                    cp.BType == VK_COMPONENT_TYPE_FLOAT16_KHR &&
                    (cp.CType == VK_COMPONENT_TYPE_FLOAT32_KHR || cp.CType == VK_COMPONENT_TYPE_FLOAT16_KHR) &&
                    (cp.ResultType == VK_COMPONENT_TYPE_FLOAT32_KHR || cp.ResultType == VK_COMPONENT_TYPE_FLOAT16_KHR) &&
                    cp.scope == VK_SCOPE_SUBGROUP_KHR) {
                    enableCoopMat = true;
                    break;
                }
            }
        }
    }

    // ====== DEVICE FEATURES: Compute-only minimal feature set ======
    // Query the full pNext chain so we know what the driver supports, then
    // explicitly zero every struct and enable only the features the compute
    // stack actually needs. Enabling graphics-only features such as
    // dynamicRendering on a compute-only queue family has been observed to
    // corrupt the AMD WDDM driver stack (0xC0000409), so those stay disabled.

    VkPhysicalDeviceVulkan14Features feat14 = {};
    feat14.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_4_FEATURES;

    VkPhysicalDeviceVulkan13Features feat13 = {};
    feat13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;

    VkPhysicalDeviceVulkan12Features feat12 = {};
    feat12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;

    VkPhysicalDeviceVulkan11Features feat11 = {};
    feat11.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;

    // Cooperative matrix features intentionally omitted — dead end on AMD Vulkan.
    VkPhysicalDeviceFeatures2 features2 = {};
    features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features2.pNext = &feat14;
    feat14.pNext = &feat13;
    feat13.pNext = &feat12;
    feat12.pNext = &feat11;
    feat11.pNext = nullptr;

    vkGetPhysicalDeviceFeatures2(physicalDevice, &features2);

    features2.features = {};
    features2.features.shaderInt64 = VK_TRUE;
    features2.features.shaderInt16 = VK_TRUE;

    feat11 = {};
    feat11.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
    feat11.storageBuffer16BitAccess = VK_TRUE;

    feat12 = {};
    feat12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    feat12.bufferDeviceAddress = VK_TRUE;
    feat12.shaderFloat16 = VK_TRUE;
    feat12.shaderInt8 = VK_TRUE;
    feat12.storageBuffer8BitAccess = VK_TRUE;
    feat12.uniformAndStorageBuffer8BitAccess = VK_TRUE;
    feat12.scalarBlockLayout = VK_TRUE;
    feat12.timelineSemaphore = VK_TRUE;
    feat12.descriptorIndexing = VK_TRUE;
    feat12.descriptorBindingStorageBufferUpdateAfterBind = VK_TRUE;
    feat12.descriptorBindingPartiallyBound = VK_TRUE;
    feat12.descriptorBindingVariableDescriptorCount = VK_TRUE;
    feat12.runtimeDescriptorArray = VK_TRUE;

    feat13 = {};
    feat13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    feat13.synchronization2 = VK_TRUE;
    feat13.maintenance4 = VK_TRUE;
    feat13.subgroupSizeControl = VK_TRUE;
    feat13.computeFullSubgroups = VK_TRUE;

    feat14 = {}; // zero entirely — defensive
    feat14.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_4_FEATURES;

    // Rebuild pNext chain after zeroing
    features2.pNext = &feat14;
    feat14.pNext = &feat13;
    feat13.pNext = &feat12;
    feat12.pNext = &feat11;
    feat11.pNext = &coopFeat;
    coopFeat.pNext = nullptr;

    // Cooperative matrix features
    coopFeat.cooperativeMatrix = enableCoopMat ? VK_TRUE : VK_FALSE;

    // Build extension list — AMD extensions for wave32, DPP, native FP16.
    // Cooperative matrix is enabled when the driver advertises a usable
    // fp16×fp16+fp32 16×16×16 subgroup-scoped configuration. The actual
    // cooperative-matrix shader path still requires a compiler that supports
    // GL_KHR_cooperative_matrix (bundled glslc 1.4.350 does not).
    std::vector<const char*> enabledExts;
    if (enableCoopMat) {
        enabledExts.push_back(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);
        std::cout << "Enabling VK_KHR_cooperative_matrix device extension\n";
    } else {
        std::cout << "No usable VK_KHR_cooperative_matrix config; keeping wave32+DPP primary\n";
    }
    if (hasAMDShaderHalfFloat) enabledExts.push_back(VK_AMD_GPU_SHADER_HALF_FLOAT_EXTENSION_NAME);
    if (hasAMDShaderBallot) enabledExts.push_back(VK_AMD_SHADER_BALLOT_EXTENSION_NAME);
    if (hasShaderSubgroupRotate) enabledExts.push_back(VK_KHR_SHADER_SUBGROUP_ROTATE_EXTENSION_NAME);

    VkDeviceCreateInfo devInfo = {};
    devInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    devInfo.queueCreateInfoCount = static_cast<uint32_t>(queueInfos.size());
    devInfo.pQueueCreateInfos = queueInfos.data();
    devInfo.pNext = &features2;
    devInfo.enabledExtensionCount = static_cast<uint32_t>(enabledExts.size());
    devInfo.ppEnabledExtensionNames = enabledExts.empty() ? nullptr : enabledExts.data();

    r = vkCreateDevice(physicalDevice, &devInfo, nullptr, &device);
    if (r != VK_SUCCESS) {
        std::cerr << "vkCreateDevice failed: " << r << "\n";
        return false;
    }

    for (uint32_t i = 0; i < queueCount; ++i) {
        vkGetDeviceQueue(device, queueFamilyIndex, i, &queues[i]);
        std::cout << "[Router] Queue[" << i << "] handle=" << (void*)queues[i]
                  << " from family " << queueFamilyIndex << "\n";
    }
    for (uint32_t i = queueCount; i < 4; ++i) {
        queues[i] = queues[0];  // Duplicate if fewer than 4 queues
    }

    return true;
}

void VulkanContext::cleanup() {
    if (device) { vkDestroyDevice(device, nullptr); device = VK_NULL_HANDLE; }
    if (debugMessenger_ != VK_NULL_HANDLE) {
        auto destroyFn = (PFN_vkDestroyDebugUtilsMessengerEXT)
            vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
        if (destroyFn) destroyFn(instance, debugMessenger_, nullptr);
        debugMessenger_ = VK_NULL_HANDLE;
    }
    if (instance) { vkDestroyInstance(instance, nullptr); instance = VK_NULL_HANDLE; }
}

} // namespace rdna4
