#include "rdna4_vulkan.hpp"
#include <iostream>
#include <vector>
#include <cstring>

namespace rdna4 {

bool VulkanContext::init() {
    VkApplicationInfo appInfo = {};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "RDNA4 LLaMA";
    appInfo.apiVersion = VK_API_VERSION_1_4;

    VkInstanceCreateInfo instInfo = {};
    instInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instInfo.pApplicationInfo = &appInfo;

    VkResult r = vkCreateInstance(&instInfo, nullptr, &instance);
    if (r != VK_SUCCESS) {
        std::cerr << "vkCreateInstance failed: " << r << "\n";
        return false;
    }

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
        std::cout << "Selected GPU: " << props.deviceName << " (vendor=" << props.vendorID << ")\n";
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

    // Enumerate device extensions and check for cooperative matrix
    uint32_t extCount = 0;
    vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &extCount, nullptr);
    std::vector<VkExtensionProperties> exts(extCount);
    vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &extCount, exts.data());

    bool hasCoopMat = false;
    for (auto& e : exts) {
        if (strcmp(e.extensionName, VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME) == 0) {
            hasCoopMat = true;
        }
    }
    std::cout << "VK_KHR_cooperative_matrix: " << (hasCoopMat ? "AVAILABLE" : "not found") << "\n";

    // Query cooperative matrix properties (load function dynamically — not in vulkan-1.lib)
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
        }
    }

    // ====== DEVICE FEATURES: Query all supported, preserve in pNext chain ======
    // llama.cpp pattern: call vkGetPhysicalDeviceFeatures2 with the FULL pNext chain.
    // This populates every boolean with what the driver supports. Passing the same
    // populated chain to vkCreateDevice signals "modern capable app" and prevents
    // AMD from routing compute to the 3D engine as a compatibility fallback.

    VkPhysicalDeviceCooperativeMatrixFeaturesKHR coopFeatures = {};
    coopFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR;

    VkPhysicalDeviceVulkan14Features feat14 = {};
    feat14.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_4_FEATURES;

    VkPhysicalDeviceVulkan13Features feat13 = {};
    feat13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;

    VkPhysicalDeviceVulkan12Features feat12 = {};
    feat12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;

    VkPhysicalDeviceVulkan11Features feat11 = {};
    feat11.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;

    // Build pNext chain: features2 -> feat14 -> feat13 -> feat12 -> feat11 -> coopFeatures
    VkPhysicalDeviceFeatures2 features2 = {};
    features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features2.pNext = &feat14;
    feat14.pNext = &feat13;
    feat13.pNext = &feat12;
    feat12.pNext = &feat11;
    feat11.pNext = hasCoopMat ? &coopFeatures : nullptr;
    if (hasCoopMat) coopFeatures.pNext = nullptr;

    // Query ALL supported features in ONE call — fills every boolean in the chain
    vkGetPhysicalDeviceFeatures2(physicalDevice, &features2);

    // ====== LLAMA.CPP FEATURE PATTERN: PRESERVE ALL QUERIED FEATURES ======
    // llama.cpp queries the full pNext chain with vkGetPhysicalDeviceFeatures2,
    // then passes the SAME populated chain to vkCreateDevice unchanged.
    // The only modification: zero pipelineCacheUUID from feat13 (not a feature).
    //
    // This signals "modern capable app" to the AMD WDDM driver and prevents
    // it from classifying our app as "legacy" and routing compute through
    // the 3D engine. On RDNA4 the graphics queue family is actually ~56%
    // faster for compute, and this pattern ensures the driver sees a mature
    // Vulkan 1.4 application capable of running on any queue family.
    //
    // The queue family selection above determines WHERE work runs (ACE or
    // 3D engine); the feature chain determines HOW the driver classifies us.
    // ==================================================

    // All queried features from the FULL vkGetPhysicalDeviceFeatures2 chain
    // are already populated in features2/feat14/feat13/feat12/feat11/coopFeatures.
    // Pass them through as-is to vkCreateDevice (llama.cpp pattern).
    // No feature zeroing — the driver sees the app as it truly is.

    // Build extension list
    std::vector<const char*> enabledExts;
    if (hasCoopMat) enabledExts.push_back(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);

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
    if (instance) { vkDestroyInstance(instance, nullptr); instance = VK_NULL_HANDLE; }
}

} // namespace rdna4
