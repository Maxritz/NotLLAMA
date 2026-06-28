#include "rdna4_vulkan.hpp"
#include <iostream>
#include <vector>
#include <cstring>

namespace rdna4 {

bool VulkanContext::init() {
    VkApplicationInfo appInfo = {};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "RDNA4 LLaMA";
    appInfo.apiVersion = VK_API_VERSION_1_3;

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
        if ((qfProps[i].queueFlags & VK_QUEUE_COMPUTE_BIT) &&
            !(qfProps[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) &&
            qfProps[i].queueCount >= 4) {
            queueFamilyIndex = i;
            std::cout << "Queue family " << i << " (COMPUTE-ONLY) has " << qfProps[i].queueCount << " queues (using 4)\n";
            break;
        }
    }

    // Pass 2: compute-only with fewer queues
    if (queueFamilyIndex == 0xFFFFFFFF) {
        for (uint32_t i = 0; i < qfCount; ++i) {
            if ((qfProps[i].queueFlags & VK_QUEUE_COMPUTE_BIT) &&
                !(qfProps[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
                queueFamilyIndex = i;
                std::cout << "Queue family " << i << " (COMPUTE-ONLY fallback) has " << qfProps[i].queueCount
                          << " queues (using min(4, available))\n";
                break;
            }
        }
    }

    // Pass 3: graphics+compute
    if (queueFamilyIndex == 0xFFFFFFFF) {
        for (uint32_t i = 0; i < qfCount; ++i) {
            if ((qfProps[i].queueFlags & VK_QUEUE_COMPUTE_BIT) && qfProps[i].queueCount >= 4) {
                queueFamilyIndex = i;
                std::cout << "Queue family " << i << " (GRAPHICS+COMPUTE) has " << qfProps[i].queueCount
                          << " queues (using 4)\n";
                break;
            }
        }
    }

    // Pass 4: any compute queue family
    if (queueFamilyIndex == 0xFFFFFFFF) {
        for (uint32_t i = 0; i < qfCount; ++i) {
            if (qfProps[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                queueFamilyIndex = i;
                std::cout << "Queue family " << i << " has " << qfProps[i].queueCount
                          << " queues (fallback, using min(4, available))\n";
                break;
            }
        }
    }

    if (queueFamilyIndex == 0xFFFFFFFF) {
        std::cerr << "No compute queue family found\n";
        return false;
    }

    uint32_t queueCount = std::min(1u, qfProps[queueFamilyIndex].queueCount);
    float priorities[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    VkDeviceQueueCreateInfo qInfo = {};
    qInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qInfo.queueFamilyIndex = queueFamilyIndex;
    qInfo.queueCount = queueCount;
    qInfo.pQueuePriorities = priorities;

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

    // Enable required device features
    VkPhysicalDeviceVulkan11Features vulkan11Features = {};
    vulkan11Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
    vulkan11Features.storageBuffer16BitAccess = VK_TRUE;

    VkPhysicalDeviceVulkan12Features vulkan12Features = {};
    vulkan12Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    vulkan12Features.bufferDeviceAddress = VK_TRUE;

    VkPhysicalDeviceFeatures2 features2 = {};
    features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features2.features.shaderInt64 = VK_TRUE;
    features2.pNext = &vulkan11Features;
    vulkan11Features.pNext = &vulkan12Features;

    // Enable cooperative matrix feature if available
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR coopFeatures = {};
    if (hasCoopMat) {
        coopFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR;
        coopFeatures.cooperativeMatrix = VK_TRUE;
        coopFeatures.cooperativeMatrixRobustBufferAccess = VK_FALSE;
        vulkan12Features.pNext = &coopFeatures;
    }

    // Build extension list
    std::vector<const char*> enabledExts;
    if (hasCoopMat) enabledExts.push_back(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);

    VkDeviceCreateInfo devInfo = {};
    devInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    devInfo.queueCreateInfoCount = 1;
    devInfo.pQueueCreateInfos = &qInfo;
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
