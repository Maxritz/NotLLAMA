#include "rdna4_vulkan.hpp"
#include <iostream>
#include <vector>
#include <cstring>

namespace rdna4 {

bool VulkanContext::init() {
    // ====== INSTANCE CREATION ======
    // We request Vulkan 1.4 from the loader, but the physical device may
    // only support 1.3 (e.g., RDNA2 with older drivers). We handle this by:
    // 1. Creating a 1.4 instance (loader provides 1.4 if SDK is new enough)
    // 2. Querying each device's actual apiVersion
    // 3. Capping our feature chain to what the device supports
    VkApplicationInfo appInfo = {};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "NotLLAMA Vulkan Inference";
    appInfo.apiVersion = VK_API_VERSION_1_4;

    VkInstanceCreateInfo instInfo = {};
    instInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instInfo.pApplicationInfo = &appInfo;

    VkResult r = vkCreateInstance(&instInfo, nullptr, &instance);
    if (r != VK_SUCCESS) {
        std::cerr << "vkCreateInstance failed: " << r << "\n";
        // Fallback: try Vulkan 1.3
        appInfo.apiVersion = VK_API_VERSION_1_3;
        r = vkCreateInstance(&instInfo, nullptr, &instance);
        if (r != VK_SUCCESS) {
            std::cerr << "vkCreateInstance (1.3 fallback) failed: " << r << "\n";
            return false;
        }
        std::cout << "Using Vulkan 1.3 instance\n";
    }

    // ====== PHYSICAL DEVICE SELECTION ======
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
            vendorID = props.vendorID;
            deviceApiVersion = props.apiVersion;
            std::strncpy(deviceName, props.deviceName, VK_MAX_PHYSICAL_DEVICE_NAME_SIZE - 1);
            std::cout << "Selected AMD GPU: " << deviceName << "\n";
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
    }

    std::cout << "Device Vulkan version: " << VK_VERSION_MAJOR(deviceApiVersion)
              << "." << VK_VERSION_MINOR(deviceApiVersion)
              << "." << VK_VERSION_PATCH(deviceApiVersion) << "\n";

    // ====== SUBGROUP/WAVE SIZE QUERY ======
    // This determines if we're on Wave32 (RDNA3/4) or Wave64 (RDNA2/GCN).
    // Shaders are written to work with either, but knowing the wave size
    // helps with performance tuning.
    VkPhysicalDeviceSubgroupProperties subgroupProps = {};
    subgroupProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;

    VkPhysicalDeviceProperties2 props2 = {};
    props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    props2.pNext = &subgroupProps;
    vkGetPhysicalDeviceProperties2(physicalDevice, &props2);

    subgroupSize = subgroupProps.subgroupSize;
    std::cout << "Subgroup size (wave width): " << subgroupSize << "\n";
    std::cout << "Subgroup supported stages: 0x" << std::hex
              << subgroupProps.supportedStages << std::dec << "\n";
    if (isAmd()) {
        if (isWave32()) {
            std::cout << "AMD Wave32 mode (RDNA3/4) — using 32-wide dispatches\n";
        } else {
            std::cout << "AMD Wave64 mode (RDNA2/GCN) — using 64-wide dispatches\n";
        }
    }

    // ====== QUEUE FAMILY SELECTION ======
    uint32_t qfCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &qfCount, nullptr);
    std::vector<VkQueueFamilyProperties> qfProps(qfCount);
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &qfCount, qfProps.data());

    for (uint32_t i = 0; i < qfCount; ++i) {
        std::cout << "Queue family " << i << ": flags=0x" << std::hex << qfProps[i].queueFlags
                  << std::dec << " queues=" << qfProps[i].queueCount
                  << " (graphics=" << !!(qfProps[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)
                  << " compute=" << !!(qfProps[i].queueFlags & VK_QUEUE_COMPUTE_BIT)
                  << " transfer=" << !!(qfProps[i].queueFlags & VK_QUEUE_TRANSFER_BIT)
                  << ")\n";
    }

    // Pass 1: prefer compute-only (no graphics) with >= 4 queues
    for (uint32_t i = 0; i < qfCount; ++i) {
        bool hasCompute = !!(qfProps[i].queueFlags & VK_QUEUE_COMPUTE_BIT);
        bool hasGraphics = !!(qfProps[i].queueFlags & VK_QUEUE_GRAPHICS_BIT);
        if (hasCompute && !hasGraphics && qfProps[i].queueCount >= 4) {
            queueFamilyIndex = i;
            std::cout << "[Router] SELECTED family " << i << ": COMPUTE-ONLY, "
                      << qfProps[i].queueCount << " queues\n";
            break;
        }
    }
    // Pass 2: compute-only with fewer queues
    if (queueFamilyIndex == 0xFFFFFFFF) {
        for (uint32_t i = 0; i < qfCount; ++i) {
            if ((qfProps[i].queueFlags & VK_QUEUE_COMPUTE_BIT) &&
                !(qfProps[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
                queueFamilyIndex = i;
                std::cout << "[Router] SELECTED family " << i << ": COMPUTE-ONLY (fewer)\n";
                break;
            }
        }
    }
    // Pass 3: graphics+compute
    if (queueFamilyIndex == 0xFFFFFFFF) {
        for (uint32_t i = 0; i < qfCount; ++i) {
            if ((qfProps[i].queueFlags & VK_QUEUE_COMPUTE_BIT) && qfProps[i].queueCount >= 4) {
                queueFamilyIndex = i;
                std::cout << "[Router] WARNING: family " << i
                          << " has GRAPHICS+COMPUTE — routes to 3D engine!\n";
                break;
            }
        }
    }
    // Pass 4: any compute
    if (queueFamilyIndex == 0xFFFFFFFF) {
        for (uint32_t i = 0; i < qfCount; ++i) {
            if (qfProps[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                queueFamilyIndex = i;
                std::cout << "[Router] WARNING: family " << i << " fallback\n";
                break;
            }
        }
    }
    if (queueFamilyIndex == 0xFFFFFFFF) {
        std::cerr << "No compute queue family found\n";
        return false;
    }

    // Dedicated transfer family
    uint32_t transferFamilyIndex = 0xFFFFFFFF;
    for (uint32_t i = 0; i < qfCount; ++i) {
        if ((qfProps[i].queueFlags & VK_QUEUE_TRANSFER_BIT) &&
            !(qfProps[i].queueFlags & VK_QUEUE_COMPUTE_BIT) &&
            !(qfProps[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
            transferFamilyIndex = i;
            break;
        }
    }

    uint32_t queueCount = std::min(4u, qfProps[queueFamilyIndex].queueCount);
    float priorities[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    float transferPriority = 1.0f;

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
    }

    // ====== EXTENSIONS ======
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

    // ====== FEATURE CHAIN: cap to device-supported version ======
    // Only include Vulkan 1.4 features in the pNext chain if the device
    // actually reports Vulkan 1.4 support. Otherwise the query is invalid.
    bool deviceSupports14 = VK_VERSION_MAJOR(deviceApiVersion) >= 1 &&
                            VK_VERSION_MINOR(deviceApiVersion) >= 4;

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

    VkPhysicalDeviceFeatures2 features2 = {};
    features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;

    // Build pNext chain conditionally based on device Vulkan version:
    //   features2 -> feat14 (if 1.4) -> feat13 -> feat12 -> feat11 -> coopFeatures
    void** pNextPtr = &features2.pNext;

    if (deviceSupports14) {
        *pNextPtr = &feat14;
        feat14.pNext = &feat13;
        pNextPtr = &feat14.pNext;
    } else {
        *pNextPtr = &feat13;
    }
    feat13.pNext = &feat12;
    feat12.pNext = &feat11;
    feat11.pNext = hasCoopMat ? &coopFeatures : nullptr;
    if (hasCoopMat) coopFeatures.pNext = nullptr;

    // Query supported features (only structs in chain get filled)
    vkGetPhysicalDeviceFeatures2(physicalDevice, &features2);

    // ====== COMPUTE-ONLY FEATURE ENABLEMENT ======
    // Zero everything, then enable ONLY compute-relevant features.
    // This prevents AMD WDDM from routing compute to the 3D engine.

    VkPhysicalDeviceFeatures baseCompute = {};
    baseCompute.shaderInt64               = VK_TRUE;
    baseCompute.shaderInt16               = VK_TRUE;
    features2.features = baseCompute;

    // Vulkan 1.1
    {
        void* saved_pNext = feat11.pNext;
        feat11 = {};
        feat11.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
        feat11.pNext = saved_pNext;
        feat11.storageBuffer16BitAccess = VK_TRUE;
    }

    // Vulkan 1.2
    {
        void* saved_pNext = feat12.pNext;
        feat12 = {};
        feat12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
        feat12.pNext = saved_pNext;
        feat12.bufferDeviceAddress               = VK_TRUE;
        feat12.shaderFloat16                     = VK_TRUE;
        feat12.shaderInt8                        = VK_TRUE;
        feat12.storageBuffer8BitAccess           = VK_TRUE;
        feat12.uniformAndStorageBuffer8BitAccess = VK_TRUE;
        feat12.scalarBlockLayout                 = VK_TRUE;
        feat12.timelineSemaphore                 = VK_TRUE;
    }

    // Vulkan 1.3
    {
        void* saved_pNext = feat13.pNext;
        feat13 = {};
        feat13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
        feat13.pNext = saved_pNext;
        feat13.dynamicRendering   = VK_FALSE;
        feat13.synchronization2   = VK_TRUE;
        feat13.maintenance4       = VK_TRUE;
    }

    // Vulkan 1.4 (only if supported)
    if (deviceSupports14) {
        void* saved_pNext = feat14.pNext;
        feat14 = {};
        feat14.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_4_FEATURES;
        feat14.pNext = saved_pNext;
        // No 1.4 features needed for compute; leave all disabled
        std::cout << "Vulkan 1.4 features: enabled (device supports)\n";
    } else {
        std::cout << "Vulkan 1.4 features: SKIPPED (device only supports "
                  << VK_VERSION_MAJOR(deviceApiVersion) << "."
                  << VK_VERSION_MINOR(deviceApiVersion) << ")\n";
    }

    if (hasCoopMat) {
        coopFeatures.cooperativeMatrix = VK_TRUE;
    }

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
        queues[i] = queues[0];
    }

    return true;
}

void VulkanContext::cleanup() {
    if (device) { vkDestroyDevice(device, nullptr); device = VK_NULL_HANDLE; }
    if (instance) { vkDestroyInstance(instance, nullptr); instance = VK_NULL_HANDLE; }
}

} // namespace rdna4
