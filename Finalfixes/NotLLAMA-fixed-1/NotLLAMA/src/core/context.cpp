#include "core/context.h"
#include <set>
#include <algorithm>
#include <iostream>
#include <cstring>

static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT type,
    const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
    void* pUserData)
{
    if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
        std::cerr << "[VK] " << pCallbackData->pMessage << "\n";
    }
    return VK_FALSE;
}

namespace notllama {

Context::Context() {
    createInstance();
    pickDevice();
    createDevice();
}

Context::~Context() {
    if (device_) {
        for (auto& ace : aces_) {
            vkDestroyCommandPool(device_, ace.pool, nullptr);
        }
        vkDestroyCommandPool(device_, transferPool_, nullptr);
        vkDestroyDevice(device_, nullptr);
    }
    if (debugMessenger_) {
        auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)
            vkGetInstanceProcAddr(instance_, "vkDestroyDebugUtilsMessengerEXT");
        if (func) func(instance_, debugMessenger_, nullptr);
    }
    if (instance_) vkDestroyInstance(instance_, nullptr);
}

void Context::createInstance() {
    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "NotLLAMA";
    appInfo.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
    appInfo.pEngineName = "RDNA4";
    appInfo.engineVersion = VK_MAKE_VERSION(0, 1, 0);
    appInfo.apiVersion = VK_API_VERSION_1_3;

    std::vector<const char*> extensions = {
        VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME,
    };

    std::vector<const char*> layers;
#ifdef NDEBUG
    // No validation in release
#else
    layers.push_back("VK_LAYER_KHRONOS_validation");
    extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
#endif

    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;
    createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    createInfo.ppEnabledExtensionNames = extensions.data();
    createInfo.enabledLayerCount = static_cast<uint32_t>(layers.size());
    createInfo.ppEnabledLayerNames = layers.data();

    VK_CHECK(vkCreateInstance(&createInfo, nullptr, &instance_));

#ifndef NDEBUG
    auto func = (PFN_vkCreateDebugUtilsMessengerEXT)
        vkGetInstanceProcAddr(instance_, "vkCreateDebugUtilsMessengerEXT");
    if (func) {
        VkDebugUtilsMessengerCreateInfoEXT dbgInfo{};
        dbgInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        dbgInfo.messageSeverity =
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        dbgInfo.messageType =
            VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        dbgInfo.pfnUserCallback = debugCallback;
        func(instance_, &dbgInfo, nullptr, &debugMessenger_);
    }
#endif
}

void Context::pickDevice() {
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance_, &deviceCount, nullptr);
    if (deviceCount == 0)
        throw std::runtime_error("No Vulkan GPUs found");

    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(instance_, &deviceCount, devices.data());

    // Prefer discrete AMD GPU (RDNA4 target)
    for (auto& dev : devices) {
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(dev, &props);
        if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            physDevice_ = dev;
            gpuInfo_.devProps = props;
            break;
        }
    }
    if (physDevice_ == VK_NULL_HANDLE) {
        physDevice_ = devices[0];
        vkGetPhysicalDeviceProperties(physDevice_, &gpuInfo_.devProps);
    }

    gpuInfo_.name = gpuInfo_.devProps.deviceName;
    vkGetPhysicalDeviceMemoryProperties(physDevice_, &gpuInfo_.memProps);

    // Detect wave/subgroup size
    VkPhysicalDeviceSubgroupProperties subgroupProps{};
    subgroupProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
    VkPhysicalDeviceProperties2 props2{};
    props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    props2.pNext = &subgroupProps;
    vkGetPhysicalDeviceProperties2(physDevice_, &props2);

    gpuInfo_.subgroupSize = subgroupProps.subgroupSize;
    // RDNA4: wave32. If subgroupSize >= 64, it's wave64 (older GCN).
    gpuInfo_.waveSize = (subgroupProps.subgroupSize <= 32) ? 32 : 64;

    std::cout << "GPU: " << gpuInfo_.name << "\n";
    std::cout << "Wave size: " << gpuInfo_.waveSize << "\n";
    std::cout << "Subgroup size: " << gpuInfo_.subgroupSize << "\n";
}

void Context::createDevice() {
    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physDevice_, &queueFamilyCount, nullptr);
    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(physDevice_, &queueFamilyCount, queueFamilies.data());

    // Find compute-only families (no graphics) — route to ACE, not 3D engine
    std::vector<uint32_t> computeFamilies, transferFamily;
    for (uint32_t i = 0; i < queueFamilyCount; i++) {
        if ((queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT) &&
            !(queueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
            computeFamilies.push_back(i);
        }
        if ((queueFamilies[i].queueFlags & VK_QUEUE_TRANSFER_BIT) &&
            !(queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT) &&
            !(queueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
            transferFamily.push_back(i);
        }
    }
    // Fallback: if no pure compute family exists, use first compute-capable one
    if (computeFamilies.empty()) {
        std::cerr << "[WARN] No pure compute family; falling back to graphics+compute\n";
        for (uint32_t i = 0; i < queueFamilyCount; i++) {
            if (queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                computeFamilies.push_back(i);
                break;
            }
        }
    }
    if (computeFamilies.empty())
        throw std::runtime_error("No compute queue family found");
    if (transferFamily.empty())
        transferFamily.push_back(computeFamilies[0]); // fall back to compute

    // Create device with one queue per ACE
    float priority = 1.0f;
    std::vector<VkDeviceQueueCreateInfo> queueInfos;
    std::set<uint32_t> uniqueFamilies;

    uint32_t aceCount = std::min(static_cast<uint32_t>(computeFamilies.size()), 4u);

    for (uint32_t i = 0; i < aceCount; i++) {
        uint32_t fam = computeFamilies[i % computeFamilies.size()];
        if (uniqueFamilies.insert(fam).second) {
            VkDeviceQueueCreateInfo qi{};
            qi.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            qi.queueFamilyIndex = fam;
            qi.queueCount = 1;
            qi.pQueuePriorities = &priority;
            queueInfos.push_back(qi);
        }
    }
    // Transfer queue
    uint32_t tf = transferFamily[0];
    if (uniqueFamilies.insert(tf).second) {
        VkDeviceQueueCreateInfo qi{};
        qi.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        qi.queueFamilyIndex = tf;
        qi.queueCount = 1;
        qi.pQueuePriorities = &priority;
        queueInfos.push_back(qi);
    }

    // Required features
    VkPhysicalDeviceFeatures2 deviceFeatures{};
    deviceFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;

    // Vulkan 1.1+ subgroup features
    VkPhysicalDeviceSubgroupFeatures subgroupFeatures{};
    subgroupFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_FEATURES;
    subgroupFeatures.subgroupBroadcastDynamicId = VK_TRUE;
    deviceFeatures.pNext = &subgroupFeatures;

    // Vulkan 1.2 features
    VkPhysicalDeviceVulkan12Features feat12{};
    feat12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    feat12.storageBuffer8BitAccess = VK_TRUE;
    feat12.uniformAndStorageBuffer8BitAccess = VK_TRUE;
    feat12.shaderFloat16 = VK_TRUE;
    feat12.shaderInt8 = VK_TRUE;
    feat12.bufferDeviceAddress = VK_TRUE;
    subgroupFeatures.pNext = &feat12;

    // Vulkan 1.3 features
    VkPhysicalDeviceVulkan13Features feat13{};
    feat13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    feat13.dynamicRendering = VK_TRUE;
    feat13.synchronization2 = VK_TRUE;
    feat13.maintenance4 = VK_TRUE;
    feat12.pNext = &feat13;

    const char* deviceExtensions[] = {
        VK_KHR_SHADER_FLOAT_INT8_EXTENSION_NAME,
        VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME,
    };

    VkDeviceCreateInfo deviceInfo{};
    deviceInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceInfo.queueCreateInfoCount = static_cast<uint32_t>(queueInfos.size());
    deviceInfo.pQueueCreateInfos = queueInfos.data();
    deviceInfo.enabledExtensionCount = 2;
    deviceInfo.ppEnabledExtensionNames = deviceExtensions;
    deviceInfo.pNext = &deviceFeatures;

    VK_CHECK(vkCreateDevice(physDevice_, &deviceInfo, nullptr, &device_));

    // Collect ACE queues
    for (uint32_t i = 0; i < aceCount; i++) {
        uint32_t fam = computeFamilies[i % computeFamilies.size()];
        VkQueue q;
        vkGetDeviceQueue(device_, fam, 0, &q);

        VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        poolInfo.queueFamilyIndex = fam;
        VkCommandPool pool;
        VK_CHECK(vkCreateCommandPool(device_, &poolInfo, nullptr, &pool));

        aces_.push_back({fam, q, pool});
    }

    // Transfer queue
    vkGetDeviceQueue(device_, tf, 0, &transferQueue_);
    VkCommandPoolCreateInfo tPoolInfo{};
    tPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    tPoolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    tPoolInfo.queueFamilyIndex = tf;
    VK_CHECK(vkCreateCommandPool(device_, &tPoolInfo, nullptr, &transferPool_));

    std::cout << "ACEs (compute queues): " << aces_.size() << "\n";
}

uint32_t Context::findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags props) const {
    for (uint32_t i = 0; i < gpuInfo_.memProps.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) &&
            (gpuInfo_.memProps.memoryTypes[i].propertyFlags & props) == props) {
            return i;
        }
    }
    throw std::runtime_error("Failed to find suitable memory type");
}

uint32_t Context::findImageMemoryType(uint32_t typeFilter) const {
    // Prefer device-local for images
    for (uint32_t i = 0; i < gpuInfo_.memProps.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) &&
            (gpuInfo_.memProps.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
            return i;
        }
    }
    return findMemoryType(typeFilter, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
}

VkQueue Context::aceQueue(uint32_t aceIndex) const {
    return aces_.at(aceIndex).queue;
}

VkCommandPool Context::acePool(uint32_t aceIndex) const {
    return aces_.at(aceIndex).pool;
}

} // namespace notllama
