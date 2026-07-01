#include "vulkan/vk_device.h"
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <vector>

static bool has_extension(const std::vector<VkExtensionProperties>& available, const char* name) {
    for (auto& ext : available)
        if (!strcmp(ext.extensionName, name)) return true;
    return false;
}

static VkComponentTypeKHR component_from_string(const char* s) {
    if (!s) return VK_COMPONENT_TYPE_MAX_ENUM_KHR;
    if (!strcmp(s, "float16")) return VK_COMPONENT_TYPE_FLOAT16_KHR;
    if (!strcmp(s, "float32")) return VK_COMPONENT_TYPE_FLOAT32_KHR;
    return VK_COMPONENT_TYPE_MAX_ENUM_KHR;
}

DeviceInfo create_device(VkInstance instance) {
    DeviceInfo info{};

    uint32_t count = 0;
    vkEnumeratePhysicalDevices(instance, &count, nullptr);
    if (!count) { fprintf(stderr, "No Vulkan devices\n"); return info; }
    std::vector<VkPhysicalDevice> devices(count);
    vkEnumeratePhysicalDevices(instance, &count, devices.data());

    VkPhysicalDevice selected = VK_NULL_HANDLE;
    uint32_t selected_qf = UINT32_MAX;
    for (auto pd : devices) {
        VkPhysicalDeviceProperties p{};
        vkGetPhysicalDeviceProperties(pd, &p);
        uint32_t qcount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(pd, &qcount, nullptr);
        std::vector<VkQueueFamilyProperties> qprops(qcount);
        vkGetPhysicalDeviceQueueFamilyProperties(pd, &qcount, qprops.data());
        for (uint32_t i = 0; i < qcount; i++) {
            if (qprops[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                selected = pd;
                selected_qf = i;
                break;
            }
        }
        if (selected) {
            printf("Device: %s (queue family %u)\n", p.deviceName, selected_qf);
            break;
        }
    }
    if (!selected) { fprintf(stderr, "No compute-capable device\n"); return info; }
    info.phys_dev = selected;
    info.compute_queue_family = selected_qf;
    vkGetPhysicalDeviceProperties(selected, &info.props);

    // Check cooperative matrix support
    uint32_t ext_count = 0;
    vkEnumerateDeviceExtensionProperties(selected, nullptr, &ext_count, nullptr);
    std::vector<VkExtensionProperties> available_exts(ext_count);
    vkEnumerateDeviceExtensionProperties(selected, nullptr, &ext_count, available_exts.data());

    VkPhysicalDeviceCooperativeMatrixFeaturesKHR cm_features{};
    cm_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR;
    VkPhysicalDeviceFeatures2 f2{};
    f2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    f2.pNext = &cm_features;
    vkGetPhysicalDeviceFeatures2(selected, &f2);

    bool cm_ext_available = has_extension(available_exts, VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);

    // Load cooperative matrix properties function dynamically
    auto vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR =
        (PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR)
        vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR");

    if (cm_features.cooperativeMatrix && cm_ext_available && vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR) {
        uint32_t prop_count = 0;
        vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR(selected, &prop_count, nullptr);
        std::vector<VkCooperativeMatrixPropertiesKHR> props(prop_count);
        for (auto& p : props) { p.sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR; p.pNext = nullptr; }
        vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR(selected, &prop_count, props.data());

        info.coop_mat.supported = true;
        for (auto& p : props) {
            if (p.MSize == 16 && p.NSize == 16 && p.KSize == 16 &&
                p.AType == VK_COMPONENT_TYPE_FLOAT16_KHR &&
                p.BType == VK_COMPONENT_TYPE_FLOAT16_KHR &&
                p.CType == VK_COMPONENT_TYPE_FLOAT32_KHR &&
                p.scope == VK_SCOPE_SUBGROUP_KHR) {
                printf("Coop matrix: fp16xfp16+fp32 16x16x16 SUPPORTED\n");
                break;
            }
        }
    } else {
        printf("Cooperative matrix: NOT supported\n");
    }

    // Subgroup size and wave32 support
    VkPhysicalDeviceSubgroupProperties subgroup_props{};
    subgroup_props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
    VkPhysicalDeviceSubgroupSizeControlPropertiesEXT ssc_props{};
    ssc_props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES_EXT;
    subgroup_props.pNext = &ssc_props;
    VkPhysicalDeviceProperties2 p2{};
    p2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    p2.pNext = &subgroup_props;
    vkGetPhysicalDeviceProperties2(selected, &p2);
    info.subgroup_size = subgroup_props.subgroupSize;
    info.max_subgroup_size = ssc_props.maxSubgroupSize;
    info.min_subgroup_size = ssc_props.minSubgroupSize;
    info.wave32_supported = ssc_props.minSubgroupSize <= 32 && ssc_props.maxSubgroupSize >= 32;
    printf("Subgroup size: %u (min=%u max=%u wave32=%s)\n",
           info.subgroup_size, info.min_subgroup_size, info.max_subgroup_size,
           info.wave32_supported ? "yes" : "no");

    // Build device
    float queue_priority = 1.0f;
    VkDeviceQueueCreateInfo qci{};
    qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qci.queueFamilyIndex = selected_qf;
    qci.queueCount = 1;
    qci.pQueuePriorities = &queue_priority;

    // Features chain
    VkPhysicalDeviceVulkan11Features f11{};
    f11.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
    f11.storageBuffer16BitAccess = VK_TRUE;
    f11.uniformAndStorageBuffer16BitAccess = VK_TRUE;

    VkPhysicalDeviceVulkan12Features f12{};
    f12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    f12.bufferDeviceAddress = VK_TRUE;
    f12.shaderFloat16 = VK_TRUE;
    f12.shaderInt8 = VK_TRUE;
    f12.storageBuffer8BitAccess = VK_TRUE;
    f12.uniformAndStorageBuffer8BitAccess = VK_TRUE;
    f12.scalarBlockLayout = VK_TRUE;
    f12.timelineSemaphore = VK_TRUE;

    VkPhysicalDeviceVulkan13Features f13{};
    f13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    f13.synchronization2 = VK_TRUE;
    f13.maintenance4 = VK_TRUE;
    f13.dynamicRendering = VK_FALSE;
    f13.subgroupSizeControl = VK_TRUE;
    f13.computeFullSubgroups = VK_TRUE;

    void* next = &f11;
    f11.pNext = &f12;
    f12.pNext = &f13;

    if (info.coop_mat.supported) {
        cm_features.pNext = f13.pNext;
        f13.pNext = &cm_features;
    }

    // Extensions
    std::vector<const char*> exts = {
        VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME,
        "VK_KHR_maintenance4",
        VK_KHR_16BIT_STORAGE_EXTENSION_NAME,
        VK_KHR_8BIT_STORAGE_EXTENSION_NAME,
        VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME,
        VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME,
        VK_EXT_SCALAR_BLOCK_LAYOUT_EXTENSION_NAME,
        VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME,
        VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME,
    };

    // VK_EXT_subgroup_size_control is core in Vulkan 1.3; we set
    // f13.subgroupSizeControl/computeFullSubgroups above.
    // The extension name is included for robustness on 1.2 fallback.
    if (has_extension(available_exts, VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME))
        exts.push_back(VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME);

    VkDeviceCreateInfo dci{};
    dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = (uint32_t)exts.size();
    dci.ppEnabledExtensionNames = exts.data();
    dci.pNext = next;

    VkResult res = vkCreateDevice(selected, &dci, nullptr, &info.device);
    if (res != VK_SUCCESS) { fprintf(stderr, "vkCreateDevice failed: %d\n", res); return info; }

    vkGetDeviceQueue(info.device, selected_qf, 0, &info.compute_queue);
    return info;
}

void destroy_device(const DeviceInfo& info) {
    if (info.device) vkDestroyDevice(info.device, nullptr);
}
