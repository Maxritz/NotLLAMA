#pragma once
#include <vulkan/vulkan.h>
#include <vector>
#include <cstdint>

struct CoopMatConfig {
    bool supported = false;
    uint32_t m_size = 16;
    uint32_t n_size = 16;
    uint32_t k_size = 16;
};

struct DeviceInfo {
    VkPhysicalDevice phys_dev = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    uint32_t compute_queue_family = UINT32_MAX;
    VkQueue compute_queue = VK_NULL_HANDLE;
    CoopMatConfig coop_mat;
    uint32_t subgroup_size = 32;
    uint32_t min_subgroup_size = 32;
    uint32_t max_subgroup_size = 32;
    bool wave32_supported = false;
    uint32_t min_storage_alignment = 16;

    VkPhysicalDeviceProperties props{};
};

DeviceInfo create_device(VkInstance instance);
void destroy_device(const DeviceInfo& info);
