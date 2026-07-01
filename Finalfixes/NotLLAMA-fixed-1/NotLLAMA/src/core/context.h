#pragma once
#define VK_USE_PLATFORM_WIN32_KHR
#include <vulkan/vulkan.h>
#include <vector>
#include <string>
#include <stdexcept>
#include <functional>
#include <memory>
#include <optional>
#include <unordered_map>

#define VK_CHECK(call) do { \
    VkResult _r = (call); \
    if (_r != VK_SUCCESS) \
        throw std::runtime_error("Vulkan error " + std::to_string(_r) + " at " + __FILE__ + ":" + std::to_string(__LINE__)); \
} while(0)

namespace notllama {

struct QueueFamily {
    uint32_t index = UINT32_MAX;
    VkQueue queue = VK_NULL_HANDLE;
    VkQueueFamilyProperties props{};
};

struct GPUInfo {
    std::string name;
    VkPhysicalDeviceMemoryProperties memProps;
    VkPhysicalDeviceProperties devProps;
    uint32_t waveSize; // 32 for RDNA4
    uint32_t subgroupSize;
    std::vector<QueueFamily> computeFamilies;
    std::vector<QueueFamily> transferFamilies;
};

class Context {
public:
    Context();
    ~Context();

    VkInstance instance() const { return instance_; }
    VkPhysicalDevice physicalDevice() const { return physDevice_; }
    VkDevice device() const { return device_; }
    const GPUInfo& gpuInfo() const { return gpuInfo_; }

    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags props) const;
    uint32_t findImageMemoryType(uint32_t typeFilter) const;

    // ACE (Asynchronous Compute Engine) queues
    VkQueue aceQueue(uint32_t aceIndex) const;
    VkCommandPool acePool(uint32_t aceIndex) const;
    uint32_t aceCount() const { return static_cast<uint32_t>(aces_.size()); }

    VkQueue transferQueue() const { return transferQueue_; }

private:
    void createInstance();
    void pickDevice();
    void createDevice();

    VkInstance instance_ = VK_NULL_HANDLE;
    VkPhysicalDevice physDevice_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    GPUInfo gpuInfo_{};

    struct ACE {
        uint32_t queueFamily;
        VkQueue queue;
        VkCommandPool pool;
    };
    std::vector<ACE> aces_;

    VkQueue transferQueue_ = VK_NULL_HANDLE;
    VkCommandPool transferPool_ = VK_NULL_HANDLE;

    VkDebugUtilsMessengerEXT debugMessenger_ = VK_NULL_HANDLE;
};

} // namespace notllama
