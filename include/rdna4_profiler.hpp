#pragma once
#include <vulkan/vulkan.h>
#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
#include <chrono>

namespace rdna4 {

struct ProfileEvent {
    std::string name;
    uint64_t startNs;
    uint64_t endNs;
    uint64_t gpuStartNs;
    uint64_t gpuEndNs;
    uint32_t aceIndex;
};

class Profiler {
public:
    VkDevice device;
    VkPhysicalDevice physicalDevice;

    VkQueryPool queryPool;
    uint32_t queryCount;
    uint32_t nextQuery;
    float timestampPeriod;  // ns per tick

    std::vector<ProfileEvent> events;
    std::unordered_map<std::string, std::pair<uint32_t, std::chrono::high_resolution_clock::time_point>> active;

    Profiler(VkDevice dev, VkPhysicalDevice pdev);
    ~Profiler();

    // CPU timing
    void beginCpu(const std::string& name, uint32_t ace = 0);
    void endCpu(const std::string& name);

    // GPU timing: returns query index pair for wrapping around dispatches
    uint32_t allocateQueryRange(uint32_t count);
    void writeTimestamp(VkCommandBuffer cmd, uint32_t queryIndex, VkPipelineStageFlagBits stage);
    void recordGpuTime(const std::string& name, uint32_t startQuery, uint32_t endQuery, uint32_t ace = 0);

    void report() const;
    void reset();

private:
    void readbackQueries();
};

} // namespace rdna4
