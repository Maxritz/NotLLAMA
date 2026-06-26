#include "rdna4_profiler.hpp"
#include <iostream>
#include <iomanip>
#include <cstring>

namespace rdna4 {

Profiler::Profiler(VkDevice dev, VkPhysicalDevice pdev)
    : device(dev), physicalDevice(pdev), queryCount(256), nextQuery(0) {

    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(pdev, &props);
    timestampPeriod = props.limits.timestampPeriod;

    VkQueryPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    poolInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
    poolInfo.queryCount = queryCount;

    vkCreateQueryPool(device, &poolInfo, nullptr, &queryPool);
    vkResetQueryPool(device, queryPool, 0, queryCount);
}

Profiler::~Profiler() {
    if (queryPool) vkDestroyQueryPool(device, queryPool, nullptr);
}

void Profiler::beginCpu(const std::string& name, uint32_t ace) {
    active[name] = {ace, std::chrono::high_resolution_clock::now()};
}

void Profiler::endCpu(const std::string& name) {
    auto it = active.find(name);
    if (it == active.end()) return;

    auto end = std::chrono::high_resolution_clock::now();
    auto start = it->second.second;
    auto dur = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    events.push_back({name, static_cast<uint64_t>(dur), 0, 0, it->second.first});
    active.erase(it);
}

uint32_t Profiler::allocateQueryRange(uint32_t count) {
    uint32_t start = nextQuery;
    nextQuery += count;
    if (nextQuery >= queryCount) {
        // Wrap around (simplified: just reset)
        vkResetQueryPool(device, queryPool, 0, queryCount);
        nextQuery = count;
        start = 0;
    }
    return start;
}

void Profiler::writeTimestamp(VkCommandBuffer cmd, uint32_t queryIndex, VkPipelineStageFlagBits stage) {
    vkCmdWriteTimestamp(cmd, stage, queryPool, queryIndex);
}

void Profiler::recordGpuTime(const std::string& name, uint32_t startQuery, uint32_t endQuery, uint32_t ace) {
    uint64_t timestamps[2];
    VkResult r = vkGetQueryPoolResults(device, queryPool, startQuery, 2, sizeof(timestamps),
                                        timestamps, sizeof(uint64_t),
                                        VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
    if (r != VK_SUCCESS) return;

    uint64_t gpuStart = static_cast<uint64_t>(timestamps[0] * timestampPeriod);
    uint64_t gpuEnd = static_cast<uint64_t>(timestamps[1] * timestampPeriod);

    events.push_back({name, 0, 0, gpuStart, gpuEnd, ace});
}

void Profiler::readbackQueries() {
    // Called before report to ensure all queries are ready
}

void Profiler::report() const {
    std::cout << "\n=== Performance Report ===\n";
    std::cout << std::setw(30) << "Kernel" << std::setw(12) << "CPU (us)"
              << std::setw(12) << "GPU (us)" << std::setw(8) << "ACE" << "\n";
    std::cout << std::string(62, '-') << "\n";

    for (const auto& e : events) {
        std::cout << std::setw(30) << e.name;
        if (e.startNs > 0) {
            std::cout << std::setw(12) << (e.startNs / 1000);
        } else {
            std::cout << std::setw(12) << "-";
        }
        if (e.gpuEndNs > e.gpuStartNs) {
            std::cout << std::setw(12) << ((e.gpuEndNs - e.gpuStartNs) / 1000);
        } else {
            std::cout << std::setw(12) << "-";
        }
        std::cout << std::setw(8) << e.aceIndex << "\n";
    }
}

void Profiler::reset() {
    events.clear();
    active.clear();
    nextQuery = 0;
    vkResetQueryPool(device, queryPool, 0, queryCount);
}

} // namespace rdna4
