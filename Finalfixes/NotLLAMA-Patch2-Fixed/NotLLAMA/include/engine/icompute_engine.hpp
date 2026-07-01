#pragma once
#include "types.hpp"
#include <cstdint>
#include <vector>

namespace notllama {

enum class WatchdogStatus {
    OK,
    HUNG,
    RECOVERED
};

class IComputeEngine {
public:
    virtual ~IComputeEngine() = default;

    virtual bool AddSequence(uint32_t seq_id, const std::vector<uint32_t>& tokens) = 0;
    virtual void RemoveSequence(uint32_t seq_id) = 0;
    virtual bool StepBatch() = 0;

    virtual void SetMaxUtilization(float percent) = 0;
    virtual void Throttle() = 0;

    virtual void StartWatchdog() = 0;
    virtual void StopWatchdog() = 0;
    virtual WatchdogStatus GetLastFrameStatus() = 0;

    virtual void ResetExecutionEngine() = 0;
    virtual void EnableProfiling(bool enable) = 0;
};

} // namespace notllama
