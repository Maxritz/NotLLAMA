# The Synchronization 2 Advantage

## Key Points from Vulkan Tutorial:

### Legacy API Problem:
- srcStageMask and dstStageMask were function-level, not per-barrier
- Forced union of all stages = over-synchronization bottleneck
- Multiple barrier structures (VkMemoryBarrier, VkBufferMemoryBarrier, VkImageMemoryBarrier) were separate

### Synchronization 2 Solution:
- Unifies barriers into vk::DependencyInfo container
- Each barrier has its own srcStageMask/dstStageMask (64-bit)
- vk::PipelineStageFlagBits2::eNone for explicit no-op
- Granular control: driver can start transition on one resource while another is still busy

### Compute-to-Compute Pattern (from PhysicsSystem example):
`cpp
// Legacy (still works but suboptimal)
cmd.pipelineBarrier(
    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
    {},
    VkMemoryBarrier{VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT},
    nullptr, nullptr);
`

### Modern Sync2 Pattern:
`cpp
VkMemoryBarrier2 memoryBarrier{
    .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
    .srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT,
    .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
    .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT
};

VkDependencyInfo dependencyInfo{
    .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
    .memoryBarrierCount = 1,
    .pMemoryBarriers = &memoryBarrier
};

vkCmdPipelineBarrier2(cmd, &dependencyInfo);
`

### Our Implementation Matches:
- srcStage = COMPUTE_SHADER (0x800)
- dstStage = COMPUTE_SHADER (0x800)
- srcAccess = SHADER_WRITE (0x40)
- dstAccess = SHADER_READ (0x20)
- Global VkMemoryBarrier (not per-resource)
