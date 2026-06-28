# Execution vs Memory Dependencies

## Two Types of Dependencies:

### 1. Execution Dependency (the WHEN)
- Tells GPU: "Don't start Command B until Command A finishes"
- Just ordering, not data visibility
- Stage masks define which pipeline stages are involved

### 2. Memory Dependency (the WHERE)
- Ensures data is visible between producer and consumer
- Two steps:
  - Availability: flush source caches to shared memory
  - Visibility: invalidate destination caches, force fresh read
- Access masks define what type of memory access is being synchronized

### Without Both = Hazards:
- RAW (Read-After-Write): fragment reads stale data from compute
- WAR (Write-After-Read): compute overwrites data fragment needs
- WAW (Write-After-Write): two writes reordered

### Cache Architecture (RDNA4):
- L1 cache per CU (shader core)
- L2 cache shared across CUs
- VRAM (device-local memory)
- Pipeline barriers flush L1->L2 and invalidate destination L1

### Practical Rule:
Every pipeline barrier needs BOTH:
1. Stage masks (execution dependency)
2. Access masks (memory dependency)
