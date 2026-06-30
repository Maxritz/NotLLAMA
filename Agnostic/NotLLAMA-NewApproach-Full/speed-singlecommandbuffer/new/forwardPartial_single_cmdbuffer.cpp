uint32_t InferenceEngine::forwardPartial(uint32_t tokenId, uint32_t seqPos, uint32_t maxLayers) {
    uint32_t dim = model->embeddingLength;
    uint32_t headDim = dim / model->headCount;
    uint32_t hiddenDim = model->feedForwardLength;

    allocator->reset();

    uint32_t seqLen = seqPos + 1;
    size_t hiddenSize = dim * sizeof(float);
    size_t qkvSize = (size_t)seqLen * headDim * model->headCount * sizeof(float);
    size_t kvSize = (size_t)seqLen * headDim * model->headCountKv * sizeof(float);
    size_t attnOutSize = dim * sizeof(float);
    size_t mlpOutSize = dim * sizeof(float);
    size_t logitsSize = (size_t)model->vocabSize * sizeof(float);
    size_t sampleSize = 16;

    uint64_t hiddenAddr = allocator->alloc(hiddenSize);
    uint64_t qAddr = allocator->alloc(qkvSize);
    uint64_t kAddr = allocator->alloc(kvSize);
    uint64_t vAddr = allocator->alloc(kvSize);
    uint64_t attnOutAddr = allocator->alloc(attnOutSize);
    uint64_t mlpOutAddr = allocator->alloc(mlpOutSize);
    uint64_t logitsAddr = allocator->alloc(logitsSize);
    uint64_t sampleOutAddr = allocator->alloc(sampleSize);

    if (!hiddenAddr || !qAddr || !kAddr || !vAddr || !attnOutAddr || !mlpOutAddr || !logitsAddr || !sampleOutAddr) {
        fprintf(stderr, "[FATAL] Ring allocator overflow at seqPos=%u\n", seqPos);
        return 0;
    }

    uint64_t qRowAddr = qAddr + (uint64_t)seqPos * dim * sizeof(float);
    uint64_t kRowAddr = kAddr + (uint64_t)seqPos * headDim * model->headCountKv * sizeof(float);
    uint64_t vRowAddr = vAddr + (uint64_t)seqPos * headDim * model->headCountKv * sizeof(float);

    // === ONE COMMAND BUFFER FOR ENTIRE FORWARD PASS ===
    scheduler->beginBatch(0);

    // === Embedding ===
    uint64_t embedAddr = findTensorAddr(*model, "token_embd.weight");
    if (embedAddr) {
        uint64_t embedAddr_dq = embedCacheReady ? embedCacheAddr : 0;
        if (!embedAddr_dq) {
            embedAddr_dq = dequantWeightInBatch(scheduler, pipelines, dequantAddr, dequantCapacity, *model, "token_embd.weight");
        }
        EmbedPushConstants embedPC = {embedAddr_dq ? embedAddr_dq : embedAddr, hiddenAddr, tokenId, seqPos, dim};
        scheduler->dispatchInBatch(pipelines->getPipeline("embed"), pipelines->getLayout("embed"),
            &embedPC, sizeof(embedPC), (dim + 31) / 32, 1, 1);
        scheduler->barrierBetweenGroups(
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
    }

    uint32_t layersToRun = std::min(maxLayers, model->blockCount);

    for (uint32_t layer = 0; layer < layersToRun; ++layer) {
        std::string prefix = "blk." + std::to_string(layer);

        // === Attention Dequant ===
        size_t qSize = (size_t)dim * dim * sizeof(float);
        size_t kSize = (size_t)(headDim * model->headCountKv) * dim * sizeof(float);
        size_t vSize = kSize;

        uint64_t addrAttnNorm_dq = dequantWeightInBatch(scheduler, pipelines, dequantAddr, dequantCapacity, *model, prefix + ".attn_norm.weight", 0);
        uint64_t addrQW_dq = dequantWeightInBatch(scheduler, pipelines, dequantAddr, dequantCapacity, *model, prefix + ".attn_q.weight", 0);
        uint64_t addrKW_dq = dequantWeightInBatch(scheduler, pipelines, dequantAddr, dequantCapacity, *model, prefix + ".attn_k.weight", qSize);
        uint64_t addrVW_dq = dequantWeightInBatch(scheduler, pipelines, dequantAddr, dequantCapacity, *model, prefix + ".attn_v.weight", qSize + kSize);
        uint64_t addrOW_dq = dequantWeightInBatch(scheduler, pipelines, dequantAddr, dequantCapacity, *model, prefix + ".attn_output.weight", qSize + kSize + vSize);

        scheduler->barrierBetweenGroups(
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);

        // === Attention Compute ===
        uint64_t addrAttnNorm = findTensorAddr(*model, prefix + ".attn_norm.weight");
        RmsNormPushConstants normPC = {hiddenAddr, addrAttnNorm_dq ? addrAttnNorm_dq : addrAttnNorm, attnOutAddr, dim, 1, 1e-6f};
        scheduler->dispatchInBatch(pipelines->getPipeline("rms_norm"), pipelines->getLayout("rms_norm"),
            &normPC, sizeof(normPC), 1, 1, 1);

        uint64_t normHidden = attnOutAddr;
        uint64_t addrQW = findTensorAddr(*model, prefix + ".attn_q.weight");
        uint64_t addrKW = findTensorAddr(*model, prefix + ".attn_k.weight");
        uint64_t addrVW = findTensorAddr(*model, prefix + ".attn_v.weight");

        GemmPushConstants qPC = {normHidden, addrQW_dq ? addrQW_dq : addrQW, qRowAddr, 1, dim, dim, 1.0f, 0};
        scheduler->dispatchInBatch(pipelines->getPipeline("gemm"), pipelines->getLayout("gemm"),
            &qPC, sizeof(qPC), (dim + 31) / 32, 1, 1);
        GemmPushConstants kPC = {normHidden, addrKW_dq ? addrKW_dq : addrKW, kRowAddr, 1, (uint32_t)(headDim * model->headCountKv), dim, 1.0f, 0};
        scheduler->dispatchInBatch(pipelines->getPipeline("gemm"), pipelines->getLayout("gemm"),
            &kPC, sizeof(kPC), ((uint32_t)(headDim * model->headCountKv) + 31) / 32, 1, 1);
        GemmPushConstants vPC = {normHidden, addrVW_dq ? addrVW_dq : addrVW, vRowAddr, 1, (uint32_t)(headDim * model->headCountKv), dim, 1.0f, 0};
        scheduler->dispatchInBatch(pipelines->getPipeline("gemm"), pipelines->getLayout("gemm"),
            &vPC, sizeof(vPC), ((uint32_t)(headDim * model->headCountKv) + 31) / 32, 1, 1);

        scheduler->barrierBetweenGroups(
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);

        uint32_t ropeTotal = model->headCount * headDim;
        RopePushConstants ropePC = {qRowAddr, kRowAddr, seqLen, headDim, model->headCount, model->headCountKv, 10000.0f, 1.0f};
        scheduler->dispatchInBatch(pipelines->getPipeline("rope"), pipelines->getLayout("rope"),
            &ropePC, sizeof(ropePC), (ropeTotal + 31) / 32, 1, 1);

        uint64_t kCacheAddr = kvCache->getKBufferAddress(layer);
        uint64_t vCacheAddr = kvCache->getVBufferAddress(layer);
        if (kCacheAddr && vCacheAddr) {
            KVCacheWritePushConstants kvPC = {kRowAddr, vRowAddr, kCacheAddr, vCacheAddr, seqPos, headDim, model->headCountKv};
            scheduler->dispatchInBatch(pipelines->getPipeline("kv_cache_write"), pipelines->getLayout("kv_cache_write"),
                &kvPC, sizeof(kvPC), (headDim * model->headCountKv + 31) / 32, 1, 1);
        }

        for (uint32_t h = 0; h < model->headCount; ++h) {
            AttentionPushConstants attnPC = {qAddr, kCacheAddr, vCacheAddr, attnOutAddr, seqLen, headDim,
                model->headCount, model->headCountKv, h, 1.0f / std::sqrt(static_cast<float>(headDim))};
            scheduler->dispatchInBatch(pipelines->getPipeline("attention"), pipelines->getLayout("attention"),
                &attnPC, sizeof(attnPC), 1, 1, 1);
        }

        scheduler->barrierBetweenGroups(
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);

        uint64_t addrOW = findTensorAddr(*model, prefix + ".attn_output.weight");
        GemmPushConstants outPC = {attnOutAddr, addrOW_dq ? addrOW_dq : addrOW, mlpOutAddr, 1, dim, dim, 1.0f, 0};
        scheduler->dispatchInBatch(pipelines->getPipeline("gemm"), pipelines->getLayout("gemm"),
            &outPC, sizeof(outPC), (dim + 31) / 32, 1, 1);

        AddPushConstants addPC1 = {hiddenAddr, mlpOutAddr, hiddenAddr, dim};
        scheduler->dispatchInBatch(pipelines->getPipeline("add"), pipelines->getLayout("add"),
            &addPC1, sizeof(addPC1), (dim + 255) / 256, 1, 1);

        // Barrier: attention done → FFN dequant can overwrite buffer
        scheduler->barrierBetweenGroups(
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT,
            VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT);

        // === FFN Dequant ===
        uint64_t addrFfnNorm_dq = dequantWeightInBatch(scheduler, pipelines, dequantAddr, dequantCapacity, *model, prefix + ".ffn_norm.weight", 0);
        size_t upSize = getF16OutputSize(*model, prefix + ".ffn_up.weight");
        size_t gateSize = getF16OutputSize(*model, prefix + ".ffn_gate.weight");

        uint64_t addrUpW_dq = dequantWeightInBatch(scheduler, pipelines, dequantAddr, dequantCapacity, *model, prefix + ".ffn_up.weight", 0);
        uint64_t addrGateW_dq = dequantWeightInBatch(scheduler, pipelines, dequantAddr, dequantCapacity, *model, prefix + ".ffn_gate.weight", upSize);
        uint64_t addrDownW_dq = dequantWeightInBatch(scheduler, pipelines, dequantAddr, dequantCapacity, *model, prefix + ".ffn_down.weight", upSize + gateSize);

        scheduler->barrierBetweenGroups(
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);

        // === FFN Compute ===
        uint64_t addrFfnNorm = findTensorAddr(*model, prefix + ".ffn_norm.weight");
        RmsNormPushConstants ffnNormPC = {hiddenAddr, addrFfnNorm_dq ? addrFfnNorm_dq : addrFfnNorm, attnOutAddr, dim, 1, 1e-6f};
        scheduler->dispatchInBatch(pipelines->getPipeline("rms_norm"), pipelines->getLayout("rms_norm"),
            &ffnNormPC, sizeof(ffnNormPC), 1, 1, 1);

        scheduler->barrierBetweenGroups(
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);

        uint64_t addrUpW = findTensorAddr(*model, prefix + ".ffn_up.weight");
        uint64_t addrGateW = findTensorAddr(*model, prefix + ".ffn_gate.weight");
        uint64_t addrDownW = findTensorAddr(*model, prefix + ".ffn_down.weight");

        // NOTE: If your local tree replaced mlp_fused_gateup with separate
        // gate/up/silu_mul/down shaders, replace this block with the
        // separate dispatch sequence from your current code.
        MlpFusedGateUpPushConstants fusedPC = {attnOutAddr,
            addrGateW_dq ? addrGateW_dq : addrGateW,
            addrUpW_dq ? addrUpW_dq : addrUpW,
            addrDownW_dq ? addrDownW_dq : addrDownW,
            mlpOutAddr, dim, hiddenDim};
        scheduler->dispatchInBatch(pipelines->getPipeline("mlp_fused_gateup"), pipelines->getLayout("mlp_fused_gateup"),
            &fusedPC, sizeof(fusedPC), (dim + 31) / 32, 1, 1);

        AddPushConstants addPC2 = {hiddenAddr, mlpOutAddr, hiddenAddr, dim};
        scheduler->dispatchInBatch(pipelines->getPipeline("add"), pipelines->getLayout("add"),
            &addPC2, sizeof(addPC2), (dim + 255) / 256, 1, 1);

        // Barrier: FFN done → next layer can overwrite dequant buffer
        scheduler->barrierBetweenGroups(
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT,
            VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT);
    }

    // === Final Norm ===
    uint64_t addrOutNorm = findTensorAddr(*model, "output_norm.weight");
    if (!addrOutNorm) addrOutNorm = findTensorAddr(*model, "norm.weight");
    uint64_t addrOutNorm_dq = dequantWeightInBatch(scheduler, pipelines, dequantAddr, dequantCapacity, *model, "output_norm.weight", 0);

    scheduler->barrierBetweenGroups(
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);

    RmsNormPushConstants finalNorm = {hiddenAddr, addrOutNorm_dq ? addrOutNorm_dq : addrOutNorm, attnOutAddr, dim, 1, 1e-6f};
    scheduler->dispatchInBatch(pipelines->getPipeline("rms_norm"), pipelines->getLayout("rms_norm"),
        &finalNorm, sizeof(finalNorm), 1, 1, 1);

    scheduler->barrierBetweenGroups(
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);

    // === LM Head ===
    uint64_t addrLMHead = findTensorAddr(*model, "output.weight");
    uint64_t addrLMHead_dq = 0;
    uint32_t lmTransB = 0;
    if (addrLMHead) {
        addrLMHead_dq = dequantWeightInBatch(scheduler, pipelines, dequantAddr, dequantCapacity, *model, "output.weight", 0);
    } else {
        addrLMHead = findTensorAddr(*model, "token_embd.weight");
        if (embedCacheReady) {
            addrLMHead_dq = embedCacheAddr;
        } else {
            addrLMHead_dq = dequantWeightInBatch(scheduler, pipelines, dequantAddr, dequantCapacity, *model, "token_embd.weight", 0);
        }
        lmTransB = 1;
    }

    GemmPushConstants lmPC = {attnOutAddr, addrLMHead_dq ? addrLMHead_dq : addrLMHead, logitsAddr, 1, model->vocabSize, dim, 1.0f, lmTransB};
    scheduler->dispatchInBatch(pipelines->getPipeline("gemm"), pipelines->getLayout("gemm"),
        &lmPC, sizeof(lmPC), (model->vocabSize + 31) / 32, 1, 1);

    scheduler->barrierBetweenGroups(
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);

    // === TopK Sampling ===
    size_t sampleScratchSize = (size_t)model->vocabSize * sizeof(float);
    uint64_t sampleScratchAddr = allocator->alloc(sampleScratchSize);
    uint32_t seed = static_cast<uint32_t>(std::chrono::steady_clock::now().time_since_epoch().count());
    TopKPushConstants topkPC = {logitsAddr, sampleOutAddr, sampleScratchAddr, model->vocabSize, 1.0f, 40, 0.9f, seed};
    scheduler->dispatchInBatch(pipelines->getPipeline("topk"), pipelines->getLayout("topk"),
        &topkPC, sizeof(topkPC), (model->vocabSize + 255) / 256, 1, 1);

    // === ONE SUBMIT + ONE SYNC FOR ENTIRE TOKEN ===
    scheduler->endBatch(VK_NULL_HANDLE);
    scheduler->syncAll();

    // Update KV cache seq lengths for next token (after GPU is done)
    for (uint32_t layer = 0; layer < layersToRun; ++layer) {
        kvCache->incrementSeqLen(layer);
    }

    // Read back sampled token
    uint32_t nextToken = 0;
    if (sampleOutAddr && allocator->mappedPtr) {
        size_t localOffset = sampleOutAddr - allocator->baseAddress;
        size_t alignedOffset = localOffset & ~(static_cast<uint64_t>(4096) - 1);
        size_t alignedEnd = (localOffset + sizeof(uint32_t) + 4095) & ~(static_cast<uint64_t>(4095));
        VkMappedMemoryRange range = {};
        range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
        range.memory = allocator->memory;
        range.offset = alignedOffset;
        range.size = alignedEnd - alignedOffset;
        vkInvalidateMappedMemoryRanges(ctx->device, 1, &range);
        std::memcpy(&nextToken, allocator->mappedPtr + localOffset, sizeof(uint32_t));
    }
    if (!nextToken && allocator->mappedPtr) {
        // Fallback: CPU argmax on logits
        size_t logitsOffset = logitsAddr - allocator->baseAddress;
        size_t alignedOff = logitsOffset & ~(static_cast<uint64_t>(4095));
        size_t alignedEnd = (logitsOffset + logitsSize + 4095) & ~(static_cast<uint64_t>(4095));
        VkMappedMemoryRange range = {};
        range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
        range.memory = allocator->memory;
        range.offset = alignedOff;
        range.size = alignedEnd - alignedOff;
        vkInvalidateMappedMemoryRanges(ctx->device, 1, &range);
        float* cpuLogits = reinterpret_cast<float*>(allocator->mappedPtr + logitsOffset);
        if (cpuLogits) {
            nextToken = sampleArgmax(cpuLogits, model->vocabSize);
        }
    }

    lastLogitsAddr = logitsAddr;
    lastLogitsOffset = logitsAddr - allocator->baseAddress;
    return nextToken;
}
