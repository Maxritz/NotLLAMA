#include "engine/model_adapter.hpp"
#include "engine/imemory_allocator.hpp"
#include <algorithm>

namespace notllama {

ModelAdapter::ModelAdapter(VkDevice device, VkPhysicalDevice physical_device, uint32_t queue_family_index)
    : device_(device), physical_device_(physical_device), queue_family_index_(queue_family_index),
      uploader_(device, physical_device, queue_family_index) {}

std::string ModelAdapter::GetArchitecture() {
    return model_.architecture;
}

size_t ModelAdapter::GetNumLayers() {
    return model_.blockCount;
}

size_t ModelAdapter::GetHeadDim() {
    return model_.headDim;
}

size_t ModelAdapter::GetNumKVHeads() {
    return model_.headCountKv;
}

size_t ModelAdapter::GetEmbeddingDim() {
    return model_.embeddingLength;
}

float ModelAdapter::GetRoPEBase() {
    // TODO: read from model metadata
    return 10000.0f;
}

float ModelAdapter::GetRoPEScale() {
    // TODO: read from model metadata
    return 1.0f;
}

std::string ModelAdapter::RemapTensorName(const std::string& gguf_name) {
    // Identity remap for now; add architecture-specific mapping here.
    return gguf_name;
}

DataType ModelAdapter::ConvertFormat(rdna4::QuantFormat fmt) const {
    switch (fmt) {
        case rdna4::QuantFormat::F32: return DataType::F32;
        case rdna4::QuantFormat::F16: return DataType::F16;
        case rdna4::QuantFormat::Q4_0: return DataType::Q4_0;
        case rdna4::QuantFormat::Q4_1: return DataType::Q4_1;
        case rdna4::QuantFormat::Q5_0: return DataType::Q5_0;
        case rdna4::QuantFormat::Q5_1: return DataType::Q5_1;
        case rdna4::QuantFormat::Q8_0: return DataType::Q8_0;
        case rdna4::QuantFormat::Q4_K: return DataType::Q4_K;
        case rdna4::QuantFormat::Q5_K: return DataType::Q5_K;
        case rdna4::QuantFormat::Q6_K: return DataType::Q6_K;
        case rdna4::QuantFormat::Q8_K: return DataType::Q8_K;
        default: return DataType::UNKNOWN;
    }
}

std::vector<TensorMeta> ModelAdapter::GetWeightTensors() {
    std::vector<TensorMeta> result;
    result.reserve(model_.tensors.size());

    for (const auto& t : model_.tensors) {
        TensorMeta m{};
        m.num_dims = static_cast<uint32_t>(t.shape.size());
        for (size_t i = 0; i < t.shape.size() && i < 4; ++i) {
            m.dims[i] = t.shape[i];
        }
        m.dtype = ConvertFormat(t.format);
        m.block_size = t.blockSize;
        m.alloc.buffer = t.buffer;
        m.alloc.offset = 0;
        m.alloc.size = t.sizeBytes;
        m.alloc.device_address = t.gpuAddress;
        result.push_back(m);
    }

    return result;
}

std::vector<rdna4::TensorDesc> ModelAdapter::GetRawTensors() {
    return model_.tensors;
}

bool ModelAdapter::LoadFromPath(const std::string& path, IMemoryAllocator* allocator) {
    (void)allocator;
    std::string json_path = path;
    std::string bin_path = path;
    const size_t pos = bin_path.rfind(".json");
    if (pos != std::string::npos) {
        bin_path.replace(pos, 5, ".bin");
    } else {
        bin_path += ".bin";
    }

    try {
        model_ = uploader_.loadMetadata(json_path, bin_path);
        if (model_.tensors.empty()) return false;
        layers_loaded_.assign(model_.blockCount, false);
        return true;
    } catch (...) {
        return false;
    }
}

bool ModelAdapter::LoadFromGGUF(const std::string& path) {
    try {
        model_ = uploader_.loadFromGGUF(path);
        if (model_.tensors.empty()) return false;
        layers_loaded_.assign(model_.blockCount, false);
        return true;
    } catch (...) {
        return false;
    }
}

bool ModelAdapter::StreamLayerWeights(uint32_t layer_index, IMemoryAllocator* allocator) {
    (void)allocator;

    if (model_.tensors.empty()) {
        fprintf(stderr, "[ModelAdapter] No model loaded, cannot stream layer %u\n", layer_index);
        fflush(stderr);
        return false;
    }

    if (layer_index >= model_.blockCount) {
        fprintf(stderr, "[ModelAdapter] Layer %u out of range (max %u)\n", layer_index, model_.blockCount);
        fflush(stderr);
        return false;
    }

    if (layers_loaded_.size() > layer_index && layers_loaded_[layer_index]) {
        return true;
    }

    bool ok = uploader_.uploadLayer(model_, layer_index);
    if (ok && layers_loaded_.size() > layer_index) {
        layers_loaded_[layer_index] = true;
    }
    return ok;
}

bool ModelAdapter::UploadGlobalWeights() {
    if (model_.tensors.empty()) return false;
    bool all_ok = true;
    for (auto& t : model_.tensors) {
        if (t.name.compare(0, 4, "blk.") == 0) continue;
        if (t.gpuAddress != 0) continue;
        if (t.binSize == 0) continue;
        if (!uploader_.uploadTensor(t)) {
            fprintf(stderr, "[ModelAdapter] Failed to upload global tensor %s\n", t.name.c_str());
            fflush(stderr);
            all_ok = false;
        }
    }
    return all_ok;
}

const void* ModelAdapter::GetWeightShadowCopy(const std::string& tensor_name) {
    (void)tensor_name;
    // TODO: keep CPU shadow copy for recovery
    return nullptr;
}

} // namespace notllama
