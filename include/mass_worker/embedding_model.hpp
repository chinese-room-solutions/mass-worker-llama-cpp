#pragma once

#include <condition_variable>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "mass_worker/chat_model.hpp"  // for ModelError + ModelErrorCode
#include "mass_worker/llama_handles.hpp"

namespace mass_worker {

// Per-model embedding load configuration. Mirrors LlamaEmbeddingConfig.
struct EmbeddingModelLoadConfig {
    std::filesystem::path path;
    int32_t context_size{0};      // 0 → 4096
    int32_t gpu_layers{0};        // wire convention (0=auto-all, -1=cpu)
    int32_t threads{0};
    int32_t max_concurrent{1};    // pool size: number of contexts
    std::string main_gpu;
    std::vector<float> tensor_split;
};

struct EmbedPoolSlot;

// EmbeddingModel owns a llama_model + a pool of N contexts configured with
// embeddings=true and mean-pooling. Concurrent embed() calls borrow distinct
// contexts; the model weights are shared.
class EmbeddingModel {
    friend struct EmbedPoolSlot;
public:
    [[nodiscard]] static std::expected<std::shared_ptr<EmbeddingModel>, ModelError>
    load(EmbeddingModelLoadConfig cfg);

    ~EmbeddingModel();
    EmbeddingModel(const EmbeddingModel&) = delete;
    EmbeddingModel& operator=(const EmbeddingModel&) = delete;

    // Embed one input string. Returns the per-sequence pooled vector.
    [[nodiscard]] std::expected<std::vector<float>, ModelError>
    embed(const std::string& text);

    // Embed many inputs in sequence. Returns one vector per input in order.
    [[nodiscard]] std::expected<std::vector<std::vector<float>>, ModelError>
    embed_batch(const std::vector<std::string>& inputs);

    [[nodiscard]] std::vector<std::filesystem::path> backing_paths() const;
    [[nodiscard]] const EmbeddingModelLoadConfig& config() const { return cfg_; }

    [[nodiscard]] int32_t pool_size() const {
        return static_cast<int32_t>(ctx_pool_.size());
    }

private:
    explicit EmbeddingModel(EmbeddingModelLoadConfig cfg);
    [[nodiscard]] std::expected<void, ModelError> initialize();

    llama_context* acquire_ctx();
    void           release_ctx(llama_context* ctx);

    EmbeddingModelLoadConfig     cfg_;
    LlamaModelPtr                model_;
    std::vector<LlamaContextPtr> ctx_pool_;
    int32_t                      n_embd_{0};

    std::mutex                   pool_mu_;
    std::condition_variable      pool_cv_;
    std::vector<llama_context*>  free_ctxs_;
};

}  // namespace mass_worker
