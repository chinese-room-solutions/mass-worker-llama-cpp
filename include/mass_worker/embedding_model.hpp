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
#include "mass_worker/ctx_pool.hpp"
#include "mass_worker/llama_handles.hpp"

namespace mass_worker {

// Per-model embedding load configuration. Mirrors LlamaEmbeddingConfig.
struct EmbeddingModelLoadConfig {
    std::filesystem::path path;
    int32_t context_size{0};  // 0 → 4096
    int32_t gpu_layers{0};    // wire convention (0=auto-all, -1=cpu)
    int32_t threads{0};
    int32_t max_concurrent{0};  // 0 → grow to the headroom watermark; N>0 → explicit ceiling
    // VRAM headroom watermark, 1-100 (see CtxPoolHeadroom). Embed slots
    // are small individually, but an ungated pool on an iGPU grows into
    // the tens of slots and leaves decode-time allocations nothing.
    int32_t vram_headroom_pct{75};

    // Operator-controlled device whitelist (see ChatModelLoadConfig).
    std::vector<ggml_backend_dev_t> allowed_devices;

    // Canonical IDs ("gpu:N" / "cpu:0") of every device this model will
    // actually occupy (see ChatModelLoadConfig::device_ids).
    std::vector<std::string> device_ids;

    // On-disk calibration cache for the auto slot ceiling (see
    // calib_cache.hpp). Empty → measure on every load.
    std::filesystem::path calib_cache_file;
};

struct EmbedPoolSlot;

// EmbeddingModel owns a llama_model + a pool of N contexts configured with
// embeddings=true and mean-pooling. Concurrent embed() calls borrow distinct
// contexts; the model weights are shared.
class EmbeddingModel {
    friend struct EmbedPoolSlot;

public:
    [[nodiscard]] static std::expected<std::shared_ptr<EmbeddingModel>, ModelError> load(
        EmbeddingModelLoadConfig cfg);

    ~EmbeddingModel();
    EmbeddingModel(const EmbeddingModel&) = delete;
    EmbeddingModel& operator=(const EmbeddingModel&) = delete;

    // Embed one input string. Returns the per-sequence pooled vector.
    // is_cancelled is polled while waiting for a pool slot and before the
    // decode; on cancel returns ModelErrorCode::Cancelled. Safe to call
    // concurrently up to pool_size(); batch jobs fan out over this via
    // run_batch_items in the worker service.
    [[nodiscard]] std::expected<std::vector<float>, ModelError> embed(
        const std::string& text, const IsCancelledFn& is_cancelled = nullptr);

    [[nodiscard]] std::vector<std::filesystem::path> backing_paths() const;
    [[nodiscard]] const EmbeddingModelLoadConfig& config() const { return cfg_; }

    [[nodiscard]] int32_t pool_size() const { return static_cast<int32_t>(ctx_pool_.size()); }

    // Canonical device IDs this model occupies. Mirrors cfg.device_ids
    // populated at load time by WorkerService.
    [[nodiscard]] const std::vector<std::string>& device_ids() const { return cfg_.device_ids; }

    // See ChatModel::bench_probe.
    [[nodiscard]] ModelBenchProbe bench_probe(const std::vector<DevMemSnap>& before_load);

private:
    explicit EmbeddingModel(EmbeddingModelLoadConfig cfg);
    [[nodiscard]] std::expected<void, ModelError> initialize();

    // Same contract as ChatModel::acquire_ctx: blocks until a slot frees or
    // is_cancelled fires (polled), returning nullptr on cancel.
    llama_context* acquire_ctx(const IsCancelledFn& is_cancelled);
    void release_ctx(llama_context* ctx);

    EmbeddingModelLoadConfig cfg_;
    LlamaModelPtr model_;
    // See ChatModel::cparams_.
    llama_context_params cparams_{};
    std::vector<LlamaContextPtr> ctx_pool_;
    int32_t n_embd_{0};

    std::mutex pool_mu_;
    std::condition_variable pool_cv_;
    std::vector<llama_context*> free_ctxs_;
};

}  // namespace mass_worker
