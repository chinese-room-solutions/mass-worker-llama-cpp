#include "mass_worker/embedding_model.hpp"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <thread>
#include <utility>

#include "ggml-backend.h"
#include "llama.h"
#include "mass_worker/calib_cache.hpp"
#include "mass_worker/ctx_pool.hpp"
#include "mass_worker/llama_backend.hpp"

namespace mass_worker {

namespace fs = std::filesystem;

namespace {

int32_t to_llama_gpu_layers(int32_t wire) {
    switch (wire) {
        case 0:
            return -1;
        case -1:
            return 0;
        default:
            return wire;
    }
}

// L2-normalize a vector in place. Standard convention for embeddings —
// dot product becomes cosine similarity.
void l2_normalize(std::vector<float>& v) {
    double sum = 0;
    for (float x : v) sum += static_cast<double>(x) * x;
    if (sum <= 0) return;
    const auto inv = static_cast<float>(1.0 / std::sqrt(sum));
    for (float& x : v) x *= inv;
}

}  // namespace

EmbeddingModel::EmbeddingModel(EmbeddingModelLoadConfig cfg) : cfg_(std::move(cfg)) {}

EmbeddingModel::~EmbeddingModel() {
    if (model_) spdlog::info("embedding model unloaded: {}", cfg_.path.string());
}

std::expected<std::shared_ptr<EmbeddingModel>, ModelError> EmbeddingModel::load(
    EmbeddingModelLoadConfig cfg) {
    if (cfg.path.empty()) {
        return std::unexpected(
            ModelError{ModelErrorCode::InvalidConfig, "EmbeddingModelLoadConfig.path is empty"});
    }
    init_llama_backend_once();
    auto m = std::shared_ptr<EmbeddingModel>(new EmbeddingModel(std::move(cfg)));
    if (auto r = m->initialize(); !r) return std::unexpected(r.error());
    return m;
}

std::expected<void, ModelError> EmbeddingModel::initialize() {
    int32_t context_size = cfg_.context_size > 0 ? cfg_.context_size : 4096;
    int32_t threads = cfg_.threads > 0
                          ? cfg_.threads
                          : static_cast<int32_t>(std::max(1u, std::thread::hardware_concurrency()));

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = to_llama_gpu_layers(cfg_.gpu_layers);
    mparams.use_mmap = true;

    // Operator-controlled device whitelist (see chat_model.cpp).
    std::vector<ggml_backend_dev_t> allowed_with_sentinel;
    if (!cfg_.allowed_devices.empty()) {
        allowed_with_sentinel = cfg_.allowed_devices;
        allowed_with_sentinel.push_back(nullptr);
        mparams.devices = allowed_with_sentinel.data();
    }

    spdlog::info(
        "loading embedding model path={} gpu_layers={} ctx={} allowed_devices={}",
        cfg_.path.string(), mparams.n_gpu_layers, context_size,
        cfg_.allowed_devices.empty() ? "<all>" : std::to_string(cfg_.allowed_devices.size()));

    LlamaModelPtr model(llama_model_load_from_file(cfg_.path.string().c_str(), mparams));
    if (!model) {
        return std::unexpected(
            ModelError{ModelErrorCode::LoadFailed,
                       "llama_model_load_from_file failed for " + cfg_.path.string()});
    }
    model_ = std::move(model);

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = static_cast<uint32_t>(context_size);
    cparams.n_batch = static_cast<uint32_t>(context_size);  // batch == ctx for embeddings
    cparams.n_threads = threads;
    cparams.n_threads_batch = threads;
    cparams.embeddings = true;
    cparams.pooling_type = LLAMA_POOLING_TYPE_MEAN;

    auto pool =
        grow_ctx_pool(model_.get(), cparams,
                      {.max_concurrent = cfg_.max_concurrent,
                       .vram_headroom_pct = cfg_.vram_headroom_pct,
                       .allowed_devices = cfg_.allowed_devices,
                       .calib_cache_file = cfg_.calib_cache_file,
                       .calib_key = cfg_.calib_cache_file.empty()
                                        ? std::string{}
                                        : calib_cache_key(cfg_.path, cparams, cfg_.device_ids)});
    if (!pool) {
        model_.reset();
        return std::unexpected(
            ModelError{ModelErrorCode::ContextCreateFailed, std::move(pool).error()});
    }
    ctx_pool_ = std::move(*pool);
    for (auto& ctx : ctx_pool_) free_ctxs_.push_back(ctx.get());
    const auto pool_size = static_cast<int32_t>(ctx_pool_.size());
    n_embd_ = llama_model_n_embd(model_.get());

    spdlog::info("embedding model loaded: {} (n_embd={}, n_ctx={}, slots={})", cfg_.path.string(),
                 n_embd_, llama_n_ctx(ctx_pool_.front().get()), pool_size);
    return {};
}

llama_context* EmbeddingModel::acquire_ctx(const IsCancelledFn& is_cancelled) {
    std::unique_lock lk(pool_mu_);
    // Poll-wake: cancellation never notifies pool_cv_ (see ChatModel).
    while (free_ctxs_.empty()) {
        if (is_cancelled && is_cancelled()) return nullptr;
        pool_cv_.wait_for(lk, std::chrono::milliseconds(50),
                          [this] { return !free_ctxs_.empty(); });
    }
    llama_context* c = free_ctxs_.back();
    free_ctxs_.pop_back();
    return c;
}

void EmbeddingModel::release_ctx(llama_context* ctx) {
    {
        std::scoped_lock lk(pool_mu_);
        free_ctxs_.push_back(ctx);
    }
    pool_cv_.notify_one();
}

struct EmbedPoolSlot {
    EmbeddingModel* owner;
    llama_context* ctx;
    EmbedPoolSlot(EmbeddingModel* o, llama_context* c) : owner(o), ctx(c) {}
    ~EmbedPoolSlot() {
        if (owner && ctx) owner->release_ctx(ctx);
    }
    EmbedPoolSlot(const EmbedPoolSlot&) = delete;
    EmbedPoolSlot& operator=(const EmbedPoolSlot&) = delete;
};

std::expected<std::vector<float>, ModelError> EmbeddingModel::embed(
    const std::string& text, const IsCancelledFn& is_cancelled) {
    if (!model_ || ctx_pool_.empty()) {
        return std::unexpected(ModelError{ModelErrorCode::InvalidConfig, "model not loaded"});
    }
    const llama_vocab* vocab = llama_model_get_vocab(model_.get());

    int32_t n_needed = llama_tokenize(vocab, text.c_str(), static_cast<int32_t>(text.size()),
                                      nullptr, 0, /*add_special=*/true, /*parse_special=*/true);
    if (n_needed < 0) n_needed = -n_needed;
    std::vector<llama_token> tokens(static_cast<std::size_t>(n_needed));
    const int32_t produced =
        llama_tokenize(vocab, text.c_str(), static_cast<int32_t>(text.size()), tokens.data(),
                       n_needed, /*add_special=*/true, /*parse_special=*/true);
    if (produced < 0) {
        return std::unexpected(
            ModelError{ModelErrorCode::TokenizeFailed, "tokenize: " + std::to_string(produced)});
    }
    tokens.resize(static_cast<std::size_t>(produced));

    EmbedPoolSlot slot(this, acquire_ctx(is_cancelled));
    if (!slot.ctx) {
        return std::unexpected(
            ModelError{ModelErrorCode::Cancelled, "cancelled while waiting for a context slot"});
    }
    llama_context* ctx = slot.ctx;

    // A cancel that arrived while we held no slot yet — abort before the
    // decode rather than burning a pass the operator abandoned.
    if (is_cancelled && is_cancelled()) {
        return std::unexpected(ModelError{ModelErrorCode::Cancelled, "cancelled by operator"});
    }

    if (std::cmp_greater(produced, llama_n_ctx(ctx))) {
        return std::unexpected(
            ModelError{ModelErrorCode::ContextOverflow,
                       "input tokens (" + std::to_string(produced) + ") exceed n_ctx"});
    }

    llama_memory_clear(llama_get_memory(ctx), /*data=*/true);
    llama_batch batch = llama_batch_get_one(tokens.data(), produced);
    if (llama_decode(ctx, batch) != 0) {
        return std::unexpected(ModelError{ModelErrorCode::DecodeFailed, "embedding decode failed"});
    }

    const float* emb = llama_get_embeddings_seq(ctx, 0);
    if (!emb) emb = llama_get_embeddings_ith(ctx, -1);
    if (!emb) {
        return std::unexpected(ModelError{ModelErrorCode::DecodeFailed, "no embeddings produced"});
    }

    std::vector<float> out(emb, emb + n_embd_);
    l2_normalize(out);
    return out;
}

std::vector<fs::path> EmbeddingModel::backing_paths() const {
    return {cfg_.path};
}

}  // namespace mass_worker
