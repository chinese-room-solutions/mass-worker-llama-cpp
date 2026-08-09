#pragma once

#include <condition_variable>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "ggml-backend.h"
#include "mass_worker/ctx_pool.hpp"
#include "mass_worker/llama_handles.hpp"

namespace mass_worker {

// Per-model load configuration translated from the proto's LlamaChatConfig.
// Path is the absolute filesystem path to the GGUF (already resolved by the
// fetch layer). All other fields use 0 / empty as "auto/use default".
struct ChatModelLoadConfig {
    std::filesystem::path path;
    std::filesystem::path mmproj_path;  // optional vision/audio projector

    // Context shape.
    int32_t context_size{0};  // 0 → 4096 default
    int32_t batch_size{0};    // 0 → auto (gpu: 2048, cpu: 512)

    // Placement.
    int32_t gpu_layers{0};      // 0 = auto-all-on-gpu, -1 = cpu-only, N>0 = N layers
    int32_t threads{0};         // 0 → hardware concurrency
    int32_t max_concurrent{0};  // 0 → grow-until-OOM with VRAM headroom; N>0 → explicit ceiling
    // VRAM headroom watermark, 1-100. Pool stops growing when the next
    // slot's projected allocation would push any allowed GPU's
    // used/total memory ratio past this percentage. Skipped when
    // max_concurrent > 0 (operator override wins).
    int32_t vram_headroom_pct{75};

    // Quality knobs.
    std::string flash_attn;     // "" auto / "enabled" / "disabled"
    std::string cache_type;     // "" / "f16" / "q8_0" / "q4_0"
    std::string chat_template;  // optional override
    bool thinking{false};       // reasoning mode

    // Operator-controlled device whitelist. Empty = use every backend
    // llama.cpp enumerates (default). When non-empty, becomes the
    // null-terminated mparams.devices array — llama.cpp restricts layer
    // placement to these backends.
    std::vector<ggml_backend_dev_t> allowed_devices;

    // Canonical IDs ("gpu:N" / "cpu:0") of every device this model will
    // actually occupy. Computed by WorkerService alongside allowed_devices
    // (the device pointers and string IDs share the same backend walk).
    // Surfaced unchanged on heartbeat so MASS can score split models by
    // min(GFLOPS) across the placement set.
    std::vector<std::string> device_ids;
};

// True when a model restricted to `allowed_devices` may touch a GPU: scans
// the whitelist when set, every enumerated backend otherwise. Drives the
// bool-only knobs a device array can't reach — mtmd's use_gpu and the
// batch-size default — which would otherwise grab a GPU the model itself
// is barred from.
[[nodiscard]] bool gpu_usable(const std::vector<ggml_backend_dev_t>& allowed_devices);

// Errors from the model layer.
enum class ModelErrorCode : std::uint8_t {
    InvalidConfig,
    LoadFailed,
    ContextCreateFailed,
    TokenizeFailed,
    DecodeFailed,
    TemplateFailed,
    ContextOverflow,
    Cancelled,
};
struct ModelError {
    ModelErrorCode code;
    std::string message;
};

// One piece of an image attached to a multimodal user message. data is the
// raw encoded bytes (PNG/JPEG/etc) — mtmd_helper_bitmap_init_from_buf
// auto-detects the format.
struct ImagePart {
    std::vector<unsigned char> data;
    std::string mime_type;
};

// Audio counterpart of ImagePart. Currently unused by callers but the
// format mirrors ImagePart so the chat path can fan in both kinds.
struct AudioPart {
    std::vector<unsigned char> data;
    std::string mime_type;
};

// One message in a chat completion request. role is
// "system"/"user"/"assistant"/"tool", matching the OpenAI convention.
// When images or audios are non-empty the message is multimodal: each
// entry expands to a media marker in the rendered prompt, in the order
// text → images → audios. The parent ChatModel must have been loaded
// with an mmproj for any non-text parts to be honoured.
struct ChatMessage {
    std::string role;
    std::string content;
    std::vector<ImagePart> images;
    std::vector<AudioPart> audios;
};

// Sampling parameters decoded from the wire SamplingParams. Each optional
// mirrors wire presence: nullopt = "the client didn't say" → the worker
// default documented per field; a value applies EXACTLY as sent, including
// zero (temperature 0 → greedy, seed 0 → seed 0 reproducibly).
struct SamplingParams {
    std::optional<int32_t> max_tokens;       // absent → bounded only by remaining context
    std::optional<float> temperature;        // absent → 1.0; present <= 0 → greedy
    std::optional<float> top_p;              // absent → 1.0 (disabled)
    std::optional<int32_t> top_k;            // absent → disabled; <= 0 is a no-op sampler
    std::optional<int32_t> seed;             // absent → random per request
    std::optional<float> min_p;              // absent → 0.0 (disabled)
    std::optional<float> repeat_penalty;     // absent → 1.0 (disabled)
    std::optional<float> frequency_penalty;  // absent → 0.0
    std::optional<float> presence_penalty;   // absent → 0.0
    std::vector<std::string> stop;
    bool enable_thinking{false};
};

// SamplerPlan is the pure decision half of sampler-chain construction,
// split from the llama_sampler_* calls so the contract is unit-testable.
// Decision table (OpenAI-compatible; an explicit zero is honoured, only
// absence falls back to a default):
//   temperature absent → 1.0; present <= 0 → greedy argmax. When greedy,
//     top_k/top_p/min_p are irrelevant and skipped; penalties still apply
//     — they reshape logits before argmax.
//   seed absent → LLAMA_DEFAULT_SEED (llama.cpp draws a random seed per
//     sampler chain, i.e. per request); present → that exact seed,
//     INCLUDING 0.
//   top_k applied ⇔ present and > 0 (k <= 0 is a no-op sampler).
//   top_p applied ⇔ present and < 1.0 (p >= 1 is a no-op; an explicit 0
//     is a real, maximally narrow filter).
//   min_p applied ⇔ present and > 0.
//   repeat_penalty absent → 1.0 (disabled); an explicit 0 is a real value.
//   penalties sampler is added ⇔ effective repeat_penalty != 1.0 OR
//     frequency_penalty != 0 OR presence_penalty != 0 (each independent).
struct SamplerPlan {
    bool greedy{false};
    float temperature{1.0f};            // effective; meaningless when greedy
    uint32_t seed{LLAMA_DEFAULT_SEED};  // effective dist-sampler seed
    float repeat_penalty{1.0f};         // effective
    float frequency_penalty{0.0f};      // effective
    float presence_penalty{0.0f};       // effective
    bool use_penalties{false};
    bool use_top_k{false};
    bool use_top_p{false};
    bool use_min_p{false};
};
[[nodiscard]] SamplerPlan plan_sampler(const SamplingParams& sp);

// IsCancelledFn is polled between generation steps (and while waiting for a
// pool slot / between batch items). Returning true exits the operation
// cleanly with ModelErrorCode::Cancelled so the caller can emit a terminal
// "cancelled" frame. Optional — pass nullptr for a non-cancellable run.
// Shared by ChatModel and EmbeddingModel.
using IsCancelledFn = std::function<bool()>;

// Result of a chat completion run.
struct CompletionResult {
    std::string text;               // assistant content (post-thinking-strip if enabled)
    std::string reasoning_content;  // thinking content (only if enable_thinking)
    int32_t prompt_tokens{0};
    int32_t completion_tokens{0};
    double tokens_per_second{0};
    std::string finish_reason;  // "stop" / "length" / "eos"
};

// ChatModel owns one loaded llama_model and a pool of N llama_contexts,
// one per concurrency slot, sized from cfg.max_concurrent. Inference
// borrows a free context for the duration of a request; the model
// weights are shared read-only across all slots.
//
// PoolSlot is the internal RAII guard around acquire_ctx/release_ctx,
// defined in the .cpp next to the pool implementation. Forward-friended
// so the guard can call the model's private pool primitives without
// exposing them publicly.
struct PoolSlot;

class ChatModel {
    friend struct PoolSlot;

public:
    [[nodiscard]] static std::expected<std::shared_ptr<ChatModel>, ModelError> load(
        ChatModelLoadConfig cfg);

    ~ChatModel();
    ChatModel(const ChatModel&) = delete;
    ChatModel& operator=(const ChatModel&) = delete;

    // Tokenize uses the model's vocab directly; thread-safe per llama.cpp.
    [[nodiscard]] std::expected<std::vector<int32_t>, ModelError> tokenize(
        const std::string& text, bool add_special = true) const;

    // chat_completion borrows one context from the pool, runs the request
    // end-to-end, returns it. Blocks if the pool is full.
    [[nodiscard]] std::expected<CompletionResult, ModelError> chat_completion(
        const std::vector<ChatMessage>& messages, const SamplingParams& sampling);

    // OnTokenFn is invoked once per generated token with the token's text
    // piece. Empty pieces (typical after BPE space normalization) and
    // pieces consumed by stop-sequence detection are still delivered for
    // streaming continuity. Return value is ignored.
    using OnTokenFn = std::function<void(std::string_view piece)>;

    // chat_completion_stream is the streaming variant: same final
    // result, but on_token fires once per generated piece during
    // inference. The returned CompletionResult carries usage and
    // finish_reason exactly as the non-streaming variant; text is the
    // full assistant message (post thinking-strip when enabled).
    //
    // is_cancelled is polled at the top of each sampling step (a cheap
    // function call between llama_sampler_sample iterations). The check
    // can't be inserted inside llama.cpp's own routines, so the worst-
    // case latency is one token-decode (~10-200ms depending on
    // hardware). Pass nullptr to opt out.
    [[nodiscard]] std::expected<CompletionResult, ModelError> chat_completion_stream(
        const std::vector<ChatMessage>& messages, const SamplingParams& sampling,
        const OnTokenFn& on_token, const IsCancelledFn& is_cancelled = nullptr);

    [[nodiscard]] std::vector<std::filesystem::path> backing_paths() const;
    [[nodiscard]] const ChatModelLoadConfig& config() const { return cfg_; }

    // Actual pool size after initialize() — may be smaller than the
    // requested max_concurrent if some slots failed to allocate.
    [[nodiscard]] int32_t pool_size() const { return static_cast<int32_t>(ctx_pool_.size()); }

    // Canonical device IDs this model occupies. Mirrors cfg.device_ids
    // populated at load time by WorkerService.
    [[nodiscard]] const std::vector<std::string>& device_ids() const { return cfg_.device_ids; }

    // bench_probe measures this model for a MASS model benchmark (see
    // probe_model_bench). Valid only on a pool-of-1 load with nothing in
    // flight — the benchmark's exclusivity contract is what guarantees
    // that, and the whole-device memory readings are wrong without it.
    [[nodiscard]] ModelBenchProbe bench_probe(const std::vector<DevMemSnap>& before_load);

private:
    explicit ChatModel(ChatModelLoadConfig cfg);
    [[nodiscard]] std::expected<void, ModelError> initialize();

    // Borrow / return a free context. Acquire blocks under cv_ until a slot
    // frees OR is_cancelled fires (polled — nothing external notifies the CV
    // on cancellation), returning nullptr on cancel so a shutdown never
    // deadlocks behind a saturated pool. RAII-wrapped in a Slot guard inside
    // the .cpp.
    llama_context* acquire_ctx(const IsCancelledFn& is_cancelled);
    void release_ctx(llama_context* ctx);

    ChatModelLoadConfig cfg_;
    LlamaModelPtr model_;
    // The exact parameters every pool context was built with, kept so a
    // benchmark can price one more slot identically.
    llama_context_params cparams_{};
    std::vector<LlamaContextPtr> ctx_pool_;  // owns the pool's contexts

    // Free-list of borrowable contexts plus its synchronisation. The list
    // shrinks under acquire and grows under release; cv_ wakes acquirers.
    std::mutex pool_mu_;
    std::condition_variable pool_cv_;
    std::vector<llama_context*> free_ctxs_;

    struct ChatTemplates;
    std::unique_ptr<ChatTemplates> templates_;

    struct Multimodal;
    std::unique_ptr<Multimodal> mm_;
};

}  // namespace mass_worker
