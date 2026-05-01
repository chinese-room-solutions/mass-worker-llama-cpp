#pragma once

#include <condition_variable>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

#include "ggml-backend.h"

#include "mass_worker/llama_handles.hpp"

namespace mass_worker {

// Per-model load configuration translated from the proto's LlamaChatConfig.
// Path is the absolute filesystem path to the GGUF (already resolved by the
// fetch layer). All other fields use 0 / empty as "auto/use default".
struct ChatModelLoadConfig {
    std::filesystem::path path;
    std::filesystem::path mmproj_path;        // optional vision/audio projector

    // Context shape.
    int32_t context_size{0};                  // 0 → 4096 default
    int32_t batch_size{0};                    // 0 → auto (gpu: 2048, cpu: 512)

    // Placement.
    int32_t gpu_layers{0};                    // 0 = auto-all-on-gpu, -1 = cpu-only, N>0 = N layers
    int32_t threads{0};                       // 0 → hardware concurrency
    int32_t max_concurrent{1};                // pool size: number of llama_contexts to allocate

    std::string main_gpu;                     // optional GPU index as string ("0", "1", ...)
    std::vector<float> tensor_split;          // optional tensor split ratios for multi-GPU

    // Quality knobs.
    std::string flash_attn;                   // "" auto / "enabled" / "disabled"
    std::string cache_type;                   // "" / "f16" / "q8_0" / "q4_0"
    std::string chat_template;                // optional override
    bool thinking{false};                     // reasoning mode

    // Operator-controlled device whitelist. Empty = use every backend
    // llama.cpp enumerates (default). When non-empty, becomes the
    // null-terminated mparams.devices array — llama.cpp restricts layer
    // placement to these backends.
    std::vector<ggml_backend_dev_t> allowed_devices;
};

// Errors from the model layer.
enum class ModelErrorCode {
    InvalidConfig,
    LoadFailed,
    ContextCreateFailed,
    TokenizeFailed,
    DecodeFailed,
    TemplateFailed,
    ContextOverflow,
};
struct ModelError {
    ModelErrorCode code;
    std::string    message;
};

// One piece of an image attached to a multimodal user message. data is the
// raw encoded bytes (PNG/JPEG/etc) — mtmd_helper_bitmap_init_from_buf
// auto-detects the format.
struct ImagePart {
    std::vector<unsigned char> data;
    std::string                mime_type;
};

// Audio counterpart of ImagePart. Currently unused by callers but the
// format mirrors ImagePart so the chat path can fan in both kinds.
struct AudioPart {
    std::vector<unsigned char> data;
    std::string                mime_type;
};

// One message in a chat completion request. role is "system" / "user" /
// "assistant" / "tool" matching the OpenAI convention. When `images` /
// `audios` are non-empty, the message is multimodal — each entry expands
// to a media marker in the rendered prompt, in the order text -> images
// -> audios. The parent ChatModel must have been loaded with an mmproj
// for any non-text parts to be honoured.
struct ChatMessage {
    std::string             role;
    std::string             content;
    std::vector<ImagePart>  images;
    std::vector<AudioPart>  audios;
};

// Sampling parameters from the wire SamplingParams. Zero values mean
// "use llama.cpp default" (matching the Go worker's behaviour); the only
// way to explicitly request a value is to set it non-zero.
struct SamplingParams {
    int32_t  max_tokens{0};      // 0 → context_size - prompt_len, capped reasonably
    float    temperature{0};
    float    top_p{0};
    int32_t  top_k{0};
    int32_t  seed{0};            // 0 → time-based (LLAMA_DEFAULT_SEED)
    float    min_p{0};
    float    repeat_penalty{0};  // 1.0 = disabled; 0 → use default 1.0
    float    frequency_penalty{0};
    float    presence_penalty{0};
    std::vector<std::string> stop;
    bool     enable_thinking{false};
};

// Result of a chat completion run.
struct CompletionResult {
    std::string text;                    // assistant content (post-thinking-strip if enabled)
    std::string reasoning_content;       // thinking content (only if enable_thinking)
    int32_t     prompt_tokens{0};
    int32_t     completion_tokens{0};
    double      tokens_per_second{0};
    std::string finish_reason;           // "stop" / "length" / "eos"
};

// ChatModel owns a loaded llama_model and a pool of N llama_contexts (one
// per concurrency slot, sized by cfg.max_concurrent). Inference borrows a
// free context for the duration of a request; the model weights are shared
// read-only across all slots.
// Internal RAII guard around acquire_ctx/release_ctx; defined in the .cpp
// next to the pool implementation. Forward-friended so the guard can call
// the model's private pool primitives without exposing them publicly.
struct PoolSlot;

class ChatModel {
    friend struct PoolSlot;
public:
    [[nodiscard]] static std::expected<std::shared_ptr<ChatModel>, ModelError>
    load(ChatModelLoadConfig cfg);

    ~ChatModel();
    ChatModel(const ChatModel&) = delete;
    ChatModel& operator=(const ChatModel&) = delete;

    // Tokenize uses the model's vocab directly; thread-safe per llama.cpp.
    [[nodiscard]] std::expected<std::vector<int32_t>, ModelError>
    tokenize(const std::string& text, bool add_special = true) const;

    // chat_completion borrows one context from the pool, runs the request
    // end-to-end, returns it. Blocks if the pool is full.
    [[nodiscard]] std::expected<CompletionResult, ModelError>
    chat_completion(const std::vector<ChatMessage>& messages,
                    const SamplingParams&           sampling);

    // OnTokenFn is invoked once per generated token with the token's text
    // piece. Empty pieces (typical after BPE space normalization) and
    // pieces consumed by stop-sequence detection are still delivered for
    // streaming continuity. Return value is ignored.
    using OnTokenFn = std::function<void(std::string_view piece)>;

    // chat_completion_stream is the streaming variant: same final result,
    // but on_token fires once per generated piece during inference. The
    // returned CompletionResult carries usage + finish_reason exactly as
    // the non-streaming variant — text is the full assistant message
    // (post thinking-strip when enabled).
    [[nodiscard]] std::expected<CompletionResult, ModelError>
    chat_completion_stream(const std::vector<ChatMessage>& messages,
                           const SamplingParams&           sampling,
                           OnTokenFn                       on_token);

    [[nodiscard]] std::vector<std::filesystem::path> backing_paths() const;
    [[nodiscard]] const ChatModelLoadConfig& config() const { return cfg_; }

    // Actual pool size after initialize() — may be smaller than the
    // requested max_concurrent if some slots failed to allocate.
    [[nodiscard]] int32_t pool_size() const {
        return static_cast<int32_t>(ctx_pool_.size());
    }

private:
    explicit ChatModel(ChatModelLoadConfig cfg);
    [[nodiscard]] std::expected<void, ModelError> initialize();

    // Borrow / return a free context. Acquire blocks under cv_; release
    // notifies one waiter. RAII-wrapped in a Slot guard inside the .cpp.
    llama_context* acquire_ctx();
    void           release_ctx(llama_context* ctx);

    ChatModelLoadConfig          cfg_;
    LlamaModelPtr                model_;
    std::vector<LlamaContextPtr> ctx_pool_;     // owns the pool's contexts

    // Free-list of borrowable contexts plus its synchronisation. The list
    // shrinks under acquire and grows under release; cv_ wakes acquirers.
    std::mutex                   pool_mu_;
    std::condition_variable      pool_cv_;
    std::vector<llama_context*>  free_ctxs_;

    struct ChatTemplates;
    std::unique_ptr<ChatTemplates> templates_;

    struct Multimodal;
    std::unique_ptr<Multimodal> mm_;
};

}  // namespace mass_worker
