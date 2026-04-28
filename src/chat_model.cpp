#include "mass_worker/chat_model.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <limits>
#include <thread>
#include <vector>

#include <spdlog/spdlog.h>

#include "mass_worker/llama_backend.hpp"

// llama.cpp's `llama-common` target exports `common/` on its include path,
// so headers inside it are reached without the `common/` prefix.
#include "chat.h"
#include "common.h"
#include "ggml-backend.h"
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"

namespace mass_worker {

struct ChatModel::ChatTemplates {
    common_chat_templates_ptr ptr;
};

struct ChatModel::Multimodal {
    mtmd::context_ptr ctx;
};

namespace fs = std::filesystem;

namespace {

bool any_gpu_backend_available() {
    const std::size_t n = ggml_backend_dev_count();
    for (std::size_t i = 0; i < n; ++i) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        if (!d) continue;
        const auto t = ggml_backend_dev_type(d);
        if (t == GGML_BACKEND_DEVICE_TYPE_GPU || t == GGML_BACKEND_DEVICE_TYPE_IGPU) {
            return true;
        }
    }
    return false;
}

// Translate the wire convention of gpu_layers (0=auto-all, -1=cpu-only,
// N>0=N layers) into llama.cpp's (-1=auto-all, 0=cpu-only, N>0=N layers).
// Same shape as the Go worker.
int32_t to_llama_gpu_layers(int32_t wire) {
    switch (wire) {
        case 0:  return -1;  // auto: all on GPU
        case -1: return 0;   // CPU only
        default: return wire;
    }
}

}  // namespace

std::expected<std::shared_ptr<ChatModel>, ModelError>
ChatModel::load(ChatModelLoadConfig cfg) {
    if (cfg.path.empty()) {
        return std::unexpected(ModelError{ModelErrorCode::InvalidConfig,
                                          "ChatModelLoadConfig.path is empty"});
    }
    init_llama_backend_once();

    auto m = std::shared_ptr<ChatModel>(new ChatModel(std::move(cfg)));
    if (auto r = m->initialize(); !r) {
        return std::unexpected(r.error());
    }
    return m;
}

ChatModel::ChatModel(ChatModelLoadConfig cfg) : cfg_(std::move(cfg)) {}

ChatModel::~ChatModel() {
    if (model_) {
        spdlog::info("model unloaded: {}", cfg_.path.string());
    }
}

std::expected<void, ModelError> ChatModel::initialize() {
    // ---- Defaults ----
    const bool gpu_present = any_gpu_backend_available();

    int32_t context_size = cfg_.context_size > 0 ? cfg_.context_size : 4096;
    int32_t batch_size   = cfg_.batch_size   > 0 ? cfg_.batch_size
                                                 : (gpu_present ? 2048 : 512);
    int32_t threads      = cfg_.threads      > 0
                              ? cfg_.threads
                              : static_cast<int32_t>(std::max(1u, std::thread::hardware_concurrency()));

    // ---- Model params ----
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = to_llama_gpu_layers(cfg_.gpu_layers);
    mparams.use_mmap     = true;
    mparams.use_mlock    = false;  // mlock pinning is platform-fragile; opt-in only

    // Multi-GPU placement. main_gpu accepts an integer GPU index as a
    // string ("0", "1", ...). Empty → llama.cpp default (0).
    if (!cfg_.main_gpu.empty()) {
        try {
            mparams.main_gpu = std::stoi(cfg_.main_gpu);
        } catch (const std::exception&) {
            return std::unexpected(ModelError{ModelErrorCode::InvalidConfig,
                "main_gpu not parseable as int: " + cfg_.main_gpu});
        }
    }
    if (!cfg_.tensor_split.empty()) {
        mparams.tensor_split = cfg_.tensor_split.data();
    }

    spdlog::info("loading model path={} gpu_layers={} ctx={} batch={} threads={}",
                 cfg_.path.string(), mparams.n_gpu_layers, context_size,
                 batch_size, threads);

    LlamaModelPtr model(llama_model_load_from_file(cfg_.path.string().c_str(), mparams));
    if (!model) {
        return std::unexpected(ModelError{ModelErrorCode::LoadFailed,
            "llama_model_load_from_file failed for " + cfg_.path.string()});
    }
    model_ = std::move(model);

    // ---- Context params ----
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx        = static_cast<uint32_t>(context_size);
    cparams.n_batch      = static_cast<uint32_t>(batch_size);
    cparams.n_threads    = threads;
    cparams.n_threads_batch = threads;

    // Flash attention. Quantized KV cache types require flash_attn = on,
    // and FA itself requires GPU; respect explicit overrides, otherwise
    // turn FA on iff a GPU is present.
    if (cfg_.flash_attn == "enabled") {
        cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    } else if (cfg_.flash_attn == "disabled") {
        cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    } else {
        cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;
    }

    // KV cache type. f16 is safe; quantized variants need FA on.
    auto cache_type = cfg_.cache_type;
    if (cache_type.empty() && cparams.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_DISABLED) {
        cache_type = "f16";  // fallback: quantized KV without FA = corrupted output
    }
    if (cache_type == "f16") {
        cparams.type_k = GGML_TYPE_F16;
        cparams.type_v = GGML_TYPE_F16;
    } else if (cache_type == "q8_0") {
        cparams.type_k = GGML_TYPE_Q8_0;
        cparams.type_v = GGML_TYPE_Q8_0;
    } else if (cache_type == "q4_0") {
        cparams.type_k = GGML_TYPE_Q4_0;
        cparams.type_v = GGML_TYPE_Q4_0;
    }
    // else: leave llama.cpp defaults

    // Initialise the chat templates handle from the model. Stays alive for
    // the model's lifetime — the per-request format/parser path consults it.
    templates_ = std::make_unique<ChatTemplates>();
    templates_->ptr = common_chat_templates_init(model_.get(), cfg_.chat_template);
    if (!templates_->ptr) {
        templates_.reset();
        model_.reset();
        return std::unexpected(ModelError{ModelErrorCode::TemplateFailed,
            "common_chat_templates_init failed"});
    }

    // Multimodal projector loads BEFORE the context pool. Two reasons:
    // (1) mmproj VRAM is a fixed cost — the pool size is what flexes when
    // VRAM is tight, so the fixed cost has to be paid first;
    // (2) mtmd_init_from_file is treated as fatal by the upstream library
    // when its allocations fail, so we can't recover after an OOM by
    // freeing pool contexts. Loading it first guarantees enough headroom.
    if (!cfg_.mmproj_path.empty()) {
        mtmd_context_params mp = mtmd_context_params_default();
        mp.use_gpu         = gpu_present;
        mp.print_timings   = false;
        mp.n_threads       = threads;
        mp.flash_attn_type = cparams.flash_attn_type;

        mm_ = std::make_unique<Multimodal>();
        mm_->ctx.reset(mtmd_init_from_file(
            cfg_.mmproj_path.string().c_str(), model_.get(), mp));
        if (!mm_->ctx) {
            mm_.reset();
            templates_.reset();
            model_.reset();
            return std::unexpected(ModelError{ModelErrorCode::LoadFailed,
                "mtmd_init_from_file failed for " + cfg_.mmproj_path.string()});
        }
        spdlog::info("mmproj loaded: {} (vision={} audio={})",
                     cfg_.mmproj_path.string(),
                     mtmd_support_vision(mm_->ctx.get()),
                     mtmd_support_audio(mm_->ctx.get()));
    }

    // Grow the context pool one slot at a time until allocation fails (the
    // GPU is the truth source for "how many fit"). When the user pinned
    // max_concurrent we treat it as a ceiling, not a target — useful for
    // latency-sensitive deployments that don't want a single model owning
    // the entire device.
    const int32_t ceiling = cfg_.max_concurrent > 0 ? cfg_.max_concurrent
                                                    : std::numeric_limits<int32_t>::max();
    while (static_cast<int32_t>(ctx_pool_.size()) < ceiling) {
        LlamaContextPtr ctx(llama_init_from_model(model_.get(), cparams));
        if (!ctx) {
            if (ctx_pool_.empty()) {
                mm_.reset();
                templates_.reset();
                model_.reset();
                return std::unexpected(ModelError{ModelErrorCode::ContextCreateFailed,
                    "llama_init_from_model failed at slot 0 — not enough VRAM "
                    "for even one context"});
            }
            break;
        }
        free_ctxs_.push_back(ctx.get());
        ctx_pool_.push_back(std::move(ctx));
    }

    // Drop the last allocated slot to leave VRAM headroom. Two reasons:
    // (1) Saturating VRAM forces Vulkan to spill into shared host memory
    //     over PCIe — measured ~3× per-request slowdown on a 4B Q4_K_M
    //     model when going from 1 active slot to 7 in a full GPU.
    // (2) Multimodal models (mmproj loaded) build per-input compute
    //     graphs whose size depends on image/audio dimensions; without
    //     headroom, mtmd_helper_eval_chunks OOMs mid-request — fatal.
    // Skipped when the user explicitly pinned max_concurrent: that's an
    // override we honour as-is.
    if (cfg_.max_concurrent <= 0 && ctx_pool_.size() > 1) {
        spdlog::info("dropping last pool slot for VRAM headroom ({}→{} slots)",
                     ctx_pool_.size(), ctx_pool_.size() - 1);
        ctx_pool_.pop_back();
        free_ctxs_.pop_back();
    }
    const int32_t pool_size = static_cast<int32_t>(ctx_pool_.size());

    spdlog::info("model loaded: {} (n_ctx={} slots={} ceiling={})",
                 cfg_.path.string(),
                 llama_n_ctx(ctx_pool_.front().get()),
                 pool_size,
                 cfg_.max_concurrent > 0 ? std::to_string(cfg_.max_concurrent)
                                         : std::string("auto"));
    return {};
}

llama_context* ChatModel::acquire_ctx() {
    std::unique_lock lk(pool_mu_);
    pool_cv_.wait(lk, [this] { return !free_ctxs_.empty(); });
    llama_context* ctx = free_ctxs_.back();
    free_ctxs_.pop_back();
    spdlog::debug("pool acquire: ctx={} free_after={}",
                  static_cast<const void*>(ctx), free_ctxs_.size());
    return ctx;
}

void ChatModel::release_ctx(llama_context* ctx) {
    {
        std::lock_guard lk(pool_mu_);
        free_ctxs_.push_back(ctx);
        spdlog::debug("pool release: ctx={} free_after={}",
                      static_cast<const void*>(ctx), free_ctxs_.size());
    }
    pool_cv_.notify_one();
}

std::expected<std::vector<int32_t>, ModelError>
ChatModel::tokenize(const std::string& text, bool add_special) const {
    if (!model_) {
        return std::unexpected(ModelError{ModelErrorCode::TokenizeFailed,
                                          "model not loaded"});
    }
    // llama_tokenize against a vocab pointer is documented thread-safe
    // (read-only access to the model's vocabulary). No pool borrow needed.
    const llama_vocab* vocab = llama_model_get_vocab(model_.get());
    if (!vocab) {
        return std::unexpected(ModelError{ModelErrorCode::TokenizeFailed,
                                          "model has no vocab"});
    }

    // First call probes the required size (returns negative count = needed
    // buffer length). Allocate that, then call again to fill.
    const int32_t txt_len = static_cast<int32_t>(text.size());
    int32_t n_needed = llama_tokenize(vocab, text.c_str(), txt_len,
                                      nullptr, 0, add_special, /*parse_special=*/true);
    if (n_needed == INT32_MIN) {
        return std::unexpected(ModelError{ModelErrorCode::TokenizeFailed,
                                          "tokenize size overflow"});
    }
    if (n_needed < 0) n_needed = -n_needed;

    std::vector<int32_t> out(static_cast<std::size_t>(n_needed));
    static_assert(sizeof(int32_t) == sizeof(llama_token));
    const int32_t produced =
        llama_tokenize(vocab, text.c_str(), txt_len,
                       reinterpret_cast<llama_token*>(out.data()),
                       n_needed, add_special, /*parse_special=*/true);
    if (produced < 0) {
        return std::unexpected(ModelError{ModelErrorCode::TokenizeFailed,
            "llama_tokenize returned " + std::to_string(produced)});
    }
    out.resize(static_cast<std::size_t>(produced));
    return out;
}

std::vector<fs::path> ChatModel::backing_paths() const {
    std::vector<fs::path> out;
    out.push_back(cfg_.path);
    if (!cfg_.mmproj_path.empty()) out.push_back(cfg_.mmproj_path);
    return out;
}

// --- chat completion ---

namespace {

// Render the chat template via the `common` library and return both the
// prompt and the format shape downstream parsers need. For multimodal
// messages we feed `content_parts` with interleaved text + media markers;
// the marker text becomes the model's `<__media__>` sentinel that
// mtmd_tokenize uses to split chunks.
std::expected<common_chat_params, ModelError>
apply_chat_template(common_chat_templates*          templates,
                    const std::vector<ChatMessage>& messages,
                    bool                            enable_thinking) {
    const std::string media_marker = mtmd_default_marker();

    common_chat_templates_inputs inputs;
    inputs.messages.reserve(messages.size());
    for (const auto& m : messages) {
        common_chat_msg cm;
        cm.role = m.role;

        const std::size_t n_media = m.images.size() + m.audios.size();
        if (n_media == 0) {
            cm.content = m.content;
        } else {
            if (!m.content.empty()) {
                cm.content_parts.push_back({/*type=*/"text", /*text=*/m.content});
            }
            for (std::size_t i = 0; i < n_media; ++i) {
                cm.content_parts.push_back({"media_marker", media_marker});
            }
        }
        inputs.messages.push_back(std::move(cm));
    }
    inputs.add_generation_prompt = true;
    inputs.use_jinja             = true;
    inputs.enable_thinking       = enable_thinking;
    inputs.reasoning_format      = COMMON_REASONING_FORMAT_AUTO;

    try {
        return common_chat_templates_apply(templates, inputs);
    } catch (const std::exception& e) {
        return std::unexpected(ModelError{ModelErrorCode::TemplateFailed,
            std::string("common_chat_templates_apply: ") + e.what()});
    }
}


// Convert a single token ID to its text piece (UTF-8 string). Returns
// empty string if the token has no printable form (e.g. some special
// tokens). `special=false` matches Go worker behaviour: control tokens
// don't appear in the output text.
std::string token_to_piece(const llama_vocab* vocab, llama_token id) {
    char buf[256];
    int32_t n = llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, /*special=*/false);
    if (n < 0) {
        // Buffer too small — extremely rare; reallocate and retry.
        std::vector<char> big(static_cast<std::size_t>(-n));
        n = llama_token_to_piece(vocab, id, big.data(),
                                 static_cast<int32_t>(big.size()), 0, false);
        if (n < 0) return {};
        return std::string(big.data(), static_cast<std::size_t>(n));
    }
    return std::string(buf, static_cast<std::size_t>(n));
}

// Build a sampler chain matching the requested params. Returns a chain
// that ends with either a greedy or distribution sampler so it terminates
// in a concrete token choice.
LlamaSamplerPtr build_sampler(const llama_vocab*    vocab,
                              const SamplingParams& sp,
                              int32_t               n_ctx_train) {
    llama_sampler_chain_params sp_params = llama_sampler_chain_default_params();
    LlamaSamplerPtr chain(llama_sampler_chain_init(sp_params));

    // Penalties first (cheaper if applied before nucleus filters in some
    // workloads — order matches llama-cli).
    if (sp.repeat_penalty != 0 && sp.repeat_penalty != 1.0f) {
        llama_sampler_chain_add(chain.get(),
            llama_sampler_init_penalties(/*last_n=*/64,
                                         sp.repeat_penalty,
                                         sp.frequency_penalty,
                                         sp.presence_penalty));
    }
    if (sp.top_k > 0) {
        llama_sampler_chain_add(chain.get(), llama_sampler_init_top_k(sp.top_k));
    }
    if (sp.top_p > 0 && sp.top_p < 1.0f) {
        llama_sampler_chain_add(chain.get(), llama_sampler_init_top_p(sp.top_p, /*min_keep=*/1));
    }
    if (sp.min_p > 0) {
        llama_sampler_chain_add(chain.get(), llama_sampler_init_min_p(sp.min_p, /*min_keep=*/1));
    }
    if (sp.temperature > 0) {
        llama_sampler_chain_add(chain.get(), llama_sampler_init_temp(sp.temperature));
    }

    // Terminator: if all of {temp, top_p, top_k, min_p} are zero, do
    // greedy; otherwise distribution sampling with the configured seed.
    const bool no_filters = sp.temperature == 0 && sp.top_p == 0 &&
                            sp.top_k == 0 && sp.min_p == 0;
    if (no_filters) {
        llama_sampler_chain_add(chain.get(), llama_sampler_init_greedy());
    } else {
        const uint32_t seed = sp.seed != 0
                                ? static_cast<uint32_t>(sp.seed)
                                : LLAMA_DEFAULT_SEED;
        llama_sampler_chain_add(chain.get(), llama_sampler_init_dist(seed));
    }
    (void)vocab;
    (void)n_ctx_train;
    return chain;
}

// Returns true if `text` ends with any of the configured stop sequences.
// On match, removes the matched suffix from `text` and returns true.
bool consume_stop_sequence(std::string& text,
                           const std::vector<std::string>& stops) {
    for (const auto& s : stops) {
        if (s.empty()) continue;
        if (text.size() >= s.size() &&
            text.compare(text.size() - s.size(), s.size(), s) == 0) {
            text.resize(text.size() - s.size());
            return true;
        }
    }
    return false;
}

}  // namespace

// RAII guard around an acquire/release pair on the ChatModel context pool.
// Lives only on the chat_completion stack; release happens on scope exit
// even when the function returns std::unexpected mid-flight.
struct PoolSlot {
    ChatModel*     owner;
    llama_context* ctx;
    PoolSlot(ChatModel* o, llama_context* c) : owner(o), ctx(c) {}
    ~PoolSlot() { if (owner && ctx) owner->release_ctx(ctx); }
    PoolSlot(const PoolSlot&) = delete;
    PoolSlot& operator=(const PoolSlot&) = delete;
};

std::expected<CompletionResult, ModelError>
ChatModel::chat_completion(const std::vector<ChatMessage>& messages,
                           const SamplingParams&           sampling) {
    if (!model_ || ctx_pool_.empty() || !templates_) {
        return std::unexpected(ModelError{ModelErrorCode::InvalidConfig,
                                          "model not loaded"});
    }
    const llama_vocab* vocab = llama_model_get_vocab(model_.get());

    bool has_media = false;
    for (const auto& m : messages) {
        if (!m.images.empty() || !m.audios.empty()) { has_media = true; break; }
    }
    if (has_media && !mm_) {
        return std::unexpected(ModelError{ModelErrorCode::InvalidConfig,
            "multimodal request but model loaded without mmproj"});
    }

    // 1. Apply chat template. The wire-level `enable_thinking` flag controls
    //    BOTH whether the model is *asked* to think (template gate) and
    //    whether we *parse out* the thinking afterward. Defaults to off
    //    because the playground / most clients don't set it.
    const bool thinking = sampling.enable_thinking;
    auto chat_params = apply_chat_template(templates_->ptr.get(), messages, thinking);
    if (!chat_params) return std::unexpected(chat_params.error());

    const std::string& prompt = chat_params->prompt;
    spdlog::debug("chat: thinking={} format={} prompt_len={} multimodal={}",
                  thinking,
                  common_chat_format_name(chat_params->format),
                  prompt.size(), has_media);
    if (spdlog::should_log(spdlog::level::debug)) {
        spdlog::debug("chat: prompt tail (last 200 chars)=<<<{}>>>",
                      prompt.size() > 200 ? prompt.substr(prompt.size() - 200) : prompt);
        for (std::size_t i = 0; i < chat_params->additional_stops.size(); ++i) {
            spdlog::debug("chat: stop[{}]=<<<{}>>>", i, chat_params->additional_stops[i]);
        }
    }

    // 2. Borrow a context from the pool. Blocks if every slot is busy;
    //    released back to the pool when `slot` goes out of scope.
    PoolSlot       slot(this, acquire_ctx());
    llama_context* ctx     = slot.ctx;
    const int32_t  n_ctx   = static_cast<int32_t>(llama_n_ctx(ctx));
    const int32_t  n_batch = static_cast<int32_t>(llama_n_batch(ctx));
    spdlog::debug("chat: prefill begin n_ctx={} n_batch={} multimodal={}",
                  n_ctx, n_batch, has_media);

    // 3. Reset this slot's KV cache so the request is independent of
    //    whatever the previous borrower of this context did.
    llama_memory_clear(llama_get_memory(ctx), /*data=*/true);

    // 4. Prefill. Multimodal path goes through mtmd which interleaves
    //    text decode with image encode + decode; text-only path uses the
    //    raw tokenize+decode shortcut. Both end with the KV cache holding
    //    the full prompt.
    int32_t produced = 0;  // total tokens consumed by the prompt (for usage)

    if (has_media) {
        mtmd_input_text mt;
        mt.text          = prompt.c_str();
        mt.add_special   = false;
        mt.parse_special = true;

        // Decode each image/audio buffer into an mtmd_bitmap. Order must
        // match the markers emitted by apply_chat_template (text → images
        // → audios per message). bitmap is move-only so the vector is
        // built in place and never copied.
        mtmd::bitmaps bitmaps;
        for (const auto& m : messages) {
            for (const auto& img : m.images) {
                mtmd::bitmap bmp(mtmd_helper_bitmap_init_from_buf(
                    mm_->ctx.get(), img.data.data(), img.data.size()));
                if (!bmp.ptr) {
                    return std::unexpected(ModelError{ModelErrorCode::InvalidConfig,
                        "decoding image bytes failed (mime=" + img.mime_type + ")"});
                }
                bitmaps.entries.push_back(std::move(bmp));
            }
            for (const auto& aud : m.audios) {
                mtmd::bitmap bmp(mtmd_helper_bitmap_init_from_buf(
                    mm_->ctx.get(), aud.data.data(), aud.data.size()));
                if (!bmp.ptr) {
                    return std::unexpected(ModelError{ModelErrorCode::InvalidConfig,
                        "decoding audio bytes failed (mime=" + aud.mime_type + ")"});
                }
                bitmaps.entries.push_back(std::move(bmp));
            }
        }

        auto bmp_ptrs = bitmaps.c_ptr();
        spdlog::debug("chat: mtmd_tokenize begin bitmaps={} prompt_len={}",
                      bmp_ptrs.size(), prompt.size());
        mtmd::input_chunks chunks(mtmd_input_chunks_init());
        const int32_t tok_rc = mtmd_tokenize(mm_->ctx.get(), chunks.ptr.get(),
                                             &mt, bmp_ptrs.data(), bmp_ptrs.size());
        if (tok_rc != 0) {
            return std::unexpected(ModelError{ModelErrorCode::TokenizeFailed,
                "mtmd_tokenize failed: rc=" + std::to_string(tok_rc) +
                " (1=marker count mismatch, 2=image preprocess error)"});
        }

        const std::size_t n_tokens = mtmd_helper_get_n_tokens(chunks.ptr.get());
        if (static_cast<int32_t>(n_tokens) >= n_ctx) {
            return std::unexpected(ModelError{ModelErrorCode::ContextOverflow,
                "prompt tokens (" + std::to_string(n_tokens) +
                ") exceed context (" + std::to_string(n_ctx) + ")"});
        }

        spdlog::debug("chat: mtmd_helper_eval_chunks begin n_tokens={} n_batch={}",
                      n_tokens, n_batch);
        llama_pos new_n_past = 0;
        const int32_t eval_rc = mtmd_helper_eval_chunks(
            mm_->ctx.get(), ctx, chunks.ptr.get(),
            /*n_past=*/0, /*seq_id=*/0, n_batch,
            /*logits_last=*/true, &new_n_past);
        spdlog::debug("chat: mtmd_helper_eval_chunks end rc={} new_n_past={}",
                      eval_rc, new_n_past);
        if (eval_rc != 0) {
            return std::unexpected(ModelError{ModelErrorCode::DecodeFailed,
                "mtmd_helper_eval_chunks failed: rc=" + std::to_string(eval_rc)});
        }
        produced = static_cast<int32_t>(n_tokens);
    } else {
        // Text-only fast path. We don't add BOS — the template already
        // includes the model's special start tokens.
        const int32_t prompt_len = static_cast<int32_t>(prompt.size());
        int32_t n_needed = llama_tokenize(vocab, prompt.c_str(), prompt_len,
                                          nullptr, 0,
                                          /*add_special=*/false, /*parse_special=*/true);
        if (n_needed < 0) n_needed = -n_needed;
        std::vector<llama_token> prompt_tokens(static_cast<std::size_t>(n_needed));
        const int32_t got = llama_tokenize(
            vocab, prompt.c_str(), prompt_len,
            prompt_tokens.data(), n_needed,
            /*add_special=*/false, /*parse_special=*/true);
        if (got < 0) {
            return std::unexpected(ModelError{ModelErrorCode::TokenizeFailed,
                "tokenize failed: " + std::to_string(got)});
        }
        prompt_tokens.resize(static_cast<std::size_t>(got));

        if (got >= n_ctx) {
            return std::unexpected(ModelError{ModelErrorCode::ContextOverflow,
                "prompt tokens (" + std::to_string(got) + ") exceed context (" +
                std::to_string(n_ctx) + ")"});
        }

        // Chunked prefill: feed the prompt in n_batch-sized slices. llama
        // tracks position internally so we just hand it pointer + length per
        // chunk.
        for (int32_t off = 0; off < got; off += n_batch) {
            const int32_t n = std::min(n_batch, got - off);
            llama_batch batch = llama_batch_get_one(prompt_tokens.data() + off, n);
            if (llama_decode(ctx, batch) != 0) {
                return std::unexpected(ModelError{ModelErrorCode::DecodeFailed,
                    "prompt decode failed at offset " + std::to_string(off)});
            }
        }
        produced = got;
    }

    // 5. Sampling loop. Combines user-supplied stop sequences with the
    //    template's `additional_stops` (e.g. tool-call boundary markers).
    auto sampler = build_sampler(vocab, sampling, n_ctx);

    int32_t max_new = sampling.max_tokens;
    if (max_new <= 0) max_new = std::min(1024, n_ctx - produced - 1);

    std::vector<std::string> all_stops = sampling.stop;
    for (const auto& s : chat_params->additional_stops) all_stops.push_back(s);

    std::string raw;  // model output exactly as generated (pre-parsing)
    raw.reserve(static_cast<std::size_t>(max_new) * 4);
    int32_t generated = 0;
    std::string finish = "length";

    const auto t0 = std::chrono::steady_clock::now();
    llama_token cur = 0;

    for (; generated < max_new; ++generated) {
        cur = llama_sampler_sample(sampler.get(), ctx, -1);
        llama_sampler_accept(sampler.get(), cur);

        if (llama_vocab_is_eog(vocab, cur)) {
            finish = "stop";
            break;
        }

        const std::string piece = token_to_piece(vocab, cur);
        raw.append(piece);

        if (consume_stop_sequence(raw, all_stops)) {
            finish = "stop";
            break;
        }

        llama_batch batch = llama_batch_get_one(&cur, 1);
        if (llama_decode(ctx, batch) != 0) {
            return std::unexpected(ModelError{ModelErrorCode::DecodeFailed,
                                              "decode failed during generation"});
        }
    }

    const auto t1 = std::chrono::steady_clock::now();
    const double secs = std::chrono::duration<double>(t1 - t0).count();

    spdlog::debug("chat: raw_len={} raw=<<<{}>>>", raw.size(), raw);

    // 6. Extract reasoning_content if requested. We only invoke the common
    //    parser when thinking is on — for some chat formats (notably Qwen's
    //    `peg-native`) the parser re-emits chat-template scaffolding around
    //    the raw model output when `reasoning_format = NONE`, which is a
    //    regression versus passing the raw text straight through. With
    //    thinking off, the model already produced clean text — use it as-is.
    CompletionResult r{};
    if (thinking) {
        common_chat_parser_params pparams(*chat_params);
        pparams.reasoning_format = COMMON_REASONING_FORMAT_AUTO;
        pparams.parse_tool_calls = false;  // tool calls aren't yet plumbed through

        common_chat_msg parsed;
        try {
            parsed = common_chat_parse(raw, /*is_partial=*/false, pparams);
        } catch (const std::exception& e) {
            spdlog::warn("common_chat_parse failed: {} — returning raw text", e.what());
            parsed.content = raw;
        }
        spdlog::debug("chat: parsed content_len={} reasoning_len={}",
                      parsed.content.size(), parsed.reasoning_content.size());
        r.text              = std::move(parsed.content);
        r.reasoning_content = std::move(parsed.reasoning_content);
    } else {
        r.text = raw;
    }
    r.prompt_tokens     = produced;
    r.completion_tokens = generated;
    r.tokens_per_second = secs > 0 ? generated / secs : 0;
    r.finish_reason     = finish;
    return r;
}

}  // namespace mass_worker
