#include "mass_worker/worker_service.hpp"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <system_error>
#include <unordered_set>

#include <openssl/evp.h>
#include <spdlog/spdlog.h>

#include "mass_worker/bench.hpp"

#include "service.pb.h"
#include "worker/worker.pb.h"

namespace mass_worker {

namespace {

mass::v1::worker::WorkerDeviceType to_proto_device_type(DeviceType t) {
    switch (t) {
        case DeviceType::Cpu: return mass::v1::worker::WORKER_DEVICE_TYPE_CPU;
        case DeviceType::Gpu: return mass::v1::worker::WORKER_DEVICE_TYPE_GPU;
        default:              return mass::v1::worker::WORKER_DEVICE_TYPE_UNSPECIFIED;
    }
}

std::int32_t to_proto_mb(std::int64_t mb) {
    constexpr std::int64_t kMax = std::numeric_limits<std::int32_t>::max();
    return static_cast<std::int32_t>(std::clamp<std::int64_t>(mb, 0, kMax));
}

// Compute the model fingerprint exactly like mass/pkg/llm/config.go's
// LlamaChatConfig.Fingerprint(): sha256("chat|llama|<path>|<ctx>|<batch>|
// <flash>|<thinking>|<mmproj>|<template>|<cache_type>") truncated to the
// first 16 hex chars. MASS uses this as the dedup key, so it MUST match
// the Go shape byte-for-byte.
std::string chat_fingerprint(const ChatModelLoadConfig& c) {
    std::ostringstream ss;
    ss << "chat|llama|"
       << c.path.string() << '|'
       << c.context_size  << '|'
       << c.batch_size    << '|'
       << c.flash_attn    << '|'
       << (c.thinking ? "true" : "false") << '|'
       << c.mmproj_path.string() << '|'
       << c.chat_template << '|'
       << c.cache_type;
    const std::string blob = ss.str();

    unsigned char digest[EVP_MAX_MD_SIZE];
    unsigned int  len = 0;
    EVP_MD_CTX* ctx = EVP_MD_CTX_new();
    EVP_DigestInit_ex(ctx, EVP_sha256(), nullptr);
    EVP_DigestUpdate(ctx, blob.data(), blob.size());
    EVP_DigestFinal_ex(ctx, digest, &len);
    EVP_MD_CTX_free(ctx);

    std::ostringstream hex;
    hex << std::hex << std::setfill('0');
    for (unsigned int i = 0; i < len; ++i) {
        hex << std::setw(2) << static_cast<int>(digest[i]);
    }
    return hex.str().substr(0, 16);
}

// Mirror of LlamaEmbeddingConfig.Fingerprint() in mass/pkg/llm/config.go:
// sha256("embed|llama|<path>|<context_size>") → first 16 hex chars.
std::string embed_fingerprint(const EmbeddingModelLoadConfig& c) {
    std::ostringstream ss;
    ss << "embed|llama|" << c.path.string() << '|' << c.context_size;
    const std::string blob = ss.str();

    unsigned char digest[EVP_MAX_MD_SIZE];
    unsigned int  len = 0;
    EVP_MD_CTX* ctx = EVP_MD_CTX_new();
    EVP_DigestInit_ex(ctx, EVP_sha256(), nullptr);
    EVP_DigestUpdate(ctx, blob.data(), blob.size());
    EVP_DigestFinal_ex(ctx, digest, &len);
    EVP_MD_CTX_free(ctx);

    std::ostringstream hex;
    hex << std::hex << std::setfill('0');
    for (unsigned int i = 0; i < len; ++i) {
        hex << std::setw(2) << static_cast<int>(digest[i]);
    }
    return hex.str().substr(0, 16);
}

// Convert proto LlamaEmbeddingConfig + fetched files → internal config.
EmbeddingModelLoadConfig
to_embed_load_cfg(const mass::v1::LlamaEmbeddingConfig&        lc,
                  const std::map<int, std::filesystem::path>&  files) {
    EmbeddingModelLoadConfig cfg;
    constexpr int kRolePrimary = 1;
    if (auto it = files.find(kRolePrimary); it != files.end()) cfg.path = it->second;

    if (lc.has_context_size())   cfg.context_size   = lc.context_size();
    if (lc.has_gpu_layers())     cfg.gpu_layers     = lc.gpu_layers();
    if (lc.has_threads())        cfg.threads        = lc.threads();
    if (lc.has_max_concurrent()) cfg.max_concurrent = lc.max_concurrent();
    cfg.main_gpu = lc.main_gpu();
    cfg.tensor_split.assign(lc.tensor_split().begin(), lc.tensor_split().end());
    return cfg;
}

// Convert proto LlamaChatConfig + fetched files → ChatModelLoadConfig.
ChatModelLoadConfig
to_chat_load_cfg(const mass::v1::LlamaChatConfig&             lc,
                 const std::map<int, std::filesystem::path>&  files) {
    ChatModelLoadConfig cfg;
    constexpr int kRolePrimary    = 1;
    constexpr int kRoleProjector  = 2;
    if (auto it = files.find(kRolePrimary);   it != files.end()) cfg.path        = it->second;
    if (auto it = files.find(kRoleProjector); it != files.end()) cfg.mmproj_path = it->second;

    if (lc.has_context_size())   cfg.context_size   = lc.context_size();
    if (lc.has_batch_size())     cfg.batch_size     = lc.batch_size();
    if (lc.has_gpu_layers())     cfg.gpu_layers     = lc.gpu_layers();
    if (lc.has_threads())        cfg.threads        = lc.threads();
    if (lc.has_max_concurrent()) cfg.max_concurrent = lc.max_concurrent();

    cfg.thinking      = lc.thinking();
    cfg.main_gpu      = lc.main_gpu();
    cfg.chat_template = lc.chat_template();
    cfg.tensor_split.assign(lc.tensor_split().begin(), lc.tensor_split().end());

    if (lc.has_flash_attn()) {
        cfg.flash_attn = lc.flash_attn() ? "enabled" : "disabled";
    }
    switch (lc.cache_type()) {
        case mass::v1::CACHE_TYPE_F16:  cfg.cache_type = "f16";  break;
        case mass::v1::CACHE_TYPE_Q8_0: cfg.cache_type = "q8_0"; break;
        case mass::v1::CACHE_TYPE_Q4_0: cfg.cache_type = "q4_0"; break;
        default: /* leave empty → llama.cpp default */ break;
    }
    return cfg;
}

std::string role_to_string(mass::v1::Role r) {
    switch (r) {
        case mass::v1::ROLE_SYSTEM:    return "system";
        case mass::v1::ROLE_USER:      return "user";
        case mass::v1::ROLE_ASSISTANT: return "assistant";
        case mass::v1::ROLE_TOOL:      return "tool";
        default:                       return "user";
    }
}

// Translate proto ChatCompletionRequest → internal messages + sampling.
// Multimodal content (image/audio bytes) flows straight through to
// ChatModel::chat_completion, which routes it via libmtmd when an mmproj
// is loaded. File parts remain unsupported — the worker has no fetcher
// for arbitrary mime types.
struct ChatRequestParts {
    std::vector<ChatMessage> messages;
    SamplingParams           sampling;
};
std::expected<ChatRequestParts, std::string>
to_chat_request(const mass::v1::ChatCompletionRequest& req) {
    ChatRequestParts out;
    for (const auto& msg : req.messages()) {
        ChatMessage cm{role_to_string(msg.role()), msg.content(), {}, {}};
        for (const auto& p : msg.parts()) {
            switch (p.content_case()) {
                case mass::v1::ContentPart::kText:
                    if (!cm.content.empty()) cm.content += "\n";
                    cm.content += p.text();
                    break;
                case mass::v1::ContentPart::kImage: {
                    const auto& d = p.image().data();
                    cm.images.push_back({{d.begin(), d.end()}, p.image().mime_type()});
                    break;
                }
                case mass::v1::ContentPart::kAudio: {
                    const auto& d = p.audio().data();
                    cm.audios.push_back({{d.begin(), d.end()}, p.audio().mime_type()});
                    break;
                }
                case mass::v1::ContentPart::kFile:
                    return std::unexpected(
                        std::string("file content parts are not supported"));
                default:
                    return std::unexpected(
                        std::string("unrecognized content part variant"));
            }
        }
        out.messages.push_back(std::move(cm));
    }
    if (req.has_sampling()) {
        const auto& s = req.sampling();
        out.sampling.max_tokens        = s.has_max_tokens() ? s.max_tokens() : 0;
        out.sampling.temperature       = s.temperature();
        out.sampling.top_p             = s.top_p();
        out.sampling.top_k             = s.top_k();
        out.sampling.seed              = s.has_seed() ? s.seed() : 0;
        out.sampling.min_p             = s.min_p();
        out.sampling.repeat_penalty    = s.repeat_penalty();
        out.sampling.frequency_penalty = s.frequency_penalty();
        out.sampling.presence_penalty  = s.presence_penalty();
        out.sampling.enable_thinking   = s.enable_thinking();
        for (const auto& w : s.stop()) out.sampling.stop.push_back(w);
    }
    return out;
}

// Translate proto ModelFile → fetch-layer ModelFile.
ModelFile to_fetch_file(const mass::v1::worker::ModelFile& f) {
    ModelFile out;
    out.filename   = f.filename();
    out.url        = f.url();
    out.sha256     = f.sha256();
    out.local_path = f.local_path();
    out.role       = static_cast<int>(f.role());
    for (const auto& [k, v] : f.headers()) out.headers[k] = v;
    return out;
}

}  // namespace

WorkerService::WorkerService(std::string id, std::string name, std::string models_dir)
    : id_(std::move(id)),
      name_(std::move(name)),
      models_dir_(std::move(models_dir)),
      cache_(std::filesystem::path(models_dir_)),
      fetcher_(std::filesystem::path(models_dir_)) {}

WorkerService::~WorkerService() = default;

std::unique_ptr<mass::v1::worker::WorkerRegister> WorkerService::registration() const {
    auto reg = std::make_unique<mass::v1::worker::WorkerRegister>();
    reg->set_id(id_);
    reg->set_name(name_);
    for (const auto& d : hardware_.devices()) {
        auto* dev = reg->add_devices();
        dev->set_id(d.id);
        dev->set_name(d.name);
        dev->set_type(to_proto_device_type(d.type));
        dev->set_total_memory_mb(to_proto_mb(d.total_memory_mb));
    }
    return reg;
}

std::vector<std::unique_ptr<mass::v1::worker::WorkerDeviceStats>>
WorkerService::device_stats() const {
    std::vector<std::unique_ptr<mass::v1::worker::WorkerDeviceStats>> out;
    for (const auto& s : hardware_.stats()) {
        auto pb = std::make_unique<mass::v1::worker::WorkerDeviceStats>();
        pb->set_device_id(s.id);
        pb->set_used_memory_mb(to_proto_mb(s.used_memory_mb));
        pb->set_total_memory_mb(to_proto_mb(s.total_memory_mb));
        pb->set_utilization_pct(s.utilization_pct);
        out.push_back(std::move(pb));
    }
    return out;
}

std::vector<std::string> WorkerService::cache_files() const {
    return cache_.list_gguf();
}

namespace {

std::unique_ptr<mass::v1::worker::WorkerJobResult> job_error(std::string_view msg) {
    auto r = std::make_unique<mass::v1::worker::WorkerJobResult>();
    r->mutable_error()->set_message(std::string(msg));
    return r;
}

}  // namespace

std::unique_ptr<mass::v1::worker::WorkerJobResult>
WorkerService::execute(const mass::v1::worker::HubMessage& job) {
    using HM = mass::v1::worker::HubMessage;
    switch (job.msg_case()) {
        case HM::kBenchmark: {
            std::vector<BenchResult> results;
            const std::string& target = job.benchmark().device_id();
            if (target.empty()) {
                results = bench_all(hardware_);
            } else if (auto r = bench_one(hardware_, target); r) {
                results.push_back(std::move(*r));
            } else {
                return job_error("benchmark: unknown device " + target);
            }

            auto out = std::make_unique<mass::v1::worker::WorkerJobResult>();
            auto* resp = out->mutable_benchmark();
            for (const auto& r : results) {
                auto* item = resp->add_results();
                item->set_device_id(r.device_id);
                item->set_device_name(r.device_name);
                item->set_memory_gbs(r.memory_gbs);
                item->set_compute_gflops(r.compute_gflops);
            }
            return out;
        }
        case HM::kLoadChatModel: {
            const auto& req = job.load_chat_model();
            if (!req.config().has_llama()) {
                return job_error("LoadChatModel: missing llama config");
            }

            // 1. Fetch files. Phase 4 fetcher handles loopback + sha256
            //    + retries; we just translate proto → internal.
            std::vector<ModelFile> files;
            for (const auto& f : req.files()) files.push_back(to_fetch_file(f));
            auto fetched = fetcher_.fetch_all(files, fetch_cancel_);
            if (!fetched) {
                return job_error("LoadChatModel: " + fetched.error().message);
            }

            // 2. Build the internal config + fingerprint.
            auto cfg = to_chat_load_cfg(req.config().llama(), *fetched);
            const std::string fp = chat_fingerprint(cfg);

            // 3. Idempotent load: if already loaded under this fingerprint,
            //    return the existing entry. Race window: two loads of the
            //    same fp arrive concurrently → second one drops its
            //    duplicate after acquiring the lock.
            {
                std::shared_lock lk(models_mu_);
                if (auto it = chat_models_.find(fp); it != chat_models_.end()) {
                    auto out = std::make_unique<mass::v1::worker::WorkerJobResult>();
                    auto* lm = out->mutable_load_model();
                    lm->set_fingerprint(fp);
                    lm->set_pool_size(it->second->pool_size());
                    return out;
                }
            }

            spdlog::info("loading chat model fingerprint={} path={}",
                         fp, cfg.path.string());
            auto loaded = ChatModel::load(std::move(cfg));
            if (!loaded) {
                return job_error("LoadChatModel: " + loaded.error().message);
            }
            const int32_t actual_pool = (*loaded)->pool_size();

            std::unique_lock lk(models_mu_);
            // Check again under the write lock — another load of the same
            // fingerprint may have raced us. If it did, our newly-loaded
            // model just gets dropped here (RAII frees it).
            if (auto it = chat_models_.find(fp); it != chat_models_.end()) {
                spdlog::debug("dropping duplicate concurrent load: {}", fp);
            } else {
                chat_models_[fp] = std::move(*loaded);
            }
            lk.unlock();

            auto out = std::make_unique<mass::v1::worker::WorkerJobResult>();
            auto* lm = out->mutable_load_model();
            lm->set_fingerprint(fp);
            lm->set_pool_size(actual_pool);
            return out;
        }

        case HM::kUnloadModel: {
            const std::string& fp = job.unload_model().fingerprint();
            std::unique_lock lk(models_mu_);
            if (auto it = chat_models_.find(fp); it != chat_models_.end()) {
                chat_models_.erase(it);
                lk.unlock();
                spdlog::info("unloaded chat model: {}", fp);
                auto out = std::make_unique<mass::v1::worker::WorkerJobResult>();
                out->mutable_unload_model();
                return out;
            }
            if (auto it = embed_models_.find(fp); it != embed_models_.end()) {
                embed_models_.erase(it);
                lk.unlock();
                spdlog::info("unloaded embedding model: {}", fp);
                auto out = std::make_unique<mass::v1::worker::WorkerJobResult>();
                out->mutable_unload_model();
                return out;
            }
            return job_error("UnloadModel: model " + fp + " not found");
        }

        case HM::kTokenize: {
            const auto& req = job.tokenize();
            std::shared_ptr<ChatModel> model;
            {
                std::shared_lock lk(models_mu_);
                if (auto it = chat_models_.find(req.fingerprint());
                    it != chat_models_.end()) {
                    model = it->second;
                }
            }
            if (!model) {
                return job_error("Tokenize: model " + req.fingerprint() + " not loaded");
            }

            auto tokens = model->tokenize(req.request().text(), /*add_special=*/true);
            if (!tokens) {
                return job_error("Tokenize: " + tokens.error().message);
            }

            auto out = std::make_unique<mass::v1::worker::WorkerJobResult>();
            auto* resp = out->mutable_tokenize();
            for (auto t : *tokens) resp->add_tokens(t);
            return out;
        }

        case HM::kChatCompletion: {
            const auto& req = job.chat_completion();
            std::shared_ptr<ChatModel> model;
            {
                std::shared_lock lk(models_mu_);
                if (auto it = chat_models_.find(req.fingerprint());
                    it != chat_models_.end()) {
                    model = it->second;
                }
            }
            if (!model) {
                return job_error("ChatCompletion: model " + req.fingerprint() + " not loaded");
            }

            auto parts = to_chat_request(req.request());
            if (!parts) return job_error("ChatCompletion: " + parts.error());

            auto result = model->chat_completion(parts->messages, parts->sampling);
            if (!result) {
                return job_error("ChatCompletion: " + result.error().message);
            }

            // Build response. Mirror Go worker shape: id is a fresh UUID-ish
            // string; model field echoes the fingerprint; usage from llama
            // perf counters via tokens_per_second + token counts.
            auto out = std::make_unique<mass::v1::worker::WorkerJobResult>();
            auto* resp = out->mutable_chat_completion();
            // We don't pull in libuuid; the job_id from MASS already provides
            // the correlation handle. Use a short timestamp-derived id.
            resp->set_id("cpp-" + std::to_string(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now().time_since_epoch()).count()));
            resp->set_model(req.fingerprint());

            auto* msg = resp->mutable_message();
            msg->set_role(mass::v1::ROLE_ASSISTANT);
            msg->set_content(result->text);

            if (result->finish_reason == "stop") {
                resp->set_finish_reason(mass::v1::FINISH_REASON_STOP);
            } else if (result->finish_reason == "length") {
                resp->set_finish_reason(mass::v1::FINISH_REASON_LENGTH);
            } else {
                resp->set_finish_reason(mass::v1::FINISH_REASON_UNSPECIFIED);
            }

            resp->set_reasoning_content(result->reasoning_content);
            auto* usage = resp->mutable_usage();
            usage->set_prompt_tokens(result->prompt_tokens);
            usage->set_completion_tokens(result->completion_tokens);
            usage->set_total_tokens(result->prompt_tokens + result->completion_tokens);
            resp->set_tokens_per_second(result->tokens_per_second);
            return out;
        }

        case HM::kLoadEmbeddingModel: {
            const auto& req = job.load_embedding_model();
            if (!req.config().has_llama()) {
                return job_error("LoadEmbeddingModel: missing llama config");
            }

            std::vector<ModelFile> files;
            for (const auto& f : req.files()) files.push_back(to_fetch_file(f));
            auto fetched = fetcher_.fetch_all(files, fetch_cancel_);
            if (!fetched) {
                return job_error("LoadEmbeddingModel: " + fetched.error().message);
            }

            auto cfg = to_embed_load_cfg(req.config().llama(), *fetched);
            const std::string fp = embed_fingerprint(cfg);

            {
                std::shared_lock lk(models_mu_);
                if (auto it = embed_models_.find(fp); it != embed_models_.end()) {
                    auto out = std::make_unique<mass::v1::worker::WorkerJobResult>();
                    auto* lm = out->mutable_load_model();
                    lm->set_fingerprint(fp);
                    lm->set_pool_size(it->second->pool_size());
                    return out;
                }
            }

            spdlog::info("loading embedding model fingerprint={} path={}",
                         fp, cfg.path.string());
            auto loaded = EmbeddingModel::load(std::move(cfg));
            if (!loaded) {
                return job_error("LoadEmbeddingModel: " + loaded.error().message);
            }
            const int32_t actual_pool = (*loaded)->pool_size();

            std::unique_lock lk(models_mu_);
            if (!embed_models_.contains(fp)) {
                embed_models_[fp] = std::move(*loaded);
            }
            lk.unlock();

            auto out = std::make_unique<mass::v1::worker::WorkerJobResult>();
            auto* lm = out->mutable_load_model();
            lm->set_fingerprint(fp);
            lm->set_pool_size(actual_pool);
            return out;
        }

        case HM::kEmbedding: {
            const auto& req = job.embedding();
            std::shared_ptr<EmbeddingModel> model;
            {
                std::shared_lock lk(models_mu_);
                if (auto it = embed_models_.find(req.fingerprint());
                    it != embed_models_.end()) {
                    model = it->second;
                }
            }
            if (!model) {
                return job_error("Embedding: model " + req.fingerprint() + " not loaded");
            }

            auto vec = model->embed(req.request().input());
            if (!vec) return job_error("Embedding: " + vec.error().message);

            auto out = std::make_unique<mass::v1::worker::WorkerJobResult>();
            auto* resp = out->mutable_embedding();
            resp->set_id("cpp-emb-" + std::to_string(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now().time_since_epoch()).count()));
            resp->set_model(req.fingerprint());
            for (float v : *vec) resp->add_embedding(v);
            return out;
        }

        case HM::kBatchEmbedding: {
            const auto& req = job.batch_embedding();
            std::shared_ptr<EmbeddingModel> model;
            {
                std::shared_lock lk(models_mu_);
                if (auto it = embed_models_.find(req.fingerprint());
                    it != embed_models_.end()) {
                    model = it->second;
                }
            }
            if (!model) {
                return job_error("BatchEmbedding: model " + req.fingerprint() + " not loaded");
            }

            std::vector<std::string> inputs;
            for (const auto& s : req.request().inputs()) inputs.push_back(s);
            auto vecs = model->embed_batch(inputs);
            if (!vecs) return job_error("BatchEmbedding: " + vecs.error().message);

            auto out = std::make_unique<mass::v1::worker::WorkerJobResult>();
            auto* resp = out->mutable_batch_embedding();
            resp->set_id("cpp-bemb-" + std::to_string(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now().time_since_epoch()).count()));
            resp->set_model(req.fingerprint());
            for (std::size_t i = 0; i < vecs->size(); ++i) {
                auto* item = resp->add_embeddings();
                item->set_index(static_cast<int32_t>(i));
                for (float v : (*vecs)[i]) item->add_embedding(v);
            }
            return out;
        }
        case HM::kDeleteCacheFiles:
            // Handled inline by Runner before dispatch — should never reach here.
            return nullptr;
        case HM::MSG_NOT_SET:
            return job_error("HubMessage with no msg case set");
    }
    return job_error("unknown HubMessage case");
}

void WorkerService::delete_cache_files(const std::vector<std::string>& filenames) {
    std::unordered_set<std::filesystem::path> loaded;
    {
        std::shared_lock lock(models_mu_);
        for (const auto& [_, m] : chat_models_) {
            for (const auto& p : m->backing_paths()) loaded.insert(p);
        }
        for (const auto& [_, m] : embed_models_) {
            for (const auto& p : m->backing_paths()) loaded.insert(p);
        }
    }
    cache_.delete_files(filenames, loaded);
}

void WorkerService::shutdown() {
    spdlog::info("worker shutting down");
    fetch_cancel_.store(true, std::memory_order_release);
    std::unique_lock lock(models_mu_);
    chat_models_.clear();   // RAII unloads each ChatModel
    embed_models_.clear();  // and each EmbeddingModel
}

}  // namespace mass_worker
