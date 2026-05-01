#include "mass_worker/worker_service.hpp"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <map>
#include <sstream>
#include <system_error>
#include <unordered_set>

#include <spdlog/spdlog.h>

#include "llama-cpp/payload.pb.h"
#include "mass_worker/bench.hpp"
#include "worker/worker.pb.h"

namespace mass_worker {

namespace {

namespace pb     = ::mass::v1::worker;
namespace lcpp   = ::mass::v1::llamacpp;

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

// Convert decoded LoadHints + fetched files → internal ChatModelLoadConfig.
// `files` maps role tag ("primary", "mmproj") to absolute path.
ChatModelLoadConfig
to_chat_load_cfg(const lcpp::LoadHints&                      h,
                 const std::map<std::string, std::filesystem::path>& files) {
    ChatModelLoadConfig cfg;
    if (auto it = files.find("primary"); it != files.end()) cfg.path = it->second;
    if (auto it = files.find("mmproj");  it != files.end()) cfg.mmproj_path = it->second;

    cfg.context_size = h.context_size();
    if (h.has_batch_size())     cfg.batch_size     = h.batch_size();
    if (h.has_gpu_layers())     cfg.gpu_layers     = h.gpu_layers();
    if (h.has_threads())        cfg.threads        = h.threads();
    if (h.has_max_concurrent()) cfg.max_concurrent = h.max_concurrent();

    cfg.thinking      = h.thinking();
    cfg.main_gpu      = h.main_gpu();
    cfg.chat_template = h.chat_template();
    cfg.tensor_split.assign(h.tensor_split().begin(), h.tensor_split().end());

    if (h.has_flash_attn()) {
        cfg.flash_attn = h.flash_attn() ? "enabled" : "disabled";
    }
    switch (h.cache_type()) {
        case lcpp::CACHE_TYPE_F16:  cfg.cache_type = "f16";  break;
        case lcpp::CACHE_TYPE_Q8_0: cfg.cache_type = "q8_0"; break;
        case lcpp::CACHE_TYPE_Q4_0: cfg.cache_type = "q4_0"; break;
        default: /* leave empty → llama.cpp default */ break;
    }
    return cfg;
}

EmbeddingModelLoadConfig
to_embed_load_cfg(const lcpp::LoadHints&                              h,
                  const std::map<std::string, std::filesystem::path>& files) {
    EmbeddingModelLoadConfig cfg;
    if (auto it = files.find("primary"); it != files.end()) cfg.path = it->second;

    cfg.context_size = h.context_size();
    if (h.has_gpu_layers())     cfg.gpu_layers     = h.gpu_layers();
    if (h.has_threads())        cfg.threads        = h.threads();
    if (h.has_max_concurrent()) cfg.max_concurrent = h.max_concurrent();
    cfg.main_gpu = h.main_gpu();
    cfg.tensor_split.assign(h.tensor_split().begin(), h.tensor_split().end());
    return cfg;
}

std::string role_to_string(lcpp::Role r) {
    switch (r) {
        case lcpp::ROLE_SYSTEM:    return "system";
        case lcpp::ROLE_USER:      return "user";
        case lcpp::ROLE_ASSISTANT: return "assistant";
        case lcpp::ROLE_TOOL:      return "tool";
        default:                   return "user";
    }
}

lcpp::Role string_to_role(std::string_view s) {
    if (s == "system")    return lcpp::ROLE_SYSTEM;
    if (s == "user")      return lcpp::ROLE_USER;
    if (s == "assistant") return lcpp::ROLE_ASSISTANT;
    if (s == "tool")      return lcpp::ROLE_TOOL;
    return lcpp::ROLE_UNSPECIFIED;
}

// Translate llamacpp::ChatJob → internal messages + sampling. Multimodal
// content (image/audio bytes) flows straight through to chat_completion,
// which routes via libmtmd when an mmproj is loaded.
struct ChatRequestParts {
    std::vector<ChatMessage> messages;
    SamplingParams           sampling;
};

std::expected<ChatRequestParts, std::string>
to_chat_request(const lcpp::ChatJob& req) {
    ChatRequestParts out;
    for (const auto& msg : req.messages()) {
        ChatMessage cm{role_to_string(msg.role()), msg.content(), {}, {}};
        for (const auto& p : msg.parts()) {
            switch (p.content_case()) {
                case lcpp::ContentPart::kText:
                    if (!cm.content.empty()) cm.content += "\n";
                    cm.content += p.text();
                    break;
                case lcpp::ContentPart::kImage: {
                    const auto& d = p.image().data();
                    cm.images.push_back({{d.begin(), d.end()}, p.image().mime_type()});
                    break;
                }
                case lcpp::ContentPart::kAudio: {
                    const auto& d = p.audio().data();
                    cm.audios.push_back({{d.begin(), d.end()}, p.audio().mime_type()});
                    break;
                }
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

ModelFile to_fetch_file(const pb::ModelFile& f) {
    ModelFile out;
    out.filename   = f.filename();
    out.url        = f.url();
    out.sha256     = f.sha256();
    out.local_path = f.local_path();
    // Carry the wire enum value through to the fetch layer — it uses the
    // numeric role to detect duplicates and ensure a primary file is present.
    out.role       = static_cast<int>(f.role());
    for (const auto& [k, v] : f.headers()) out.headers[k] = v;
    return out;
}

lcpp::FinishReason finish_reason_to_proto(std::string_view r) {
    if (r == "stop")   return lcpp::FINISH_REASON_STOP;
    if (r == "length") return lcpp::FINISH_REASON_LENGTH;
    return lcpp::FINISH_REASON_UNSPECIFIED;
}

}  // namespace

WorkerService::WorkerService(std::string id, std::string name, std::string models_dir)
    : id_(std::move(id)),
      name_(std::move(name)),
      models_dir_(std::move(models_dir)),
      cache_(std::filesystem::path(models_dir_)),
      fetcher_(std::filesystem::path(models_dir_)) {}

WorkerService::~WorkerService() = default;

std::unique_ptr<pb::WorkerRegister> WorkerService::registration() const {
    auto reg = std::make_unique<pb::WorkerRegister>();
    reg->set_id(id_);
    reg->set_name(name_);
    reg->set_runtime_name(std::string(kRuntimeName));
    for (const auto& d : hardware_.devices()) {
        auto* dev = reg->add_devices();
        dev->set_id(d.id);
        dev->set_name(d.name);
        dev->set_type(to_proto_device_type(d.type));
        dev->set_total_memory_mb(to_proto_mb(d.total_memory_mb));
    }
    return reg;
}

std::unique_ptr<pb::WorkerHeartbeat> WorkerService::heartbeat() const {
    auto hb = std::make_unique<pb::WorkerHeartbeat>();

    for (const auto& s : hardware_.stats()) {
        auto* ds = hb->add_device_stats();
        ds->set_device_id(s.id);
        ds->set_used_memory_mb(to_proto_mb(s.used_memory_mb));
        ds->set_total_memory_mb(to_proto_mb(s.total_memory_mb));
        ds->set_utilization_pct(s.utilization_pct);
    }

    for (const auto& cf : cache_files()) hb->add_cache_files(cf);

    int32_t total_capacity = 0;
    {
        std::shared_lock lock(models_mu_);
        for (const auto& [id, m] : chat_models_) {
            const int32_t pool   = m->pool_size();
            int32_t       active = 0;
            {
                std::lock_guard lk(active_mu_);
                if (auto it = active_per_model_.find(id); it != active_per_model_.end()) {
                    active = it->second;
                }
            }
            total_capacity += std::max(0, pool - active);
            auto* lm = hb->add_loaded_models();
            lm->set_model_id(id);
            lm->set_pool_size(pool);
            lm->set_active(active);
        }
        for (const auto& [id, m] : embed_models_) {
            const int32_t pool   = m->pool_size();
            int32_t       active = 0;
            {
                std::lock_guard lk(active_mu_);
                if (auto it = active_per_model_.find(id); it != active_per_model_.end()) {
                    active = it->second;
                }
            }
            total_capacity += std::max(0, pool - active);
            auto* lm = hb->add_loaded_models();
            lm->set_model_id(id);
            lm->set_pool_size(pool);
            lm->set_active(active);
        }
    }
    hb->set_active_jobs(active_total_.load(std::memory_order_relaxed));
    hb->set_available_capacity(total_capacity);
    return hb;
}

std::vector<std::string> WorkerService::cache_files() const {
    return cache_.list_gguf();
}

namespace {

// terminal_error wraps a string as a WorkerMessage carrying a
// WorkerJobResult.error frame for the given job.
std::unique_ptr<pb::WorkerMessage> terminal_error(std::string_view job_id, std::string_view msg) {
    auto out = std::make_unique<pb::WorkerMessage>();
    auto* jr = out->mutable_job_result();
    jr->set_job_id(std::string(job_id));
    jr->mutable_error()->set_message(std::string(msg));
    return out;
}

std::unique_ptr<pb::WorkerMessage> load_error(std::string_view job_id,
                                              std::string_view model_id,
                                              std::string_view msg) {
    auto out = std::make_unique<pb::WorkerMessage>();
    auto* lm = out->mutable_load_model();
    lm->set_job_id(std::string(job_id));
    lm->set_model_id(std::string(model_id));
    lm->set_error(std::string(msg));
    return out;
}

std::unique_ptr<pb::WorkerMessage> unload_error(std::string_view job_id,
                                                std::string_view model_id,
                                                std::string_view msg) {
    auto out = std::make_unique<pb::WorkerMessage>();
    auto* um = out->mutable_unload_model();
    um->set_job_id(std::string(job_id));
    um->set_model_id(std::string(model_id));
    um->set_error(std::string(msg));
    return out;
}

// model_file_role_to_string maps the proto enum to the short string tags
// downstream load helpers key on. Unspecified is treated as PRIMARY for
// backwards-tolerance with older gateways that left the field unset.
std::string model_file_role_to_string(pb::ModelFileRole role) {
    switch (role) {
        case pb::MODEL_FILE_ROLE_PRIMARY:     return "primary";
        case pb::MODEL_FILE_ROLE_MMPROJ:      return "mmproj";
        case pb::MODEL_FILE_ROLE_UNSPECIFIED: return "primary";
        default:                              return "";
    }
}

// group_by_role takes proto ModelFile entries (already fetched/resolved by
// the fetch layer onto absolute paths) and groups them by the role tag the
// gateway set ("primary" / "mmproj" / etc).
std::map<std::string, std::filesystem::path>
group_by_role(const std::map<int, std::filesystem::path>& fetched_by_idx,
              const google::protobuf::RepeatedPtrField<pb::ModelFile>& files) {
    std::map<std::string, std::filesystem::path> out;
    int idx = 0;
    for (const auto& f : files) {
        if (auto it = fetched_by_idx.find(idx); it != fetched_by_idx.end()) {
            auto role = model_file_role_to_string(f.role());
            if (!role.empty()) {
                out[std::move(role)] = it->second;
            }
        }
        ++idx;
    }
    return out;
}

}  // namespace

// active_guard: scoped RAII helper that bumps active counters on construction
// and decrements on destruction. Defined inside the .cpp because it needs to
// touch service-private mutexes; not part of the public header.
namespace {

class ActiveGuard {
public:
    ActiveGuard(std::mutex& mu, std::unordered_map<std::string, int32_t>& counts,
                std::atomic<int32_t>& total, std::string model_id)
        : mu_(mu), counts_(counts), total_(total), model_id_(std::move(model_id)) {
        std::lock_guard lk(mu_);
        ++counts_[model_id_];
        total_.fetch_add(1, std::memory_order_relaxed);
    }
    ~ActiveGuard() {
        std::lock_guard lk(mu_);
        if (auto it = counts_.find(model_id_); it != counts_.end()) {
            if (--it->second <= 0) counts_.erase(it);
        }
        total_.fetch_sub(1, std::memory_order_relaxed);
    }
    ActiveGuard(const ActiveGuard&) = delete;
    ActiveGuard& operator=(const ActiveGuard&) = delete;
private:
    std::mutex& mu_;
    std::unordered_map<std::string, int32_t>& counts_;
    std::atomic<int32_t>& total_;
    std::string model_id_;
};

}  // namespace

std::unique_ptr<pb::WorkerMessage>
WorkerService::execute(const pb::HubMessage& job, EmittedFn emit) {
    using HM = pb::HubMessage;
    switch (job.msg_case()) {
        case HM::kAssignJob: {
            const auto& aj = job.assign_job();
            const std::string& job_id  = aj.job_id();
            const std::string& mid     = aj.model_id();

            // Try chat first, then embedding. Maps are disjoint by load type.
            std::shared_ptr<ChatModel>      chat;
            std::shared_ptr<EmbeddingModel> embed;
            {
                std::shared_lock lk(models_mu_);
                if (auto it = chat_models_.find(mid); it != chat_models_.end()) chat = it->second;
                else if (auto it2 = embed_models_.find(mid); it2 != embed_models_.end()) embed = it2->second;
            }
            if (!chat && !embed) {
                return terminal_error(job_id, "AssignJob: model not loaded: " + mid);
            }

            lcpp::Job decoded;
            if (!decoded.ParseFromString(aj.payload())) {
                return terminal_error(job_id, "AssignJob: invalid payload encoding");
            }

            ActiveGuard guard(active_mu_, active_per_model_, active_total_, mid);

            switch (decoded.kind()) {
                case lcpp::JOB_KIND_CHAT: {
                    if (!chat) return terminal_error(job_id, "Chat: not a chat model");
                    auto parts = to_chat_request(decoded.chat());
                    if (!parts) return terminal_error(job_id, "Chat: " + parts.error());

                    const bool streaming = decoded.chat().stream();

                    // For streaming, emit per-token JobChunks as they're
                    // generated. The terminal frame still ships the final
                    // chat_final with usage + finish_reason; its `message`
                    // is left empty since the body already streamed.
                    ChatModel::OnTokenFn on_token;
                    if (streaming) {
                        on_token = [&](std::string_view piece) {
                            lcpp::JobChunk delta;
                            auto* cc = delta.mutable_chat();
                            cc->set_role(lcpp::ROLE_ASSISTANT);
                            cc->set_content(std::string(piece));
                            pb::WorkerMessage msg;
                            auto* jr = msg.mutable_job_result();
                            jr->set_job_id(job_id);
                            delta.SerializeToString(jr->mutable_chunk());
                            (void)emit(msg);
                        };
                    }

                    auto result = chat->chat_completion_stream(parts->messages, parts->sampling, on_token);
                    if (!result) return terminal_error(job_id, "Chat: " + result.error().message);

                    // Build the terminal JobChunk → encode into
                    // WorkerJobResult.completed.
                    lcpp::JobChunk chunk;
                    auto* cf = chunk.mutable_chat_final();
                    cf->set_id("cpp-" + std::to_string(
                        std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now().time_since_epoch()).count()));
                    if (!streaming) {
                        auto* msg = cf->mutable_message();
                        msg->set_role(lcpp::ROLE_ASSISTANT);
                        msg->set_content(result->text);
                    }
                    cf->set_finish_reason(finish_reason_to_proto(result->finish_reason));
                    cf->set_reasoning_content(result->reasoning_content);
                    auto* usage = cf->mutable_usage();
                    usage->set_prompt_tokens(result->prompt_tokens);
                    usage->set_completion_tokens(result->completion_tokens);
                    usage->set_total_tokens(result->prompt_tokens + result->completion_tokens);
                    cf->set_tokens_per_second(result->tokens_per_second);

                    auto out = std::make_unique<pb::WorkerMessage>();
                    auto* jr = out->mutable_job_result();
                    jr->set_job_id(job_id);
                    chunk.SerializeToString(jr->mutable_completed()->mutable_final_response());
                    return out;
                }
                case lcpp::JOB_KIND_EMBED: {
                    if (!embed) return terminal_error(job_id, "Embed: not an embedding model");
                    auto vec = embed->embed(decoded.embed().input());
                    if (!vec) return terminal_error(job_id, "Embed: " + vec.error().message);

                    lcpp::JobChunk chunk;
                    auto* er = chunk.mutable_embed();
                    er->set_id("cpp-emb-" + std::to_string(
                        std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now().time_since_epoch()).count()));
                    for (float v : *vec) er->add_embedding(v);

                    auto out = std::make_unique<pb::WorkerMessage>();
                    auto* jr = out->mutable_job_result();
                    jr->set_job_id(job_id);
                    chunk.SerializeToString(jr->mutable_completed()->mutable_final_response());
                    return out;
                }
                case lcpp::JOB_KIND_BATCH_EMBED: {
                    if (!embed) return terminal_error(job_id, "BatchEmbed: not an embedding model");
                    std::vector<std::string> inputs;
                    for (const auto& s : decoded.batch_embed().inputs()) inputs.push_back(s);
                    auto vecs = embed->embed_batch(inputs);
                    if (!vecs) return terminal_error(job_id, "BatchEmbed: " + vecs.error().message);

                    lcpp::JobChunk chunk;
                    auto* br = chunk.mutable_batch_embed();
                    br->set_id("cpp-bemb-" + std::to_string(
                        std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now().time_since_epoch()).count()));
                    for (std::size_t i = 0; i < vecs->size(); ++i) {
                        auto* item = br->add_items();
                        item->set_index(static_cast<int32_t>(i));
                        for (float v : (*vecs)[i]) item->add_embedding(v);
                    }

                    auto out = std::make_unique<pb::WorkerMessage>();
                    auto* jr = out->mutable_job_result();
                    jr->set_job_id(job_id);
                    chunk.SerializeToString(jr->mutable_completed()->mutable_final_response());
                    return out;
                }
                case lcpp::JOB_KIND_TOKENIZE: {
                    if (!chat) return terminal_error(job_id, "Tokenize: not a chat model");
                    auto tokens = chat->tokenize(decoded.tokenize().text(), /*add_special=*/true);
                    if (!tokens) return terminal_error(job_id, "Tokenize: " + tokens.error().message);

                    lcpp::JobChunk chunk;
                    auto* tr = chunk.mutable_tokenize();
                    for (auto t : *tokens) tr->add_tokens(t);

                    auto out = std::make_unique<pb::WorkerMessage>();
                    auto* jr = out->mutable_job_result();
                    jr->set_job_id(job_id);
                    chunk.SerializeToString(jr->mutable_completed()->mutable_final_response());
                    return out;
                }
                default:
                    return terminal_error(job_id, "AssignJob: unknown job kind");
            }
        }

        case HM::kCancelJob:
            // Best-effort cancellation isn't yet wired into the chat loop;
            // for now the cancel arrives, we log it, and the job completes
            // normally. Future work: pass a token to chat_completion that
            // the sampler checks every N tokens.
            spdlog::debug("CancelJob received for {}", job.cancel_job().job_id());
            return nullptr;

        case HM::kLoadModel: {
            const auto& req = job.load_model();
            const std::string& job_id = req.job_id();
            const std::string& mid    = req.model_id();

            lcpp::LoadHints hints;
            if (!hints.ParseFromString(req.load_hints())) {
                return load_error(job_id, mid, "LoadModel: invalid load_hints encoding");
            }

            // Fetch files via the Phase 4 fetcher (loopback / sha256 / retry).
            std::vector<ModelFile> files;
            for (const auto& f : req.files()) files.push_back(to_fetch_file(f));
            auto fetched = fetcher_.fetch_all(files, fetch_cancel_);
            if (!fetched) {
                return load_error(job_id, mid, "LoadModel: " + fetched.error().message);
            }
            auto by_role = group_by_role(*fetched, req.files());

            const lcpp::LoadKind kind = hints.kind();

            if (kind == lcpp::LOAD_KIND_CHAT || kind == lcpp::LOAD_KIND_UNSPECIFIED) {
                // Default to chat if unspecified — gateways should set kind
                // explicitly but this matches the gateway's current default.
                {
                    std::shared_lock lk(models_mu_);
                    if (auto it = chat_models_.find(mid); it != chat_models_.end()) {
                        auto out = std::make_unique<pb::WorkerMessage>();
                        auto* lm = out->mutable_load_model();
                        lm->set_job_id(job_id);
                        lm->set_model_id(mid);
                        lm->set_pool_size(it->second->pool_size());
                        return out;
                    }
                }
                auto cfg = to_chat_load_cfg(hints, by_role);
                cfg.allowed_devices = allowed_load_devices();
                spdlog::info("loading chat model id={} path={}", mid, cfg.path.string());
                auto loaded = ChatModel::load(std::move(cfg));
                if (!loaded) {
                    return load_error(job_id, mid, "LoadModel(chat): " + loaded.error().message);
                }
                const int32_t pool = (*loaded)->pool_size();
                {
                    std::unique_lock lk(models_mu_);
                    if (!chat_models_.contains(mid)) chat_models_[mid] = std::move(*loaded);
                }
                auto out = std::make_unique<pb::WorkerMessage>();
                auto* lm = out->mutable_load_model();
                lm->set_job_id(job_id);
                lm->set_model_id(mid);
                lm->set_pool_size(pool);
                return out;
            }

            // LOAD_KIND_EMBEDDING.
            {
                std::shared_lock lk(models_mu_);
                if (auto it = embed_models_.find(mid); it != embed_models_.end()) {
                    auto out = std::make_unique<pb::WorkerMessage>();
                    auto* lm = out->mutable_load_model();
                    lm->set_job_id(job_id);
                    lm->set_model_id(mid);
                    lm->set_pool_size(it->second->pool_size());
                    return out;
                }
            }
            auto cfg = to_embed_load_cfg(hints, by_role);
            cfg.allowed_devices = allowed_load_devices();
            spdlog::info("loading embedding model id={} path={}", mid, cfg.path.string());
            auto loaded = EmbeddingModel::load(std::move(cfg));
            if (!loaded) {
                return load_error(job_id, mid, "LoadModel(embed): " + loaded.error().message);
            }
            const int32_t pool = (*loaded)->pool_size();
            {
                std::unique_lock lk(models_mu_);
                if (!embed_models_.contains(mid)) embed_models_[mid] = std::move(*loaded);
            }
            auto out = std::make_unique<pb::WorkerMessage>();
            auto* lm = out->mutable_load_model();
            lm->set_job_id(job_id);
            lm->set_model_id(mid);
            lm->set_pool_size(pool);
            return out;
        }

        case HM::kUnloadModel: {
            const auto& req = job.unload_model();
            const std::string& job_id = req.job_id();
            const std::string& mid    = req.model_id();
            std::unique_lock lk(models_mu_);
            if (auto it = chat_models_.find(mid); it != chat_models_.end()) {
                chat_models_.erase(it);
                lk.unlock();
                spdlog::info("unloaded chat model: {}", mid);
                auto out = std::make_unique<pb::WorkerMessage>();
                auto* um = out->mutable_unload_model();
                um->set_job_id(job_id);
                um->set_model_id(mid);
                return out;
            }
            if (auto it = embed_models_.find(mid); it != embed_models_.end()) {
                embed_models_.erase(it);
                lk.unlock();
                spdlog::info("unloaded embedding model: {}", mid);
                auto out = std::make_unique<pb::WorkerMessage>();
                auto* um = out->mutable_unload_model();
                um->set_job_id(job_id);
                um->set_model_id(mid);
                return out;
            }
            return unload_error(job_id, mid, "UnloadModel: model not loaded");
        }

        case HM::kDeleteCacheFiles:
            // Handled inline by Runner before dispatch — should never reach here.
            return nullptr;

        case HM::kSetEnabledDevices: {
            const auto& s = job.set_enabled_devices();
            std::vector<std::string> ids(s.enabled_device_ids().begin(),
                                         s.enabled_device_ids().end());
            set_enabled_devices(std::move(ids));
            return nullptr;  // fire-and-forget, no terminal frame
        }

        case HM::kBenchmark: {
            const auto& b = job.benchmark();
            const std::string& job_id   = b.job_id();
            const std::string& dev_id   = b.device_id();

            std::vector<BenchResult> results;
            std::string err;
            try {
                if (dev_id.empty()) {
                    results = bench_all(hardware_);
                } else if (auto r = bench_one(hardware_, dev_id); r) {
                    results.push_back(*r);
                } else {
                    err = "unknown device: " + dev_id;
                }
            } catch (const std::exception& e) {
                err = e.what();
            }

            auto out = std::make_unique<pb::WorkerMessage>();
            auto* br = out->mutable_benchmark();
            br->set_job_id(job_id);
            if (!err.empty()) {
                br->set_error(err);
            }
            for (const auto& r : results) {
                auto* d = br->add_results();
                d->set_device_id(r.device_id);
                d->set_device_name(r.device_name);
                d->set_memory_gbs(r.memory_gbs);
                d->set_compute_gflops(r.compute_gflops);
            }
            return out;
        }

        case HM::MSG_NOT_SET:
            return terminal_error("", "HubMessage with no msg case set");
    }
    return terminal_error("", "unknown HubMessage case");
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

void WorkerService::set_enabled_devices(std::vector<std::string> ids) {
    std::unique_lock lk(enabled_mu_);
    if (ids.empty()) {
        // Empty wire payload = bootstrap default ("all advertised devices").
        // Keep as nullopt so allowed_load_devices() returns an empty list and
        // ChatModel/EmbeddingModel see the unconstrained mparams.devices.
        enabled_devices_.reset();
    } else {
        enabled_devices_.emplace(std::make_move_iterator(ids.begin()),
                                 std::make_move_iterator(ids.end()));
    }
    spdlog::info("set enabled devices: {}",
                 enabled_devices_ ? std::to_string(enabled_devices_->size()) : "<all>");
}

std::vector<ggml_backend_dev_t> WorkerService::allowed_load_devices() const {
    std::shared_lock lk(enabled_mu_);
    if (!enabled_devices_) {
        return {};  // nullopt = let llama.cpp use every backend (default).
    }
    // Mirror Hardware::Hardware()'s ggml-backend walk so the (gpu:N → device)
    // mapping matches what MASS sees over the wire. CPU is always permitted —
    // llama.cpp needs the CPU backend for tokenization and a GPU-only
    // whitelist would refuse to load.
    std::vector<ggml_backend_dev_t> out;
    const auto& set = *enabled_devices_;
    int gpu_index = 0;
    const std::size_t n = ggml_backend_dev_count();
    for (std::size_t i = 0; i < n; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (!dev) continue;
        ggml_backend_dev_props props{};
        ggml_backend_dev_get_props(dev, &props);
        switch (props.type) {
            case GGML_BACKEND_DEVICE_TYPE_CPU:
                // Always include CPU regardless of the operator whitelist.
                out.push_back(dev);
                break;
            case GGML_BACKEND_DEVICE_TYPE_GPU:
            case GGML_BACKEND_DEVICE_TYPE_IGPU: {
                std::string id = "gpu:" + std::to_string(gpu_index++);
                if (set.contains(id)) out.push_back(dev);
                break;
            }
            default:
                break;
        }
    }
    return out;
}

void WorkerService::shutdown() {
    spdlog::info("worker shutting down");
    fetch_cancel_.store(true, std::memory_order_release);
    std::unique_lock lock(models_mu_);
    chat_models_.clear();
    embed_models_.clear();
}

}  // namespace mass_worker
