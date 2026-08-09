#include "mass_worker/worker_service.hpp"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <filesystem>
#include <map>
#include <optional>
#include <sstream>
#include <stop_token>
#include <system_error>
#include <unordered_set>

#include "llama-cpp/payload.pb.h"
#include "mass_worker/batch_runner.hpp"
#include "mass_worker/bench.hpp"
#include "mass_worker/version.hpp"
#include "worker/worker.pb.h"

namespace mass_worker {

namespace {

namespace pb = ::mass::v1::worker;
namespace lcpp = ::mass::v1::llamacpp;

mass::v1::worker::WorkerDeviceType to_proto_device_type(DeviceType t) {
    switch (t) {
        case DeviceType::Cpu:
            return mass::v1::worker::WORKER_DEVICE_TYPE_CPU;
        case DeviceType::Gpu:
            return mass::v1::worker::WORKER_DEVICE_TYPE_GPU;
        default:
            return mass::v1::worker::WORKER_DEVICE_TYPE_UNSPECIFIED;
    }
}

std::int32_t to_proto_mb(std::int64_t mb) {
    constexpr std::int64_t kMax = std::numeric_limits<std::int32_t>::max();
    return static_cast<std::int32_t>(std::clamp<std::int64_t>(mb, 0, kMax));
}

// Convert decoded LoadHints + fetched files → internal ChatModelLoadConfig.
// `files` maps role tag ("primary", "mmproj") to absolute path.
// default_headroom is the worker-wide --vram-headroom-pct fallback used
// when the hints don't carry a vram_headroom_pct override.
ChatModelLoadConfig to_chat_load_cfg(const lcpp::LoadHints& h,
                                     const std::map<std::string, std::filesystem::path>& files,
                                     int32_t default_headroom) {
    ChatModelLoadConfig cfg;
    if (auto it = files.find("primary"); it != files.end()) cfg.path = it->second;
    if (auto it = files.find("mmproj"); it != files.end()) cfg.mmproj_path = it->second;

    cfg.context_size = h.context_size();
    if (h.has_batch_size()) cfg.batch_size = h.batch_size();
    if (h.has_gpu_layers()) cfg.gpu_layers = h.gpu_layers();
    if (h.has_threads()) cfg.threads = h.threads();
    if (h.has_max_concurrent()) cfg.max_concurrent = h.max_concurrent();
    cfg.vram_headroom_pct = h.has_vram_headroom_pct() ? h.vram_headroom_pct() : default_headroom;

    cfg.thinking = h.thinking();
    cfg.chat_template = h.chat_template();

    if (h.has_flash_attn()) {
        cfg.flash_attn = h.flash_attn() ? "enabled" : "disabled";
    }
    switch (h.cache_type()) {
        case lcpp::CACHE_TYPE_F16:
            cfg.cache_type = "f16";
            break;
        case lcpp::CACHE_TYPE_Q8_0:
            cfg.cache_type = "q8_0";
            break;
        case lcpp::CACHE_TYPE_Q4_0:
            cfg.cache_type = "q4_0";
            break;
        default: /* leave empty → llama.cpp default */
            break;
    }
    return cfg;
}

EmbeddingModelLoadConfig to_embed_load_cfg(
    const lcpp::LoadHints& h, const std::map<std::string, std::filesystem::path>& files,
    int32_t default_headroom) {
    EmbeddingModelLoadConfig cfg;
    if (auto it = files.find("primary"); it != files.end()) cfg.path = it->second;

    cfg.context_size = h.context_size();
    if (h.has_gpu_layers()) cfg.gpu_layers = h.gpu_layers();
    if (h.has_threads()) cfg.threads = h.threads();
    if (h.has_max_concurrent()) cfg.max_concurrent = h.max_concurrent();
    cfg.vram_headroom_pct = h.has_vram_headroom_pct() ? h.vram_headroom_pct() : default_headroom;
    return cfg;
}

std::string role_to_string(lcpp::Role r) {
    switch (r) {
        case lcpp::ROLE_SYSTEM:
            return "system";
        case lcpp::ROLE_USER:
            return "user";
        case lcpp::ROLE_ASSISTANT:
            return "assistant";
        case lcpp::ROLE_TOOL:
            return "tool";
        default:
            return "user";
    }
}

// Translate llamacpp::ChatJob → internal messages + sampling. Multimodal
// content (image/audio bytes) flows straight through to chat_completion,
// which routes via libmtmd when an mmproj is loaded.
struct ChatRequestParts {
    std::vector<ChatMessage> messages;
    SamplingParams sampling;
};

// Generalized over any proto carrying repeated ChatMessage messages + an
// optional SamplingParams (ChatJob and BatchChatItem share that shape).
template <typename ChatLike>
std::expected<ChatRequestParts, std::string> to_chat_request(const ChatLike& req) {
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
                    return std::unexpected(std::string("unrecognized content part variant"));
            }
        }
        out.messages.push_back(std::move(cm));
    }
    if (req.has_sampling()) {
        // Presence maps 1:1 onto the internal optionals: an absent wire
        // field stays nullopt (worker default), a present one applies
        // exactly as sent — including zero (temperature 0 → greedy,
        // seed 0 → seed 0).
        const auto& s = req.sampling();
        if (s.has_max_tokens()) out.sampling.max_tokens = s.max_tokens();
        if (s.has_temperature()) out.sampling.temperature = s.temperature();
        if (s.has_top_p()) out.sampling.top_p = s.top_p();
        if (s.has_top_k()) out.sampling.top_k = s.top_k();
        if (s.has_seed()) out.sampling.seed = s.seed();
        if (s.has_min_p()) out.sampling.min_p = s.min_p();
        if (s.has_repeat_penalty()) out.sampling.repeat_penalty = s.repeat_penalty();
        if (s.has_frequency_penalty()) out.sampling.frequency_penalty = s.frequency_penalty();
        if (s.has_presence_penalty()) out.sampling.presence_penalty = s.presence_penalty();
        out.sampling.enable_thinking = s.enable_thinking();
        for (const auto& w : s.stop()) out.sampling.stop.push_back(w);
    }
    return out;
}

ModelFile to_fetch_file(const pb::ModelFile& f) {
    ModelFile out;
    out.filename = f.filename();
    out.url = f.url();
    out.sha256 = f.sha256();
    out.local_path = f.local_path();
    // Carry the wire enum value through to the fetch layer — it uses the
    // numeric role to detect duplicates and ensure a primary file is present.
    out.role = static_cast<int>(f.role());
    for (const auto& [k, v] : f.headers()) out.headers[k] = v;
    return out;
}

lcpp::FinishReason finish_reason_to_proto(std::string_view r) {
    if (r == "stop") return lcpp::FINISH_REASON_STOP;
    if (r == "length") return lcpp::FINISH_REASON_LENGTH;
    return lcpp::FINISH_REASON_UNSPECIFIED;
}

std::string monotonic_id(std::string_view prefix) {
    return std::string(prefix) +
           std::to_string(std::chrono::duration_cast<std::chrono::nanoseconds>(
                              std::chrono::steady_clock::now().time_since_epoch())
                              .count());
}

// Populate a ChatFinal from a completion result. include_message is false for
// streaming chat (the body already streamed) and true for non-streaming and
// batch chat (the message is the response).
template <typename Result>
void fill_chat_final(lcpp::ChatFinal* cf, const Result& result, bool include_message) {
    cf->set_id(monotonic_id("cpp-"));
    if (include_message) {
        auto* msg = cf->mutable_message();
        msg->set_role(lcpp::ROLE_ASSISTANT);
        msg->set_content(result.text);
    }
    cf->set_finish_reason(finish_reason_to_proto(result.finish_reason));
    cf->set_reasoning_content(result.reasoning_content);
    auto* usage = cf->mutable_usage();
    usage->set_prompt_tokens(result.prompt_tokens);
    usage->set_completion_tokens(result.completion_tokens);
    usage->set_total_tokens(result.prompt_tokens + result.completion_tokens);
    cf->set_tokens_per_second(result.tokens_per_second);
}

}  // namespace

std::vector<std::string> enabled_placement_ids(
    const std::optional<std::unordered_set<std::string>>& enabled, int gpu_count, bool has_cpu) {
    std::vector<std::string> ids;
    for (int i = 0; i < gpu_count; ++i) {
        std::string id = "gpu:" + std::to_string(i);
        if (!enabled || enabled->contains(id)) ids.push_back(std::move(id));
    }
    if (ids.empty() && has_cpu && (!enabled || enabled->contains("cpu:0"))) {
        ids.emplace_back("cpu:0");
    }
    return ids;
}

std::vector<std::string> loaded_model_file_keys(const pb::HubLoadModel& req) {
    std::vector<std::string> keys;
    keys.reserve(static_cast<std::size_t>(req.files_size()));
    for (const auto& f : req.files()) {
        if (!f.filename().empty()) keys.push_back(f.filename());
    }
    return keys;
}

WorkerService::WorkerService(std::string id, std::string name, std::string models_dir,
                             int32_t default_vram_headroom_pct)
    : id_(std::move(id)),
      name_(std::move(name)),
      models_dir_(std::move(models_dir)),
      default_vram_headroom_pct_(default_vram_headroom_pct),
      cache_(std::filesystem::path(models_dir_)),
      fetcher_(std::filesystem::path(models_dir_)) {
    // id_ is the worker's local display label ("llama-<host>"), distinct from
    // the MASS-assigned worker_id; logged once so it appears in the process's
    // startup breadcrumb even though it no longer rides the register frame.
    spdlog::debug("worker service init id={} name={} models_dir={}", id_, name_, models_dir_);
    // Construction is process startup: no download is in flight yet, so any
    // "*.downloading" partial on disk is an orphan from a previous run.
    fetcher_.sweep_partials();
}

WorkerService::~WorkerService() = default;

std::unique_ptr<pb::WorkerRegister> WorkerService::registration() const {
    auto reg = std::make_unique<pb::WorkerRegister>();
    // The worker's identity is no longer carried in WorkerRegister: on first
    // connect MASS assigns it at join-token enrollment (WorkerEnrolled), and on
    // every later connect it arrives via the x-mass-worker-id stream metadata.
    reg->set_name(name_);
    reg->set_runtime_name(std::string(kRuntimeName));
    // The effective --vram-headroom-pct value, fixed for the process's
    // lifetime — MASS's pool-size and wall-clock projections must use it
    // instead of assuming a compiled-in default.
    reg->set_vram_headroom_pct(default_vram_headroom_pct_);
    // The worker's own semver and the runtime-version range whose payloads it
    // decodes — MASS rejects a worker whose range doesn't cover the installed
    // runtime. Both compiled in at build time (version.hpp).
    reg->set_version(kVersion);
    reg->set_compatible(kCompatibleRuntimes);
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
            const int32_t pool = m->pool_size();
            int32_t active = 0;
            {
                std::scoped_lock lk(active_mu_);
                if (auto it = active_per_model_.find(id); it != active_per_model_.end()) {
                    active = it->second;
                }
            }
            total_capacity += std::max(0, pool - active);
            auto* lm = hb->add_loaded_models();
            lm->set_model_id(id);
            lm->set_pool_size(pool);
            lm->set_active(active);
            for (const auto& did : m->device_ids()) lm->add_device_ids(did);
            if (auto fit = model_files_.find(id); fit != model_files_.end()) {
                for (const auto& key : fit->second) lm->add_files(key);
            }
        }
        for (const auto& [id, m] : embed_models_) {
            const int32_t pool = m->pool_size();
            int32_t active = 0;
            {
                std::scoped_lock lk(active_mu_);
                if (auto it = active_per_model_.find(id); it != active_per_model_.end()) {
                    active = it->second;
                }
            }
            total_capacity += std::max(0, pool - active);
            auto* lm = hb->add_loaded_models();
            lm->set_model_id(id);
            lm->set_pool_size(pool);
            lm->set_active(active);
            for (const auto& did : m->device_ids()) lm->add_device_ids(did);
            if (auto fit = model_files_.find(id); fit != model_files_.end()) {
                for (const auto& key : fit->second) lm->add_files(key);
            }
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

std::string lowercased(std::string_view msg) {
    std::string lower(msg);
    std::ranges::transform(lower, lower.begin(),
                           [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return lower;
}

}  // namespace

bool mentions_device_loss(std::string_view msg) {
    const std::string lower = lowercased(msg);
    return lower.find("devicelost") != std::string::npos ||
           lower.find("device_lost") != std::string::npos ||
           lower.find("device lost") != std::string::npos;
}

bool mentions_allocation_failure(std::string_view msg) {
    const std::string lower = lowercased(msg);
    static constexpr std::string_view kNeedles[] = {
        "out of memory",     "outofdevicememory",  "outofhostmemory",     "bad_alloc",
        "bad alloc",         "failed to allocate", "cannot allocate",     "unable to allocate",
        "allocation failed", "not enough vram",    "insufficient memory",
    };
    return std::ranges::any_of(
        kNeedles, [&](std::string_view n) { return lower.find(n) != std::string::npos; });
}

namespace {

// frame_error_text: the error message a terminal frame carries, if any —
// the one place every failure's text passes through, whatever its kind.
std::string_view frame_error_text(const pb::WorkerMessage& msg) {
    switch (msg.msg_case()) {
        case pb::WorkerMessage::kJobResult:
            return msg.job_result().has_error() ? msg.job_result().error().message()
                                                : std::string_view{};
        case pb::WorkerMessage::kLoadModel:
            return msg.load_model().error();
        case pb::WorkerMessage::kUnloadModel:
            return msg.unload_model().error();
        case pb::WorkerMessage::kBenchmark:
            return msg.benchmark().error();
        case pb::WorkerMessage::kModelBenchmark:
            return msg.model_benchmark().has_failure() ? msg.model_benchmark().failure().message()
                                                       : std::string_view{};
        default:
            return {};
    }
}

// terminal_error wraps a string as a WorkerMessage carrying a
// WorkerJobResult.error frame for the given job.
std::unique_ptr<pb::WorkerMessage> terminal_error(std::string_view job_id, std::string_view msg) {
    auto out = std::make_unique<pb::WorkerMessage>();
    auto* jr = out->mutable_job_result();
    jr->set_job_id(std::string(job_id));
    jr->mutable_error()->set_message(std::string(msg));
    return out;
}

std::unique_ptr<pb::WorkerMessage> load_error(std::string_view job_id, std::string_view model_id,
                                              std::string_view msg) {
    auto out = std::make_unique<pb::WorkerMessage>();
    auto* lm = out->mutable_load_model();
    lm->set_job_id(std::string(job_id));
    lm->set_model_id(std::string(model_id));
    lm->set_error(std::string(msg));
    return out;
}

std::unique_ptr<pb::WorkerMessage> unload_error(std::string_view job_id, std::string_view model_id,
                                                std::string_view msg) {
    auto out = std::make_unique<pb::WorkerMessage>();
    auto* um = out->mutable_unload_model();
    um->set_job_id(std::string(job_id));
    um->set_model_id(std::string(model_id));
    um->set_error(std::string(msg));
    return out;
}

std::unique_ptr<pb::WorkerMessage> bench_failure(std::string_view model_id,
                                                 pb::ModelBenchmarkFailureKind kind,
                                                 std::string_view msg) {
    spdlog::error("model benchmark failed model_id={} kind={} {}", model_id,
                  pb::ModelBenchmarkFailureKind_Name(kind), msg);
    auto out = std::make_unique<pb::WorkerMessage>();
    auto* mb = out->mutable_model_benchmark();
    mb->set_model_id(std::string(model_id));
    auto* f = mb->mutable_failure();
    f->set_kind(kind);
    f->set_message(std::string(msg));
    return out;
}

// classify_bench_failure decides whether MASS may treat a failed bench as
// a permanent capability verdict. INCAPABLE is reserved for the two
// signals that genuinely say "this device set cannot hold this model":
// the context allocator refusing slot 0 (ContextCreateFailed exists for
// nothing else), and an allocation spelling anywhere in the message.
// Everything else — a corrupt file, a template failure, an
// unclassifiable throw — is TRANSIENT, because MASS retries those and
// only persists them as incapable after the retry cap.
pb::ModelBenchmarkFailureKind classify_bench_failure(std::string_view msg,
                                                     bool context_alloc_failed) {
    return context_alloc_failed || mentions_allocation_failure(msg)
               ? pb::MODEL_BENCHMARK_FAILURE_KIND_INCAPABLE
               : pb::MODEL_BENCHMARK_FAILURE_KIND_TRANSIENT;
}

pb::ModelBenchmarkFailureKind classify_load_failure(const ModelError& e) {
    return classify_bench_failure(e.message, e.code == ModelErrorCode::ContextCreateFailed);
}

// model_file_role_to_string maps the proto enum to the short string tags
// downstream load helpers key on. Unspecified is treated as PRIMARY for
// backwards-tolerance with older gateways that left the field unset.
std::string model_file_role_to_string(pb::ModelFileRole role) {
    switch (role) {
        case pb::MODEL_FILE_ROLE_PRIMARY:
            return "primary";
        case pb::MODEL_FILE_ROLE_MMPROJ:
            return "mmproj";
        case pb::MODEL_FILE_ROLE_UNSPECIFIED:
            return "primary";
        default:
            return "";
    }
}

// group_by_role takes proto ModelFile entries (already fetched/resolved by
// the fetch layer onto absolute paths) and groups them by the role tag the
// gateway set ("primary" / "mmproj" / etc).
std::map<std::string, std::filesystem::path> group_by_role(
    const std::map<int, std::filesystem::path>& fetched_by_idx,
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
        std::scoped_lock lk(mu_);
        ++counts_[model_id_];
        total_.fetch_add(1, std::memory_order_relaxed);
    }
    ~ActiveGuard() {
        std::scoped_lock lk(mu_);
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

std::unique_ptr<pb::WorkerMessage> WorkerService::execute(const pb::HubMessage& job,
                                                          EmittedFn emit) {
    using HM = pb::HubMessage;
    std::unique_ptr<pb::WorkerMessage> out;
    try {
        out = execute_impl(job, std::move(emit));
    } catch (const std::exception& e) {
        // llama/ggml surface runtime device OOM as exceptions (ggml-vulkan
        // throws from allocation paths during load and decode), and execute
        // runs on worker threads where an escaping throw is std::terminate.
        // Fail the job with the frame its kind expects — a dead process here
        // leaves MASS redispatching the same job into the next crash.
        const std::string msg = std::string("unhandled exception: ") + e.what();
        spdlog::error("execute failed: kind={} {}", static_cast<int>(job.msg_case()), msg);
        switch (job.msg_case()) {
            case HM::kAssignJob:
                out = terminal_error(job.assign_job().job_id(), msg);
                break;
            case HM::kLoadModel:
                out = load_error(job.load_model().job_id(), job.load_model().model_id(), msg);
                break;
            case HM::kUnloadModel:
                out = unload_error(job.unload_model().job_id(), job.unload_model().model_id(), msg);
                break;
            case HM::kBenchmark: {
                out = std::make_unique<pb::WorkerMessage>();
                auto* br = out->mutable_benchmark();
                br->set_job_id(job.benchmark().job_id());
                br->set_error(msg);
                break;
            }
            case HM::kModelBenchmark:
                out =
                    bench_failure(job.model_benchmark().model_id(),
                                  classify_bench_failure(msg, /*context_alloc_failed=*/false), msg);
                break;
            default:
                break;  // fire-and-forget kinds have no error frame.
        }
    }
    // Every failure's text funnels through its terminal frame, so this one
    // check catches a lost device whichever path noticed it first — a
    // thrown load, a decode folded into a batch item error, a failed
    // unload. The flag is sticky by design (see device_lost()).
    if (out && mentions_device_loss(frame_error_text(*out)) &&
        !device_lost_.exchange(true, std::memory_order_relaxed)) {
        spdlog::error(
            "device loss detected; runner will restart the process after this frame ships");
    }
    return out;
}

std::unique_ptr<pb::WorkerMessage> WorkerService::run_job(
    const std::shared_ptr<ChatModel>& chat, const std::shared_ptr<EmbeddingModel>& embed,
    const std::string& model_id, const std::string& job_id, const std::string& payload,
    const EmittedFn& emit) {
    lcpp::Job decoded;
    if (!decoded.ParseFromString(payload)) {
        return terminal_error(job_id, "AssignJob: invalid payload encoding");
    }

    // Active accounting: a single-item job holds one job-scoped
    // guard. Batch jobs occupy up to pool_size() context slots at
    // once, so each in-flight item holds its own guard instead — a
    // job-scoped guard would report active=1 while N slots run, and
    // the heartbeat's pool − active would over-advertise capacity
    // by N−1, inviting over-dispatch from MASS. With per-item
    // guards, `active` equals slots actually in use (and the
    // heartbeat's active_jobs counts in-flight items, not jobs).
    const bool per_item_active =
        decoded.kind() == lcpp::JOB_KIND_BATCH_CHAT || decoded.kind() == lcpp::JOB_KIND_BATCH_EMBED;
    std::optional<ActiveGuard> job_guard;
    if (!per_item_active) {
        job_guard.emplace(active_mu_, active_per_model_, active_total_, model_id);
    }

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
            //
            // A failed Write means MASS is gone (restart, dropped
            // stream): nothing downstream can receive the result, so
            // generating further tokens only burns the GPU until
            // max_tokens. The flag folds into is_cancelled below —
            // both lambdas run on this job's generation thread, so a
            // plain bool suffices.
            bool emit_failed = false;
            ChatModel::OnTokenFn on_token;
            if (streaming) {
                on_token = [&](std::string_view piece) {
                    if (emit_failed) return;
                    lcpp::JobChunk delta;
                    auto* cc = delta.mutable_chat();
                    cc->set_role(lcpp::ROLE_ASSISTANT);
                    cc->set_content(std::string(piece));
                    pb::WorkerMessage msg;
                    auto* jr = msg.mutable_job_result();
                    jr->set_job_id(job_id);
                    delta.SerializeToString(jr->mutable_chunk());
                    if (!emit(msg)) {
                        emit_failed = true;
                        spdlog::warn("streaming write failed job_id={} — cancelling generation",
                                     job_id);
                    }
                };
            }

            // Cancellation: chat_completion_stream polls this
            // function between sampler steps. The set is shared
            // across all execute() calls, keyed by job_id, so
            // multiple in-flight jobs each see only their own
            // cancel signal. A worker-wide stop cancels everything,
            // and a failed streaming write cancels this job.
            auto is_cancelled = [this, &job_id, &emit_failed]() {
                return stopping_.load(std::memory_order_acquire) || emit_failed ||
                       is_job_cancel_requested(job_id);
            };
            auto result = chat->chat_completion_stream(parts->messages, parts->sampling, on_token,
                                                       is_cancelled);
            // Always clear the cancel marker after the call —
            // even on success, in case HubCancelJob arrived after
            // the job completed naturally. Stale entries would
            // never auto-clear otherwise.
            clear_job_cancel(job_id);
            if (!result) {
                // Cancellation flows the same way as any other
                // error: a terminal frame carrying the message
                // text. MASS tells the operator-induced case
                // apart by its own cancel-intent marker, not by
                // the wire string, so we keep the existing
                // "Chat: " prefix for consistency with real
                // failures.
                return terminal_error(job_id, "Chat: " + result.error().message);
            }

            // Build the terminal JobChunk → encode into
            // WorkerJobResult.completed.
            lcpp::JobChunk chunk;
            fill_chat_final(chunk.mutable_chat_final(), *result,
                            /*include_message=*/!streaming);

            auto out = std::make_unique<pb::WorkerMessage>();
            auto* jr = out->mutable_job_result();
            jr->set_job_id(job_id);
            chunk.SerializeToString(jr->mutable_completed()->mutable_final_response());
            return out;
        }
        case lcpp::JOB_KIND_EMBED: {
            if (!embed) return terminal_error(job_id, "Embed: not an embedding model");
            auto is_cancelled = [this, &job_id]() {
                return stopping_.load(std::memory_order_acquire) || is_job_cancel_requested(job_id);
            };
            auto vec = embed->embed(decoded.embed().input(), is_cancelled);
            clear_job_cancel(job_id);
            if (!vec) return terminal_error(job_id, "Embed: " + vec.error().message);

            lcpp::JobChunk chunk;
            auto* er = chunk.mutable_embed();
            er->set_id("cpp-emb-" +
                       std::to_string(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                          std::chrono::steady_clock::now().time_since_epoch())
                                          .count()));
            for (float v : *vec) er->add_embedding(v);

            auto out = std::make_unique<pb::WorkerMessage>();
            auto* jr = out->mutable_job_result();
            jr->set_job_id(job_id);
            chunk.SerializeToString(jr->mutable_completed()->mutable_final_response());
            return out;
        }
        case lcpp::JOB_KIND_BATCH_EMBED: {
            if (!embed) return terminal_error(job_id, "BatchEmbed: not an embedding model");
            const auto& inputs = decoded.batch_embed().inputs();

            // Inputs fan out across the model's context pool; more
            // threads than slots would only pile up in acquire_ctx.
            const auto max_conc = std::min<std::size_t>(
                static_cast<std::size_t>(inputs.size()),
                static_cast<std::size_t>(std::max<int32_t>(1, embed->pool_size())));
            auto run_item =
                [&](std::size_t i,
                    std::stop_token batch_abort) -> std::expected<std::vector<float>, std::string> {
                ActiveGuard item_guard(active_mu_, active_per_model_, active_total_, model_id);
                auto is_cancelled = [&]() {
                    return stopping_.load(std::memory_order_acquire) ||
                           batch_abort.stop_requested() || is_job_cancel_requested(job_id);
                };
                auto vec = embed->embed(inputs[static_cast<int>(i)], is_cancelled);
                if (!vec) return std::unexpected(vec.error().message);
                return std::move(*vec);
            };
            auto vecs =
                run_batch_items(static_cast<std::size_t>(inputs.size()), max_conc, run_item);
            clear_job_cancel(job_id);
            if (!vecs) {
                return terminal_error(job_id, "BatchEmbed item " +
                                                  std::to_string(vecs.error().index) + ": " +
                                                  vecs.error().error);
            }

            lcpp::JobChunk chunk;
            auto* br = chunk.mutable_batch_embed();
            br->set_id(monotonic_id("cpp-bemb-"));
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
        case lcpp::JOB_KIND_BATCH_CHAT: {
            if (!chat) return terminal_error(job_id, "BatchChat: not a chat model");
            const auto& items = decoded.batch_chat().items();
            if (items.empty()) return terminal_error(job_id, "BatchChat: items must be non-empty");

            // Items fan out across the model's context pool (see
            // max_conc rationale on the batch-embed case). The whole
            // batch is still one job: a single cancel aborts every
            // item, and the first item failure stops the remainder
            // via the batch stop_token. No per-token streaming for
            // batch chat.
            const auto max_conc = std::min<std::size_t>(
                static_cast<std::size_t>(items.size()),
                static_cast<std::size_t>(std::max<int32_t>(1, chat->pool_size())));
            auto run_item =
                [&](std::size_t i,
                    std::stop_token batch_abort) -> std::expected<lcpp::ChatFinal, std::string> {
                auto parts = to_chat_request(items[static_cast<int>(i)]);
                if (!parts) return std::unexpected(parts.error());
                ActiveGuard item_guard(active_mu_, active_per_model_, active_total_, model_id);
                auto is_cancelled = [&]() {
                    return stopping_.load(std::memory_order_acquire) ||
                           batch_abort.stop_requested() || is_job_cancel_requested(job_id);
                };
                auto result = chat->chat_completion_stream(parts->messages, parts->sampling,
                                                           /*on_token=*/{}, is_cancelled);
                if (!result) return std::unexpected(result.error().message);
                lcpp::ChatFinal cf;
                fill_chat_final(&cf, *result, /*include_message=*/true);
                return cf;
            };
            auto finals =
                run_batch_items(static_cast<std::size_t>(items.size()), max_conc, run_item);
            clear_job_cancel(job_id);
            if (!finals) {
                return terminal_error(job_id, "BatchChat item " +
                                                  std::to_string(finals.error().index) + ": " +
                                                  finals.error().error);
            }

            lcpp::JobChunk chunk;
            auto* br = chunk.mutable_batch_chat();
            br->set_id(monotonic_id("cpp-bchat-"));
            // Index-aligned with BatchChatJob.items (wire contract);
            // run_batch_items returns them in input order.
            for (auto& cf : *finals) *br->add_items() = std::move(cf);

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

std::unique_ptr<pb::WorkerMessage> WorkerService::execute_impl(const pb::HubMessage& job,
                                                               EmittedFn emit) {
    using HM = pb::HubMessage;
    switch (job.msg_case()) {
        case HM::kAssignJob: {
            const auto& aj = job.assign_job();
            const std::string& job_id = aj.job_id();
            const std::string& mid = aj.model_id();

            // Try chat first, then embedding. Maps are disjoint by load type.
            std::shared_ptr<ChatModel> chat;
            std::shared_ptr<EmbeddingModel> embed;
            {
                std::shared_lock lk(models_mu_);
                if (auto it = chat_models_.find(mid); it != chat_models_.end())
                    chat = it->second;
                else if (auto it2 = embed_models_.find(mid); it2 != embed_models_.end())
                    embed = it2->second;
            }
            if (!chat && !embed) {
                return terminal_error(job_id, "AssignJob: model not loaded: " + mid);
            }

            return run_job(chat, embed, mid, job_id, aj.payload(), emit);
        }

        case HM::kCancelJob: {
            const std::string& cancel_id = job.cancel_job().job_id();
            spdlog::debug("CancelJob received for {}", cancel_id);
            request_job_cancel(cancel_id);
            // Fire-and-forget on the wire: MASS doesn't expect an ack.
            // The terminal frame comes from the in-flight job thread when
            // chat_completion_stream observes the cancel between sampler
            // steps and returns ModelErrorCode::Cancelled.
            return nullptr;
        }

        case HM::kLoadModel: {
            const auto& req = job.load_model();
            const std::string& job_id = req.job_id();
            const std::string& mid = req.model_id();

            lcpp::LoadHints hints;
            if (!hints.ParseFromString(req.load_hints())) {
                return load_error(job_id, mid, "LoadModel: invalid load_hints encoding");
            }

            // No enabled devices → nothing to place the model on. Fail the
            // placement before the fetch — a multi-GB download for an
            // unplaceable load is pure waste. Computed once here; both load
            // kinds consume it below.
            auto whitelist = allowed_load_devices();
            if (whitelist.ids.empty()) {
                return load_error(job_id, mid, "LoadModel: no devices enabled for model loads");
            }

            // Fetch files via the Phase 4 fetcher (loopback / sha256 / retry).
            // Cancellable by worker shutdown AND by a CancelJob carrying this
            // load's job_id — a multi-GB download must not be immortal.
            std::vector<ModelFile> files;
            for (const auto& f : req.files()) files.push_back(to_fetch_file(f));
            auto fetch_cancelled = [this, &job_id]() {
                return fetch_cancel_.load(std::memory_order_acquire) ||
                       is_job_cancel_requested(job_id);
            };
            auto fetched = fetcher_.fetch_all(files, fetch_cancelled);
            clear_job_cancel(job_id);  // the fetch is the load's only cancellable phase
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
                auto cfg = to_chat_load_cfg(hints, by_role, default_vram_headroom_pct_);
                cfg.allowed_devices = std::move(whitelist.devices);
                cfg.device_ids = std::move(whitelist.ids);
                spdlog::info("loading chat model id={} path={}", mid, cfg.path.string());
                auto loaded = ChatModel::load(std::move(cfg));
                if (!loaded) {
                    return load_error(job_id, mid, "LoadModel(chat): " + loaded.error().message);
                }
                const int32_t pool = (*loaded)->pool_size();
                {
                    std::unique_lock lk(models_mu_);
                    if (chat_models_.try_emplace(mid, std::move(*loaded)).second) {
                        model_files_[mid] = loaded_model_file_keys(req);
                    }
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
            auto cfg = to_embed_load_cfg(hints, by_role, default_vram_headroom_pct_);
            cfg.allowed_devices = std::move(whitelist.devices);
            cfg.device_ids = std::move(whitelist.ids);
            spdlog::info("loading embedding model id={} path={}", mid, cfg.path.string());
            auto loaded = EmbeddingModel::load(std::move(cfg));
            if (!loaded) {
                return load_error(job_id, mid, "LoadModel(embed): " + loaded.error().message);
            }
            const int32_t pool = (*loaded)->pool_size();
            {
                std::unique_lock lk(models_mu_);
                if (embed_models_.try_emplace(mid, std::move(*loaded)).second) {
                    model_files_[mid] = loaded_model_file_keys(req);
                }
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
            const std::string& mid = req.model_id();
            std::unique_lock lk(models_mu_);
            if (auto it = chat_models_.find(mid); it != chat_models_.end()) {
                chat_models_.erase(it);
                model_files_.erase(mid);
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
                model_files_.erase(mid);
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
            if (s.all()) {
                set_enabled_devices(std::nullopt);
            } else {
                set_enabled_devices(
                    std::vector<std::string>(s.device_ids().begin(), s.device_ids().end()));
            }
            return nullptr;  // fire-and-forget, no terminal frame
        }

        case HM::kBenchmark: {
            const auto& b = job.benchmark();
            const std::string& job_id = b.job_id();
            const std::string& dev_id = b.device_id();

            // Per-device isolation: one failing device must not discard the
            // devices that already measured — the reply carries the partial
            // results plus an error naming each device that couldn't run.
            std::vector<BenchResult> results;
            std::vector<std::string> errors;
            auto run_device = [&](const std::string& id) {
                try {
                    if (auto r = bench_one(hardware_, id); r) {
                        results.push_back(std::move(*r));
                    } else {
                        errors.push_back(id + ": " + r.error().message);
                    }
                } catch (const std::exception& e) {
                    errors.push_back(id + ": " + e.what());
                }
            };
            if (dev_id.empty()) {
                for (const auto& d : hardware_.devices()) run_device(d.id);
            } else {
                run_device(dev_id);
            }

            auto out = std::make_unique<pb::WorkerMessage>();
            auto* br = out->mutable_benchmark();
            br->set_job_id(job_id);
            if (!errors.empty()) {
                std::string joined;
                for (const auto& e : errors) {
                    if (!joined.empty()) joined += "; ";
                    joined += e;
                }
                br->set_error(joined);
            }
            for (const auto& r : results) {
                auto* d = br->add_results();
                d->set_device_id(r.device_id);
                d->set_device_name(r.device_name);
                d->set_memory_gbs(r.memory_gbs);
                d->set_load_gbs(r.load_gbs);
                // The device's realised rate on the Q4_K matvec workload,
                // in FLOPS. Cosmetic: MASS shows it so an operator can
                // compare devices, and schedules from per-model
                // benchmarks instead. Nothing keys off the workload's
                // identity any more, so it is reported unlabelled.
                d->set_flops(r.compute_gflops * 1e9);
            }
            return out;
        }

        case HM::kModelBenchmark:
            return run_model_benchmark(job.model_benchmark());

        case HM::kEnrolled:
            // WorkerEnrolled is the enrollment handshake's job, consumed by the
            // runner before the receive loop starts. It must never reach here on
            // a normal connect (MASS sends it only on a fresh enrollment, once).
            spdlog::error("dropping unexpected WorkerEnrolled outside enrollment");
            return nullptr;

        case HM::MSG_NOT_SET:
            // No variant → no job_id to address a terminal frame to. An
            // error frame with an empty job_id is unroutable on the MASS
            // side, so log and drop instead of sending wire noise.
            spdlog::error("dropping HubMessage with no msg case set");
            return nullptr;
    }
    spdlog::error("dropping HubMessage with unknown msg case {}", static_cast<int>(job.msg_case()));
    return nullptr;
}

std::unique_ptr<pb::WorkerMessage> WorkerService::run_model_benchmark(
    const pb::HubModelBenchmark& req) {
    const std::string& mid = req.model_id();
    spdlog::info("model benchmark starting model_id={} files={} cost={}", mid, req.files_size(),
                 req.cost());

    lcpp::LoadHints hints;
    if (!hints.ParseFromString(req.load_hints())) {
        return bench_failure(mid, pb::MODEL_BENCHMARK_FAILURE_KIND_TRANSIENT,
                             "ModelBenchmark: invalid load_hints encoding");
    }

    // No enabled devices is an operator state, not a verdict on the model
    // — MASS must be free to re-bench once a device is re-enabled.
    auto whitelist = allowed_load_devices();
    if (whitelist.ids.empty()) {
        return bench_failure(mid, pb::MODEL_BENCHMARK_FAILURE_KIND_TRANSIENT,
                             "ModelBenchmark: no devices enabled for model loads");
    }

    // The files stay on disk afterwards: a bench is how a model reaches
    // this worker's cache, so the next load for it is a warm one. They
    // leave only via HubDeleteCacheFiles.
    std::vector<ModelFile> files;
    for (const auto& f : req.files()) files.push_back(to_fetch_file(f));
    auto fetch_cancelled = [this]() { return fetch_cancel_.load(std::memory_order_acquire); };
    auto fetched = fetcher_.fetch_all(files, fetch_cancelled);
    if (!fetched) {
        return bench_failure(mid, pb::MODEL_BENCHMARK_FAILURE_KIND_TRANSIENT,
                             "ModelBenchmark: " + fetched.error().message);
    }
    auto by_role = group_by_role(*fetched, req.files());

    // Memory floor to measure base_bytes against. MASS clears the device
    // set of idle residents before benching, so this reads the device's
    // own baseline rather than another model's footprint.
    const auto before_load = device_mem_snapshot(memory_tracked_devices(whitelist.devices));

    std::shared_ptr<ChatModel> chat;
    std::shared_ptr<EmbeddingModel> embed;
    if (hints.kind() == lcpp::LOAD_KIND_EMBEDDING) {
        auto cfg = to_embed_load_cfg(hints, by_role, default_vram_headroom_pct_);
        cfg.max_concurrent = 1;  // the bench prices the first slot, then one more
        cfg.allowed_devices = whitelist.devices;
        cfg.device_ids = whitelist.ids;
        auto loaded = EmbeddingModel::load(std::move(cfg));
        if (!loaded) {
            return bench_failure(mid, classify_load_failure(loaded.error()),
                                 "ModelBenchmark(embed): " + loaded.error().message);
        }
        embed = std::move(*loaded);
    } else {
        auto cfg = to_chat_load_cfg(hints, by_role, default_vram_headroom_pct_);
        cfg.max_concurrent = 1;
        cfg.allowed_devices = whitelist.devices;
        cfg.device_ids = whitelist.ids;
        auto loaded = ChatModel::load(std::move(cfg));
        if (!loaded) {
            return bench_failure(mid, classify_load_failure(loaded.error()),
                                 "ModelBenchmark(chat): " + loaded.error().message);
        }
        chat = std::move(*loaded);
    }

    // Streaming chunks are dropped: the bench's job_id is synthetic, so a
    // chunk frame would be unroutable on the MASS side. Returning true
    // keeps a streaming payload from reading the drop as a lost stream
    // and cancelling itself.
    const EmittedFn drop = [](const pb::WorkerMessage&) { return true; };
    const std::string job_id = monotonic_id("bench-");
    const auto t0 = std::chrono::steady_clock::now();
    auto result = run_job(chat, embed, mid, job_id, req.payload(), drop);
    const double elapsed_secs =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (!result || result->job_result().has_error()) {
        const std::string msg = result ? result->job_result().error().message()
                                       : std::string("payload produced no terminal frame");
        return bench_failure(mid, classify_bench_failure(msg, /*context_alloc_failed=*/false),
                             "ModelBenchmark: " + msg);
    }

    const ModelBenchProbe probe =
        chat ? chat->bench_probe(before_load) : embed->bench_probe(before_load);
    // Unload before replying: MASS lifts this worker's dispatch gate on
    // the reply frame, and the next load must not race the bench's own.
    chat.reset();
    embed.reset();

    if (!(probe.graph_secs > 0)) {
        // MASS divides a pool-size budget by graph_secs; a zero would be
        // a silent misconfiguration rather than a measurement.
        return bench_failure(mid, pb::MODEL_BENCHMARK_FAILURE_KIND_TRANSIENT,
                             "ModelBenchmark: calibration decode failed");
    }

    spdlog::info(
        "model benchmark done model_id={} elapsed={:.3f}s graph={:.3f}s base={} MiB per_slot={} "
        "MiB",
        mid, elapsed_secs, probe.graph_secs, probe.base_bytes / (1024LL * 1024),
        probe.per_slot_bytes / (1024LL * 1024));

    auto out = std::make_unique<pb::WorkerMessage>();
    auto* mb = out->mutable_model_benchmark();
    mb->set_model_id(mid);
    auto* m = mb->mutable_measurements();
    m->set_elapsed_secs(elapsed_secs);
    m->set_graph_secs(probe.graph_secs);
    m->set_base_bytes(probe.base_bytes);
    m->set_per_slot_bytes(probe.per_slot_bytes);
    return out;
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

void WorkerService::set_enabled_devices(std::optional<std::vector<std::string>> ids) {
    std::unique_lock lk(enabled_mu_);
    if (!ids) {
        enabled_devices_.reset();
    } else {
        // An empty set is a real state: NO devices enabled, new loads
        // rejected — distinct from nullopt's "all advertised devices".
        enabled_devices_.emplace(std::make_move_iterator(ids->begin()),
                                 std::make_move_iterator(ids->end()));
    }
    spdlog::info("set enabled devices: {}",
                 enabled_devices_ ? std::to_string(enabled_devices_->size()) : "<all>");
}

WorkerService::AllowedDevices WorkerService::allowed_load_devices() const {
    std::shared_lock lk(enabled_mu_);

    // Mirror Hardware::Hardware()'s ggml-backend walk so the (gpu:N →
    // device) mapping matches what MASS sees over the wire. Which of these
    // devices a load may occupy — including why CPU joins only a GPU-less
    // placement set — is enabled_placement_ids's contract.
    std::vector<ggml_backend_dev_t> gpu_devs;
    ggml_backend_dev_t cpu_dev = nullptr;
    const std::size_t n = ggml_backend_dev_count();
    for (std::size_t i = 0; i < n; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (!dev) continue;
        ggml_backend_dev_props props{};
        ggml_backend_dev_get_props(dev, &props);
        switch (props.type) {
            case GGML_BACKEND_DEVICE_TYPE_CPU:
                cpu_dev = dev;
                break;
            case GGML_BACKEND_DEVICE_TYPE_GPU:
            case GGML_BACKEND_DEVICE_TYPE_IGPU:
                gpu_devs.push_back(dev);
                break;
            default:
                break;
        }
    }

    AllowedDevices out;
    out.ids = enabled_placement_ids(enabled_devices_, static_cast<int>(gpu_devs.size()),
                                    cpu_dev != nullptr);
    if (!enabled_devices_) {
        // Unrestricted: leave devices empty (signal to llama.cpp = "use
        // everything") while ids enumerates the full placement set.
        return out;
    }
    for (const auto& id : out.ids) {
        // IDs are our own "gpu:N"/"cpu:0" formatting, so the parse is safe.
        out.devices.push_back(
            id == "cpu:0" ? cpu_dev : gpu_devs[static_cast<std::size_t>(std::stoi(id.substr(4)))]);
    }
    return out;
}

void WorkerService::request_stop() {
    stopping_.store(true, std::memory_order_release);
    fetch_cancel_.store(true, std::memory_order_release);
}

void WorkerService::shutdown() {
    spdlog::info("worker shutting down");
    request_stop();
    std::unique_lock lock(models_mu_);
    chat_models_.clear();
    embed_models_.clear();
    model_files_.clear();
}

bool WorkerService::is_job_cancel_requested(const std::string& job_id) const {
    std::scoped_lock lk(job_cancel_mu_);
    return cancel_requested_.contains(job_id);
}

void WorkerService::request_job_cancel(const std::string& job_id) {
    // Retention window for cancels that never match a job (job already
    // finished, or was never dispatched here). Long enough that a cancel
    // racing a slow dispatch still lands; short enough that MASS can't grow
    // the map without bound. The size cap is a second fence for a burst of
    // unmatched cancels inside one window.
    constexpr auto kCancelRetention = std::chrono::minutes(10);
    constexpr std::size_t kMaxPendingCancels = 4096;

    const auto now = std::chrono::steady_clock::now();
    std::scoped_lock lk(job_cancel_mu_);
    std::erase_if(cancel_requested_,
                  [&](const auto& e) { return now - e.second > kCancelRetention; });
    cancel_requested_.try_emplace(job_id, now);
    while (cancel_requested_.size() > kMaxPendingCancels) {
        auto oldest = cancel_requested_.begin();
        for (auto it = cancel_requested_.begin(); it != cancel_requested_.end(); ++it) {
            if (it->second < oldest->second) oldest = it;
        }
        cancel_requested_.erase(oldest);
    }
}

void WorkerService::clear_job_cancel(const std::string& job_id) {
    std::scoped_lock lk(job_cancel_mu_);
    cancel_requested_.erase(job_id);
}

}  // namespace mass_worker
