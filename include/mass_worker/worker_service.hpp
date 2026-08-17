#pragma once

#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "ggml-backend.h"
#include "mass_worker/cache.hpp"
#include "mass_worker/chat_model.hpp"
#include "mass_worker/embedding_model.hpp"
#include "mass_worker/fetch.hpp"
#include "mass_worker/hardware.hpp"

// Forward declarations from generated proto.
namespace mass::v1::worker {
class HubLoadModel;
class HubMessage;
class HubModelBenchmark;
class WorkerMessage;
class WorkerRegister;
class WorkerHeartbeat;
class WorkerDeviceStats;
}  // namespace mass::v1::worker

namespace mass_worker {

// enabled_placement_ids is the pure decision half of device whitelisting
// for new model loads: given the operator's enabled set (nullopt = every
// advertised device) and the advertised hardware shape, it returns the
// canonical device IDs ("gpu:N" in enumeration order, then "cpu:0") a new
// load may occupy.
//
// CPU is a placement target only when no GPU made the set: llama.cpp
// treats a listed CPU as a peer target and splits hybrid models across
// GPU+CPU, collapsing throughput. So CPU appears for default-all on a
// GPU-less box, or for a whitelist that selects no advertised GPU but
// explicitly enables "cpu:0".
//
// An empty result means NO devices are enabled — callers must fail the
// load, never fall back to unrestricted placement.
[[nodiscard]] std::vector<std::string> enabled_placement_ids(
    const std::optional<std::unordered_set<std::string>>& enabled, int gpu_count, bool has_cpu);

// loaded_model_file_keys extracts the store-relative cache keys backing a
// HubLoadModel: the ModelFile.filename values (primary + companions),
// verbatim — filename stays set on loopback local_path entries too. Entries
// with an empty filename are skipped: filename IS the cache key, so a
// keyless artifact has no identity to report. Echoed in
// LoadedModelStatus.files for MASS's cache reconciliation.
[[nodiscard]] std::vector<std::string> loaded_model_file_keys(
    const mass::v1::worker::HubLoadModel& req);

// resolve_max_concurrent picks the context-pool size for one load. Two
// sources now name the same quantity: HubLoadModel.max_concurrent, which
// the hub derives from the model's measured benchmark row, and the
// gateway-private LoadHints.max_concurrent the hub cannot decode.
//
// The hub's value wins whenever it states one (> 0), unconditionally —
// it is the side that also owns the memory gate a pinned pool is cleared
// against, so a hints value disagreeing with it would size the pool
// against numbers nothing checked. 0 means the hub said nothing and the
// hints value applies, and 0 from both is worker-side auto growth.
[[nodiscard]] int32_t resolve_max_concurrent(int32_t hub_value, int32_t hints_value);

// mentions_device_loss matches the spellings a lost GPU device produces
// across layers — vulkan.hpp exceptions ("ErrorDeviceLost"), the C API
// ("VK_ERROR_DEVICE_LOST"), and prose ("device lost") — case-insensitive,
// so the WorkerService::device_lost flag catches the error whichever
// layer's text reaches the frame.
[[nodiscard]] bool mentions_device_loss(std::string_view msg);

// mentions_allocation_failure matches the spellings an out-of-memory or
// refused allocation produces across layers — vulkan.hpp exceptions
// ("ErrorOutOfDeviceMemory"), ggml's own log prose ("failed to allocate"),
// the C++ runtime ("std::bad_alloc") — case-insensitive.
//
// It is the ONLY thing that may turn a model-benchmark failure into the
// INCAPABLE verdict, which MASS treats as permanent and never retries.
// So it matches allocation vocabulary and nothing else: a bare "oom" is
// deliberately absent (it is a substring of "headroom"), and any error
// it doesn't recognise stays TRANSIENT.
[[nodiscard]] bool mentions_allocation_failure(std::string_view msg);

// kRuntimeName is the wire identifier MASS uses to route this worker to its
// matching gateway (mass-runtime-gateway-llama-cpp). Must match the gateway's
// `runtime_name` and the URL prefix `/mass.llama-cpp.*`.
inline constexpr std::string_view kRuntimeName = "llama-cpp";

// EmittedFn is the callback used to push extra WorkerMessages back to MASS
// outside the normal HubMessage→WorkerJobResult round-trip — e.g. one
// streaming chunk per generated token. The runner's send mutex serialises
// concurrent invocations from inside service execution.
using EmittedFn = std::function<bool(const mass::v1::worker::WorkerMessage&)>;

// WorkerService is the top-level job dispatcher. It owns the loaded chat /
// embedding models keyed by gateway-supplied model_id, decodes opaque
// payload bytes into typed jobs, and tracks active-job counts so the
// heartbeat can report capacity honestly.
//
// One instance per process. Thread-safe (read-heavy maps guarded by
// shared_mutex).
class WorkerService {
public:
    // default_vram_headroom_pct is the worker-wide watermark used when a
    // LoadHints message arrives without an explicit override (1-100).
    WorkerService(std::string id, std::string name, std::string models_dir,
                  int32_t default_vram_headroom_pct = 75);
    ~WorkerService();

    WorkerService(const WorkerService&) = delete;
    WorkerService& operator=(const WorkerService&) = delete;

    // Build the WorkerRegister payload for the runner's first frame.
    [[nodiscard]] std::unique_ptr<mass::v1::worker::WorkerRegister> registration() const;

    // Build the WorkerHeartbeat payload (device stats + cache files +
    // capacity + loaded models).
    [[nodiscard]] std::unique_ptr<mass::v1::worker::WorkerHeartbeat> heartbeat() const;

    // Forward-slash relative paths to every .gguf under models_dir. Reported
    // in the heartbeat so MASS can reconcile and send back delete requests.
    // In-progress download artefacts (".downloading-*") are skipped.
    [[nodiscard]] std::vector<std::string> cache_files() const;

    // execute dispatches one HubMessage. The worker runner injects an
    // EmittedFn callback so streaming jobs can push WorkerJobResult chunks
    // before the final terminal frame.
    //
    // The returned WorkerMessage is the terminal frame — exactly one per
    // call: WorkerJobResult (chat/embed/tokenize), WorkerLoadModelResult,
    // or WorkerUnloadResult. nullptr means "no terminal frame": the
    // fire-and-forget messages (CancelJob, DeleteCacheFiles,
    // SetEnabledDevices) and malformed HubMessages with no msg case,
    // which carry no job_id a frame could be routed by (logged + dropped).
    //
    // Never throws: llama/ggml raise runtime failures (device OOM during
    // a load or decode) as C++ exceptions, and execute runs on worker
    // threads where an escaping throw is std::terminate. Exceptions are
    // folded into the message kind's error frame.
    [[nodiscard]] std::unique_ptr<mass::v1::worker::WorkerMessage> execute(
        const mass::v1::worker::HubMessage& job, const EmittedFn& emit);

    void delete_cache_files(const std::vector<std::string>& filenames);

    // Replace the in-memory whitelist of devices allowed for new model
    // loads. nullopt = every advertised device enabled (HubSetEnabledDevices
    // all=true, and the default state on first connect before MASS sends its
    // post-Register snapshot). A value = exactly that set; an EMPTY value =
    // no devices enabled, new loads are rejected. Already-loaded models are
    // unaffected.
    void set_enabled_devices(std::optional<std::vector<std::string>> ids);

    // device_lost turns true when a job or load error mentions a lost GPU
    // device (VK_ERROR_DEVICE_LOST and friends). It never resets: after a
    // device loss every llama call on that device throws, including
    // llama_free inside llama.cpp's noexcept destructors — an uncatchable
    // SIGABRT the moment a model unloads. The runner watches this flag and
    // exits the process (skipping those destructors) right after the
    // error frame is on the wire, so systemd restarts with a fresh Vulkan
    // instance instead of coredumping on the next idle eviction.
    [[nodiscard]] bool device_lost() const noexcept {
        return device_lost_.load(std::memory_order_relaxed);
    }

    // request_stop flips the service into stopping mode: every in-flight
    // job's cancel poll fires (generation aborts at the next sampler step,
    // batch jobs at the next chunk) and pending model fetches abort. Safe to
    // call from any thread; atomics only, no locks or logging — the runner's
    // watcher invokes it the moment a stop is requested, BEFORE the job
    // threads are joined, so the SCM's stop budget is never blown by an
    // in-flight generation.
    void request_stop();

    // Control-stream lifecycle, driven by the runner: begin_session() when a
    // stream is serving, end_session() the moment it closes. Work whose only
    // consumer is that stream — a device benchmark, tens of seconds long and
    // uncancellable otherwise — polls the flag and aborts, instead of running
    // to completion to produce a frame nothing can receive. Jobs keep their
    // own per-job cancellation; this is the session-wide edge.
    void begin_session();
    void end_session();

    // shutdown stops (request_stop) and releases the loaded models. Called
    // once, after the job threads have drained.
    void shutdown();

private:
    [[nodiscard]] std::unique_ptr<mass::v1::worker::WorkerMessage> execute_impl(
        const mass::v1::worker::HubMessage& job, const EmittedFn& emit);

    // run_job executes one gateway payload against an already-resolved
    // model and returns the terminal WorkerJobResult frame. Exactly one
    // of chat/embed is set. Shared by the AssignJob path and the model
    // benchmark, so a benched request runs through the same code as a
    // dispatched one — that identity is what makes the measured time
    // mean anything.
    [[nodiscard]] std::unique_ptr<mass::v1::worker::WorkerMessage> run_job(
        const std::shared_ptr<ChatModel>& chat, const std::shared_ptr<EmbeddingModel>& embed,
        const std::string& model_id, const std::string& job_id, const std::string& payload,
        const EmittedFn& emit);

    // run_model_benchmark answers a HubModelBenchmark: fetch, load with a
    // pool of one, run the payload against the wall clock, probe the
    // model (see ModelBenchProbe), unload, reply. The fetched files stay
    // on disk — benching is how a model reaches a worker's cache.
    //
    // The bench never registers the model in the loaded maps: it is not
    // dispatchable work, and a heartbeat advertising capacity for it
    // would invite the very dispatch the bench's exclusivity forbids.
    [[nodiscard]] std::unique_ptr<mass::v1::worker::WorkerMessage> run_model_benchmark(
        const mass::v1::worker::HubModelBenchmark& req);

    std::atomic<bool> device_lost_{false};

    const std::string id_;
    const std::string name_;
    const std::string models_dir_;
    const int32_t default_vram_headroom_pct_;

    Hardware hardware_;
    Cache cache_;
    Fetcher fetcher_;

    // Loaded models keyed by gateway-supplied model_id. The id is opaque to
    // MASS; we treat it as the dedup key — same model file + load hints =
    // same id = single load.
    mutable std::shared_mutex models_mu_;
    std::unordered_map<std::string, std::shared_ptr<ChatModel>> chat_models_;
    std::unordered_map<std::string, std::shared_ptr<EmbeddingModel>> embed_models_;

    // Store-relative cache keys backing each loaded model (the
    // ModelFile.filename values echoed from HubLoadModel), reported via
    // LoadedModelStatus.files in every heartbeat. Guarded by models_mu_
    // and kept in lockstep with the model maps: recorded on load, erased
    // on unload and shutdown, so an entry can never outlive its model.
    std::unordered_map<std::string, std::vector<std::string>> model_files_;

    // Live count of in-flight jobs per model_id. Used to derive
    // available_capacity in heartbeats and active-jobs in loaded_models.
    mutable std::mutex active_mu_;
    std::unordered_map<std::string, int32_t> active_per_model_;
    std::atomic<int32_t> active_total_{0};

    // Operator-controlled device whitelist. nullopt means "all
    // advertised devices enabled" — the bootstrap state before MASS
    // sends a post-Register snapshot, and the all=true wire state. An
    // empty set means NO devices are enabled (new loads are rejected).
    // Consulted at model-load time only.
    mutable std::shared_mutex enabled_mu_;
    std::optional<std::unordered_set<std::string>> enabled_devices_;

    std::atomic<bool> fetch_cancel_{false};

    // Set by request_stop()/shutdown(). ORed into every job's IsCancelledFn
    // so a worker stop cancels in-flight generation instead of letting it
    // run to max_tokens.
    std::atomic<bool> stopping_{false};

    // Set/cleared by begin_session()/end_session(). Starts open so a service
    // driven without a runner (tests, tools) behaves as if it had a consumer.
    std::atomic<bool> session_open_{true};

    // Per-job cancellation requests from MASS via HubCancelJob, keyed by
    // job_id with the arrival time. chat_completion_stream polls
    // is_job_cancel_requested() between sampler steps; on cancel it exits
    // with ModelErrorCode::Cancelled and execute() emits the terminal
    // "cancelled by operator" frame.
    //
    // Entries are added by execute()'s kCancelJob handler and erased by
    // clear_job_cancel() when the matching job's execute() path finishes.
    // A cancel with no matching job (already completed, or never
    // dispatched) has nothing to erase it, so request_job_cancel() sweeps
    // entries older than kCancelRetention and enforces kMaxPendingCancels,
    // keeping the map bounded no matter what MASS sends.
    mutable std::mutex job_cancel_mu_;
    std::unordered_map<std::string, std::chrono::steady_clock::time_point> cancel_requested_;

    // is_job_cancel_requested returns true iff a HubCancelJob has arrived
    // for job_id and no completion has cleared it yet.
    [[nodiscard]] bool is_job_cancel_requested(const std::string& job_id) const;

    // request_job_cancel marks job_id as cancel-requested and sweeps
    // expired/excess entries. Idempotent: re-sending HubCancelJob keeps the
    // original arrival time.
    void request_job_cancel(const std::string& job_id);

    // clear_job_cancel removes job_id from the cancel-requested map; called
    // when the job's execute() path completes (matched or not).
    void clear_job_cancel(const std::string& job_id);

    // Paired result of allowed_load_devices(): the ggml-backend device
    // pointers handed to mparams.devices, alongside their canonical IDs
    // ("gpu:N" / "cpu:0") for emission on heartbeat. When the operator
    // hasn't restricted anything, devices is empty (signal to llama.cpp
    // = "use everything") while ids enumerates the full placement set.
    // An empty ids list means NO devices are enabled — the load must be
    // rejected, never fall through to unrestricted placement.
    struct AllowedDevices {
        std::vector<ggml_backend_dev_t> devices;
        std::vector<std::string> ids;
    };

    // Build the ggml device whitelist for model loads given current
    // enabled state, plus the canonical IDs MASS scores against. The
    // enabled-set → placement decision lives in enabled_placement_ids;
    // this maps the surviving IDs back to backend device pointers.
    [[nodiscard]] AllowedDevices allowed_load_devices() const;
};

}  // namespace mass_worker
