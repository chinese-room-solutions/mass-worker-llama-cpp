#pragma once

#include <atomic>
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
class HubMessage;
class WorkerMessage;
class WorkerRegister;
class WorkerHeartbeat;
class WorkerDeviceStats;
}  // namespace mass::v1::worker

namespace mass_worker {

// kRuntimeName is the wire identifier MASS uses to route this worker to its
// matching gateway (mass-runtime-llama-cpp). Must match the gateway's
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
    WorkerService(std::string id, std::string name, std::string models_dir);
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
    // or WorkerUnloadResult. nullptr means "no terminal frame" (used by
    // fire-and-forget DeleteCacheFiles).
    [[nodiscard]] std::unique_ptr<mass::v1::worker::WorkerMessage>
    execute(const mass::v1::worker::HubMessage& job, EmittedFn emit);

    void delete_cache_files(const std::vector<std::string>& filenames);

    // Replace the in-memory whitelist of devices allowed for new model
    // loads. Empty list = "all enabled" (the default state on first
    // connect, before MASS sends its post-Register snapshot). Already-
    // loaded models are unaffected.
    void set_enabled_devices(std::vector<std::string> ids);

    void shutdown();

private:
    const std::string id_;
    const std::string name_;
    const std::string models_dir_;

    Hardware hardware_;
    Cache    cache_;
    Fetcher  fetcher_;

    // Loaded models keyed by gateway-supplied model_id. The id is opaque to
    // MASS; we treat it as the dedup key — same model file + load hints =
    // same id = single load.
    mutable std::shared_mutex models_mu_;
    std::unordered_map<std::string, std::shared_ptr<ChatModel>>      chat_models_;
    std::unordered_map<std::string, std::shared_ptr<EmbeddingModel>> embed_models_;

    // Live count of in-flight jobs per model_id. Used to derive
    // available_capacity in heartbeats and active-jobs in loaded_models.
    mutable std::mutex                          active_mu_;
    std::unordered_map<std::string, int32_t>    active_per_model_;
    std::atomic<int32_t>                        active_total_{0};

    // Operator-controlled device whitelist. nullopt = "all advertised
    // devices enabled" (bootstrap state before MASS sends a post-Register
    // snapshot). Consulted at model-load time only.
    mutable std::shared_mutex                              enabled_mu_;
    std::optional<std::unordered_set<std::string>>         enabled_devices_;

    std::atomic<bool> fetch_cancel_{false};

    // Build the ggml device whitelist for model loads given current
    // enabled state. Empty result == nullopt enabled set (default = all);
    // otherwise the worker's hardware list filtered by the enabled set.
    [[nodiscard]] std::vector<ggml_backend_dev_t> allowed_load_devices() const;
};

}  // namespace mass_worker
