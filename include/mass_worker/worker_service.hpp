#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "mass_worker/cache.hpp"
#include "mass_worker/chat_model.hpp"
#include "mass_worker/embedding_model.hpp"
#include "mass_worker/fetch.hpp"
#include "mass_worker/hardware.hpp"

// Forward declarations from generated proto.
namespace mass::v1::worker {
class HubMessage;
class WorkerJobResult;
class WorkerRegister;
class WorkerDeviceStats;
}  // namespace mass::v1::worker

namespace mass_worker {

// WorkerService is the top-level job dispatcher. Mirrors the responsibilities
// of internal/service/service.go in the Go worker:
//   - holds maps of loaded chat / embedding models keyed by fingerprint
//   - dispatches incoming HubMessage jobs to the right handler
//   - exposes Registration() / DeviceStats() / CacheFiles() for the heartbeat
//
// One instance per process. Thread-safe (read-heavy maps guarded by
// shared_mutex).
class WorkerService {
public:
    WorkerService(std::string id, std::string name, std::string models_dir);
    ~WorkerService();

    WorkerService(const WorkerService&) = delete;
    WorkerService& operator=(const WorkerService&) = delete;

    [[nodiscard]] std::unique_ptr<mass::v1::worker::WorkerRegister> registration() const;

    [[nodiscard]] std::vector<std::unique_ptr<mass::v1::worker::WorkerDeviceStats>>
    device_stats() const;

    // Forward-slash relative paths to every .gguf under models_dir. Reported
    // in the heartbeat so MASS can reconcile and send back delete requests
    // for stale files. In-progress download artefacts (".downloading-*") are
    // skipped — they appear here only once finalized.
    [[nodiscard]] std::vector<std::string> cache_files() const;

    [[nodiscard]] std::unique_ptr<mass::v1::worker::WorkerJobResult>
    execute(const mass::v1::worker::HubMessage& job);

    void delete_cache_files(const std::vector<std::string>& filenames);

    void shutdown();

private:
    const std::string id_;
    const std::string name_;
    const std::string models_dir_;

    Hardware hardware_;
    Cache    cache_;
    Fetcher  fetcher_;

    mutable std::shared_mutex models_mu_;
    // Loaded chat + embedding models keyed by MASS-supplied fingerprint
    // string. The fingerprint is the cache key used by MASS's scheduler —
    // same model file + config = same fingerprint = single load.
    std::unordered_map<std::string, std::shared_ptr<ChatModel>>      chat_models_;
    std::unordered_map<std::string, std::shared_ptr<EmbeddingModel>> embed_models_;
    // Atomic flag passed to fetch operations so shutdown can cancel
    // in-flight downloads.
    std::atomic<bool> fetch_cancel_{false};
};

}  // namespace mass_worker
