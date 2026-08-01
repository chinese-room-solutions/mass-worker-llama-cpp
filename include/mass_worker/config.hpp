#pragma once

#include <optional>
#include <string>

namespace mass_worker {

// Persistent local policy for the worker: the settings an operator chooses for
// THIS machine, distinct from the connection credentials (mass_url, token, ca)
// which live in the credentials file (see credentials.hpp). The interactive
// wizard writes this file; a service launch and the runner read it. Stored
// under service_data_dir() so it is operator-friendly to hand-edit and ship
// across a fleet.
//
// Precedence at launch is: explicit CLI flag > this file > compiled-in default.
// Each field is std::optional so "unset in the file" is distinguishable from
// "set to the zero value" — only set fields override, the rest fall through to
// the next source.

struct WorkerConfig {
    std::optional<std::string> name;         // --name
    std::optional<std::string> models_dir;   // --models-dir
    std::optional<std::string> gpu_backend;  // advisory: cuda|metal|vulkan|cpu
    std::optional<std::string> log_level;    // --log-level
    std::optional<std::string> log_file;     // --log-file
    std::optional<int> vram_headroom_pct;    // --vram-headroom-pct (1-100)
};

// The on-disk path of config.conf under data_dir.
[[nodiscard]] std::string config_path(const std::string& data_dir);

// Load config.conf from data_dir. Returns an all-unset WorkerConfig when the
// file is absent (first run) — that is not an error, the worker just uses its
// defaults. Returns nullopt only when the file exists but cannot be parsed, so
// the caller can refuse to launch with a corrupt config rather than silently
// ignoring operator intent.
[[nodiscard]] std::optional<WorkerConfig> load_config(const std::string& data_dir);

// Write config.conf to data_dir, creating the directory if needed. Only set
// (non-nullopt) fields are emitted, so a round-trip preserves "unset". Returns
// false on any filesystem failure.
[[nodiscard]] bool save_config(const std::string& data_dir, const WorkerConfig& cfg);

}  // namespace mass_worker
