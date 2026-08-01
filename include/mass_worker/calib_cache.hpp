#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "llama.h"

namespace mass_worker {

// Filename of the calibration cache inside the worker's models dir — it
// describes properties of those model files and dies with them.
inline constexpr const char* kCalibCacheFilename = ".calibration-cache";

// Cap on retained entries; oldest are dropped first on store. Bounded by
// models × device sets in practice, the cap is a corruption backstop.
inline constexpr std::size_t kCalibCacheMaxEntries = 64;

// One cached auto-ceiling calibration: the slot-0 full-ubatch decode time
// plus the per-device memory growth that decode caused. Both are needed
// to skip the measurement on a repeat load — the time derives the slot
// ceiling, the deltas re-seed the VRAM headroom watermark with the
// compute-buffer growth it would otherwise never observe.
struct CalibEntry {
    double graph_secs{0};
    std::vector<std::int64_t> slot_deltas;  // per headroom device, bytes
};

// The identity one measurement is valid for: the weights file (path +
// size + mtime — replacing the file re-measures), every context knob that
// shapes the calibration graph or its speed, and the devices it ran on.
// Any change produces a different key, so stale entries are never hit.
[[nodiscard]] std::string calib_cache_key(const std::filesystem::path& model_path,
                                          const llama_context_params& cparams,
                                          const std::vector<std::string>& device_ids);

// Look up key in the cache file. nullopt on any miss — absent file,
// unparsable line, unknown key. Never throws.
[[nodiscard]] std::optional<CalibEntry> calib_cache_lookup(const std::filesystem::path& file,
                                                           const std::string& key);

// Insert-or-replace key's entry and rewrite the file atomically. Best-
// effort: a failure logs and costs one re-measurement next load.
void calib_cache_store(const std::filesystem::path& file, const std::string& key,
                       const CalibEntry& entry);

}  // namespace mass_worker
