#include "mass_worker/config.hpp"

#include <spdlog/spdlog.h>

#include <charconv>
#include <filesystem>
#include <fstream>
#include <sstream>

#include "mass_worker/fsutil.hpp"

namespace mass_worker {

namespace {

namespace fs = std::filesystem;

// Flat key=value, not YAML — so a name that doesn't imply a format we don't
// parse. See load_config/save_config below; the worker deliberately ships no
// YAML library for these few scalar settings.
constexpr const char* kConfigFile = "config.conf";

}  // namespace

std::string config_path(const std::string& data_dir) {
    return (fs::path(data_dir) / kConfigFile).string();
}

std::optional<WorkerConfig> load_config(const std::string& data_dir) {
    const fs::path path = fs::path(data_dir) / kConfigFile;
    std::ifstream f(path, std::ios::binary);
    if (!f) return WorkerConfig{};  // absent → defaults, not an error

    WorkerConfig cfg;
    std::string line;
    while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();  // CRLF

        // Skip blank lines and comments — the file is hand-editable.
        const auto first = line.find_first_not_of(" \t");
        if (first == std::string::npos || line[first] == '#') continue;

        const auto eq = line.find('=');
        if (eq == std::string::npos) {
            spdlog::error("config: malformed line in {}: {}", path.string(), line);
            return std::nullopt;
        }
        std::string key = line.substr(first, eq - first);
        // Trim trailing whitespace from the key.
        if (const auto end = key.find_last_not_of(" \t"); end != std::string::npos) {
            key.erase(end + 1);
        }
        std::string value = line.substr(eq + 1);

        if (key == "name")
            cfg.name = value;
        else if (key == "models_dir")
            cfg.models_dir = value;
        else if (key == "gpu_backend")
            cfg.gpu_backend = value;
        else if (key == "log_level")
            cfg.log_level = value;
        else if (key == "log_file")
            cfg.log_file = value;
        else if (key == "vram_headroom_pct") {
            int pct = 0;
            const auto* begin = value.data();
            const auto* end = begin + value.size();
            if (std::from_chars(begin, end, pct).ec != std::errc{}) {
                spdlog::error("config: vram_headroom_pct is not an integer in {}: {}",
                              path.string(), value);
                return std::nullopt;
            }
            cfg.vram_headroom_pct = pct;
        }
        // Unknown keys are ignored so a newer file stays loadable on an older
        // worker — additive evolution without a format bump.
    }
    return cfg;
}

bool save_config(const std::string& data_dir, const WorkerConfig& cfg) {
    std::error_code ec;
    fs::create_directories(data_dir, ec);
    if (ec) {
        spdlog::error("config: cannot create data dir {}: {}", data_dir, ec.message());
        return false;
    }

    // Flat key=value, one per line — same dependency-free format as the
    // credentials file. Only set fields are written so a round-trip preserves
    // "unset". Values never contain newlines (names/paths/levels).
    std::ostringstream body;
    body << "# mass-worker-llama-cpp configuration — local policy for this machine.\n"
         << "# Flat key=value (not YAML). Connection settings (server URL, token,\n"
         << "# CA) live in the credentials file, not this one. Edit by hand or\n"
         << "# re-run the installer (mass-worker-setup).\n";
    if (cfg.name) body << "name=" << *cfg.name << '\n';
    if (cfg.models_dir) body << "models_dir=" << *cfg.models_dir << '\n';
    if (cfg.gpu_backend) body << "gpu_backend=" << *cfg.gpu_backend << '\n';
    if (cfg.log_level) body << "log_level=" << *cfg.log_level << '\n';
    if (cfg.log_file) body << "log_file=" << *cfg.log_file << '\n';
    if (cfg.vram_headroom_pct) body << "vram_headroom_pct=" << *cfg.vram_headroom_pct << '\n';

    // Owner-only + atomic (temp+rename): config can carry the worker's name and
    // paths, and a crash mid-write shouldn't leave a half-file the parser then
    // rejects on next launch.
    const fs::path path = fs::path(data_dir) / kConfigFile;
    if (!fsutil::write_private_file(path.string(), body.str())) {
        spdlog::error("config: write failed for {}", path.string());
        return false;
    }
    return true;
}

}  // namespace mass_worker
