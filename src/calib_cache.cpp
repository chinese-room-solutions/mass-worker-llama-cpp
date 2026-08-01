#include "mass_worker/calib_cache.hpp"

#include <spdlog/spdlog.h>

#include <fstream>
#include <sstream>
#include <system_error>
#include <utility>

#include "mass_worker/fsutil.hpp"

namespace fs = std::filesystem;

namespace mass_worker {

namespace {

// Line format: `<graph_secs> <n_deltas> <d0> ... <dn-1> <key to EOL>`.
// The key goes last because it may contain spaces (model paths).
struct ParsedLine {
    std::string key;
    CalibEntry entry;
};

std::optional<ParsedLine> parse_line(const std::string& line) {
    std::istringstream in(line);
    ParsedLine out;
    std::size_t n = 0;
    if (!(in >> out.entry.graph_secs >> n)) return std::nullopt;
    if (out.entry.graph_secs <= 0 || n > kCalibCacheMaxEntries) return std::nullopt;
    out.entry.slot_deltas.resize(n);
    for (auto& d : out.entry.slot_deltas) {
        if (!(in >> d)) return std::nullopt;
    }
    if (!std::getline(in, out.key)) return std::nullopt;
    if (!out.key.empty() && out.key.front() == ' ') out.key.erase(0, 1);
    if (out.key.empty()) return std::nullopt;
    return out;
}

std::string format_line(const std::string& key, const CalibEntry& entry) {
    std::ostringstream out;
    out << entry.graph_secs << ' ' << entry.slot_deltas.size();
    for (const auto d : entry.slot_deltas) out << ' ' << d;
    out << ' ' << key << '\n';
    return std::move(out).str();
}

}  // namespace

std::string calib_cache_key(const fs::path& model_path, const llama_context_params& cparams,
                            const std::vector<std::string>& device_ids) {
    std::error_code ec;
    const auto size = fs::file_size(model_path, ec);
    const std::int64_t file_size = ec ? -1 : static_cast<std::int64_t>(size);
    ec.clear();
    const auto mtime = fs::last_write_time(model_path, ec);
    const std::int64_t file_stamp =
        ec ? -1 : static_cast<std::int64_t>(mtime.time_since_epoch().count());

    std::ostringstream key;
    key << model_path.generic_string() << '|' << file_size << '|' << file_stamp
        << "|ctx=" << cparams.n_ctx << "|batch=" << cparams.n_batch
        << "|ubatch=" << cparams.n_ubatch << "|threads=" << cparams.n_threads_batch
        << "|fa=" << static_cast<int>(cparams.flash_attn_type)
        << "|kt=" << static_cast<int>(cparams.type_k) << "|vt=" << static_cast<int>(cparams.type_v)
        << "|devs=";
    for (std::size_t i = 0; i < device_ids.size(); ++i) {
        if (i > 0) key << ',';
        key << device_ids[i];
    }
    return std::move(key).str();
}

std::optional<CalibEntry> calib_cache_lookup(const fs::path& file, const std::string& key) {
    if (file.empty() || key.empty()) return std::nullopt;
    std::ifstream in(file);
    if (!in) return std::nullopt;
    std::string line;
    while (std::getline(in, line)) {
        if (auto parsed = parse_line(line); parsed && parsed->key == key) {
            return std::move(parsed->entry);
        }
    }
    return std::nullopt;
}

void calib_cache_store(const fs::path& file, const std::string& key, const CalibEntry& entry) {
    if (file.empty() || key.empty() || key.find('\n') != std::string::npos ||
        !(entry.graph_secs > 0)) {
        return;
    }
    // The cache lives in models_dir, which only the fetch path creates —
    // a worker sharing the gateway's host loads absolute paths and never
    // fetches, so the directory may not exist yet.
    if (const auto parent = file.parent_path(); !parent.empty()) {
        std::error_code ec;
        fs::create_directories(parent, ec);
    }
    std::vector<ParsedLine> entries;
    if (std::ifstream in(file); in) {
        std::string line;
        while (std::getline(in, line)) {
            if (auto parsed = parse_line(line); parsed && parsed->key != key) {
                entries.push_back(std::move(*parsed));
            }
        }
    }
    entries.push_back({key, entry});
    if (entries.size() > kCalibCacheMaxEntries) {
        entries.erase(entries.begin(),
                      entries.end() - static_cast<std::ptrdiff_t>(kCalibCacheMaxEntries));
    }
    std::string content;
    for (const auto& e : entries) content += format_line(e.key, e.entry);
    if (!fsutil::write_private_file(file.string(), content)) {
        spdlog::warn("calibration cache: write failed path={}", file.string());
    }
}

}  // namespace mass_worker
