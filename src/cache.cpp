#include "mass_worker/cache.hpp"

#include <algorithm>
#include <cctype>
#include <system_error>

#include <spdlog/spdlog.h>

namespace mass_worker {

namespace fs = std::filesystem;

namespace {

bool ext_is_gguf(const fs::path& p) {
    auto e = p.extension().string();
    if (e.size() != 5 || e[0] != '.') return false;
    // Case-insensitive match for ".gguf".
    static constexpr char want[] = {'g', 'g', 'u', 'f'};
    for (size_t i = 0; i < 4; ++i) {
        if (std::tolower(static_cast<unsigned char>(e[i + 1])) != want[i]) return false;
    }
    return true;
}

}  // namespace

Cache::Cache(fs::path models_dir) : models_dir_(std::move(models_dir)) {}

std::vector<std::string> Cache::list_gguf() const {
    std::vector<std::string> out;
    if (models_dir_.empty()) return out;

    std::error_code ec;
    if (!fs::exists(models_dir_, ec) || !fs::is_directory(models_dir_, ec)) {
        return out;
    }

    fs::recursive_directory_iterator it(models_dir_,
                                        fs::directory_options::skip_permission_denied,
                                        ec);
    if (ec) {
        spdlog::warn("walking models_dir {} failed: {}", models_dir_.string(), ec.message());
        return out;
    }

    for (; it != fs::recursive_directory_iterator(); it.increment(ec)) {
        if (ec) {
            spdlog::warn("iter error in {}: {}", models_dir_.string(), ec.message());
            ec.clear();
            continue;
        }
        if (!it->is_regular_file(ec)) continue;
        const std::string name = it->path().filename().string();
        if (name.starts_with(".downloading-")) continue;
        if (!ext_is_gguf(it->path())) continue;

        auto rel = fs::relative(it->path(), models_dir_, ec);
        if (ec) continue;
        std::string s = rel.generic_string();
        if (!s.empty()) out.push_back(std::move(s));
    }
    return out;
}

std::optional<fs::path> Cache::safe_cache_path(const std::string& rel) const {
    if (rel.empty()) return std::nullopt;

    std::string s = rel;
    std::replace(s.begin(), s.end(), '\\', '/');
    // Strip leading slashes — the wire format is relative, but be defensive.
    size_t i = 0;
    while (i < s.size() && s[i] == '/') ++i;
    s.erase(0, i);
    if (s.empty()) return std::nullopt;

    fs::path p(s);
    p = p.lexically_normal();
    const std::string norm = p.generic_string();
    if (norm == "." || norm.empty()) return std::nullopt;
    if (norm.find("..") != std::string::npos) return std::nullopt;
    if (p.is_absolute()) return std::nullopt;

    return models_dir_ / p;
}

void Cache::delete_files(const std::vector<std::string>&                  filenames,
                         const std::unordered_set<fs::path>&              loaded_paths) const {
    if (filenames.empty() || models_dir_.empty()) return;

    for (const auto& rel : filenames) {
        auto abs = safe_cache_path(rel);
        if (!abs) {
            spdlog::warn("refusing delete: path escapes models_dir: {}", rel);
            continue;
        }
        if (loaded_paths.contains(*abs)) {
            spdlog::debug("skipping delete: file is part of a loaded model: {}",
                          abs->string());
            continue;
        }

        std::error_code ec;
        if (!fs::exists(*abs, ec)) continue;
        if (!fs::remove(*abs, ec) && ec) {
            spdlog::warn("delete cache file {}: {}", abs->string(), ec.message());
            continue;
        }
        spdlog::info("deleted cache file (reconcile): {}", abs->string());
        prune_empty_parents(abs->parent_path());
    }
}

void Cache::prune_empty_parents(fs::path dir) const {
    while (true) {
        if (dir == models_dir_ || dir == dir.parent_path()) return;

        std::error_code ec;
        fs::directory_iterator it(dir, ec);
        if (ec) return;
        if (it != fs::directory_iterator()) return;  // not empty

        if (!fs::remove(dir, ec)) return;
        dir = dir.parent_path();
    }
}

}  // namespace mass_worker
