#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

namespace mass_worker {

// Cache owns the on-disk model file directory. Pure logic — no MASS-specific
// state — so it's easy to test independently of the full worker.
class Cache {
public:
    explicit Cache(std::filesystem::path models_dir);

    // List every .gguf under models_dir as forward-slash relative paths.
    // Skips ".downloading-*" tempfiles (download in progress).
    [[nodiscard]] std::vector<std::string> list_gguf() const;

    // Hard-delete each named file (forward-slash relative). Files in
    // `loaded_paths` are silently skipped — defence in depth against MASS
    // requesting a delete of a file backing a loaded model. Empty parent
    // directories are pruned up to (but not including) models_dir.
    void delete_files(const std::vector<std::string>& filenames,
                      const std::unordered_set<std::filesystem::path>& loaded_paths) const;

    // For tests: resolve a wire-format relative filename to an absolute path
    // under models_dir, rejecting traversal ("..", absolute paths). Returns
    // nullopt iff the input would escape models_dir.
    [[nodiscard]] std::optional<std::filesystem::path> safe_cache_path(
        const std::string& rel) const;

    [[nodiscard]] const std::filesystem::path& models_dir() const { return models_dir_; }

private:
    void prune_empty_parents(std::filesystem::path dir) const;

    std::filesystem::path models_dir_;
};

}  // namespace mass_worker
