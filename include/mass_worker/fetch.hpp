#pragma once

#include <atomic>
#include <expected>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

namespace mass_worker {

// Errors returned by the cache + fetch layer. Mapped to MASS error messages
// at the WorkerService boundary; never surface as exceptions to the gRPC
// handler.
enum class FetchErrorCode {
    InvalidFilename,         // filename has "..", absolute path, or is empty
    LoopbackUnreachable,     // local_path was set but file doesn't exist
    Sha256Mismatch,          // file present but hash doesn't match
    DownloadFailed,          // libcurl + retries exhausted
    NoSourceProvided,        // no local_path AND no URL
    DuplicateRole,           // two ModelFiles with the same role in one request
    NoPrimaryRole,           // request didn't include a PRIMARY file
    IoError,                 // mkdir / open / etc.
};

struct FetchError {
    FetchErrorCode code;
    std::string    message;
};

// One entry in a model-load request. Mirrors mass.v1.worker.ModelFile.
struct ModelFile {
    std::string                         filename;     // relative to models_dir
    std::string                         url;          // empty → loopback or already-cached
    std::string                         sha256;       // hex; empty → skip verify
    std::string                         local_path;   // non-empty → MASS signaled loopback
    std::map<std::string, std::string>  headers;      // extra HTTP headers (e.g. HF auth)
    int                                 role{0};      // wire enum value of mass.v1.worker.ModelFileRole (PRIMARY=1, MMPROJ=2)
};

// Fetcher downloads ModelFiles into models_dir, verifying sha256 when
// supplied. Already-present files are reused. Loopback files (local_path
// set) are validated and used in place. Thread-safe; call from anywhere.
class Fetcher {
public:
    explicit Fetcher(std::filesystem::path models_dir);

    // Fetch every file. Returns role → absolute on-disk path on success.
    [[nodiscard]] std::expected<std::map<int, std::filesystem::path>, FetchError>
    fetch_all(const std::vector<ModelFile>& files,
              std::atomic<bool>&            cancel) const;

    // For tests: validate a relative filename without touching the filesystem.
    [[nodiscard]] static std::expected<std::filesystem::path, FetchError>
    safe_rel_filename(std::string filename, const std::string& url);

    // Stream the file's sha256 and compare against `want_hex`. Empty `want_hex`
    // is a programmer error — caller should skip the call instead.
    [[nodiscard]] static std::expected<void, FetchError>
    verify_sha256(const std::filesystem::path& path, const std::string& want_hex);

private:
    [[nodiscard]] std::expected<std::filesystem::path, FetchError>
    fetch_one(const ModelFile& f, std::atomic<bool>& cancel) const;

    std::filesystem::path models_dir_;
};

}  // namespace mass_worker
