#pragma once

#include <cstdint>
#include <expected>
#include <filesystem>
#include <functional>
#include <map>
#include <string>
#include <vector>

namespace mass_worker {

// Errors returned by the cache + fetch layer. Mapped to MASS error messages
// at the WorkerService boundary; never surface as exceptions to the gRPC
// handler.
enum class FetchErrorCode : std::uint8_t {
    InvalidFilename,      // filename has "..", absolute path, or is empty
    LoopbackUnreachable,  // local_path was set but file doesn't exist
    Sha256Mismatch,       // file present but hash doesn't match
    DownloadFailed,       // libcurl + retries exhausted
    NoSourceProvided,     // no local_path AND no URL
    DuplicateRole,        // two ModelFiles with the same role in one request
    NoPrimaryRole,        // request didn't include a PRIMARY file
    IoError,              // mkdir / open / etc.
};

struct FetchError {
    FetchErrorCode code;
    std::string message;
};

// One entry in a model-load request. Mirrors mass.v1.worker.ModelFile.
struct ModelFile {
    std::string filename;                        // relative to models_dir
    std::string url;                             // empty → loopback or already-cached
    std::string sha256;                          // hex; empty → skip verify
    std::string local_path;                      // non-empty → MASS signaled loopback
    std::map<std::string, std::string> headers;  // extra HTTP headers (e.g. HF auth)
    int role{0};  // wire enum value of mass.v1.worker.ModelFileRole (PRIMARY=1, MMPROJ=2)
};

// FetchCancelFn is polled during downloads (per received chunk and between
// retry attempts). Returning true aborts the transfer with DownloadFailed
// "download cancelled". Empty function = non-cancellable. Lets callers
// compose several cancel sources (worker shutdown OR a CancelJob for the
// owning load job) without the fetch layer knowing about either.
using FetchCancelFn = std::function<bool()>;

// Whether another attempt could plausibly succeed after a failed download
// attempt. The retry ladder asks this before spending an attempt and a sleep.
//
// `http_status` is the response code the server sent, or 0 when the transfer
// died before one arrived (DNS, connect, reset, stall abort) — always worth
// retrying. A 4xx means the request itself was wrong (missing object, bad or
// expired token) and will be just as wrong on the next attempt, so retrying
// only burns the ladder and its sleeps; 408 (request timeout) and 429 (too many
// requests) are the two explicit "come back later" codes and are excepted.
//
// `write_errno` is the errno of the first short write to the temp file. ENOSPC
// is terminal: two more attempts would refill the same full disk after 3s of
// pointless sleeps, and the operator needs to read "disk full", not a third
// error that looks like a network problem.
//
// Pure; exposed for testing.
[[nodiscard]] bool download_failure_is_retryable(long http_status, int write_errno);

// Fetcher downloads ModelFiles into models_dir, verifying sha256 when
// supplied. Already-present files are reused. Loopback files (local_path
// set) are validated and used in place. Thread-safe; call from anywhere.
class Fetcher {
public:
    explicit Fetcher(std::filesystem::path models_dir);

    // Fetch every file. Returns role → absolute on-disk path on success.
    [[nodiscard]] std::expected<std::map<int, std::filesystem::path>, FetchError> fetch_all(
        const std::vector<ModelFile>& files, const FetchCancelFn& cancel) const;

    // Delete stale "*.downloading*" partials under models_dir. Call once at
    // startup: any partial found then is an orphan (no download can be in
    // flight yet), so removing them keeps the cache from silently
    // accumulating invisible junk. Best-effort; errors are skipped.
    void sweep_partials() const;

    // For tests: validate a relative filename without touching the filesystem.
    [[nodiscard]] static std::expected<std::filesystem::path, FetchError> safe_rel_filename(
        std::string filename, const std::string& url);

    // Stream the file's sha256 and compare against `want_hex`. Empty `want_hex`
    // is a programmer error — caller should skip the call instead.
    [[nodiscard]] static std::expected<void, FetchError> verify_sha256(
        const std::filesystem::path& path, const std::string& want_hex);

private:
    [[nodiscard]] std::expected<std::filesystem::path, FetchError> fetch_one(
        const ModelFile& f, const FetchCancelFn& cancel) const;

    std::filesystem::path models_dir_;
};

}  // namespace mass_worker
