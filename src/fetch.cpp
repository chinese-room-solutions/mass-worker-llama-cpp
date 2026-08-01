#include "mass_worker/fetch.hpp"

#include <curl/curl.h>
#include <openssl/evp.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <set>
#include <sstream>
#include <thread>
#include <utility>

namespace mass_worker {

namespace fs = std::filesystem;

namespace {

// libcurl process-wide init. Idempotent; safe across worker restarts.
struct CurlGlobalInit {
    CurlGlobalInit() { curl_global_init(CURL_GLOBAL_DEFAULT); }
    ~CurlGlobalInit() { curl_global_cleanup(); }
    CurlGlobalInit(const CurlGlobalInit&) = delete;
    CurlGlobalInit& operator=(const CurlGlobalInit&) = delete;
};
// Must precede any thread that could touch libcurl, so a lazy local
// static is not an option; a throw here is unrecoverable startup failure.
// NOLINTNEXTLINE(cert-err58-cpp)
CurlGlobalInit g_curl_init;

// EVP context wrapped in unique_ptr so it can't leak on early return.
struct EvpDeleter {
    void operator()(EVP_MD_CTX* p) const noexcept {
        if (p) EVP_MD_CTX_free(p);
    }
};
using EvpCtx = std::unique_ptr<EVP_MD_CTX, EvpDeleter>;

// HTTP status that says "here is the byte range you asked for". Anything else in
// answer to a Range request means the server is not resuming.
constexpr long kHttpPartialContent = 206;

// Connect handshake budget. 20s is generous for TCP+TLS to any reachable host
// and hands control back to the retry ladder three times sooner than 60s did.
constexpr long kConnectTimeoutSec = 20;
// Stall abort: under 1 KiB/s for 120 consecutive seconds. There is deliberately
// no total transfer timeout (a multi-GB GGUF over a slow link legitimately takes
// hours), but a connection that died without TCP noticing must not hang the
// worker forever. A genuinely slow link still clears 1 KiB/s; two solid minutes
// below it is dead.
constexpr long kStallBytesPerSec = 1024;
constexpr long kStallSeconds = 120;

// curl_off_t-friendly write helper that streams to FILE* and updates the
// running sha256. Returns the number of bytes written (libcurl convention).
struct WriteCtx {
    std::FILE* fp{};
    EVP_MD_CTX* hash{};  // optional; may be null if caller doesn't want streaming hash
    const mass_worker::FetchCancelFn* cancel{};
    int write_errno{0};            // errno of the first short fwrite (ENOSPC detection)
    CURL* curl{};                  // for the first-chunk Range check below
    bool expect_partial{false};    // we sent a Range: only 206 may be appended
    bool range_ignored{false};     // server answered with the whole object instead
    bool checked_response{false};  // the check is once per transfer, not per chunk
};
size_t write_cb(char* ptr, size_t size, size_t nmemb, void* user) {
    auto* ctx = static_cast<WriteCtx*>(user);
    if (ctx->cancel && *ctx->cancel && (*ctx->cancel)()) {
        return 0;  // libcurl interprets a short write as an error → abort
    }
    // Verify the resume BEFORE a single byte lands. A server that ignores Range
    // answers with the WHOLE object, which appended to our partial file yields a
    // corrupt result that only a sha256 would catch — and f.sha256 is optional,
    // so it can be accepted silently. Abort instead; the caller discards the
    // partial and restarts from zero.
    //
    // libcurl catches the usual shape of this itself (a 200 with no
    // Content-Range → CURLE_RANGE_ERROR, handled below). This catches what it
    // lets through: a server that sends a Content-Range libcurl accepts
    // alongside a status that is not 206. Status 0 means a non-HTTP protocol
    // where the transport owns the offset and there is no status to check.
    if (ctx->expect_partial && !ctx->checked_response) {
        ctx->checked_response = true;
        long status = 0;
        curl_easy_getinfo(ctx->curl, CURLINFO_RESPONSE_CODE, &status);
        if (status != 0 && status != kHttpPartialContent) {
            ctx->range_ignored = true;
            return 0;
        }
    }
    const size_t n = size * nmemb;
    if (ctx->hash) EVP_DigestUpdate(ctx->hash, ptr, n);
    const size_t written = std::fwrite(ptr, 1, n, ctx->fp);
    if (written < n && ctx->write_errno == 0) {
        ctx->write_errno = errno;
    }
    return written;
}

// Returns lowercase hex of `bytes`. Used for sha256 comparison.
std::string to_hex_lower(const unsigned char* bytes, size_t n) {
    std::ostringstream ss;
    ss << std::hex << std::setfill('0');
    for (size_t i = 0; i < n; ++i) ss << std::setw(2) << static_cast<int>(bytes[i]);
    return ss.str();
}

bool ieq(const std::string& a, const std::string& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (std::tolower(static_cast<unsigned char>(a[i])) !=
            std::tolower(static_cast<unsigned char>(b[i])))
            return false;
    }
    return true;
}

// Strip leading slashes from a forward-slash-normalized path.
std::string trim_left_slashes(std::string s) {
    size_t i = 0;
    while (i < s.size() && s[i] == '/') ++i;
    return s.substr(i);
}

// A short, stable, filename-safe digest of a URL. Only has to separate the
// handful of URLs that could ever collide on one destination name, so 8 bytes
// of sha256 is ample; this is not a security boundary.
std::string short_url_key(const std::string& url) {
    EvpCtx ctx(EVP_MD_CTX_new());
    unsigned char digest[EVP_MAX_MD_SIZE];
    unsigned int len = 0;
    if (!ctx || EVP_DigestInit_ex(ctx.get(), EVP_sha256(), nullptr) != 1 ||
        EVP_DigestUpdate(ctx.get(), url.data(), url.size()) != 1 ||
        EVP_DigestFinal_ex(ctx.get(), digest, &len) != 1) {
        return "nokey";
    }
    return to_hex_lower(digest, std::min<unsigned int>(len, 8));
}

// Returns the tempfile sibling for `dest`: `<dest>.<url-key>.downloading`.
// Mirrors the Go worker's `WithTempSuffix(".downloading")` convention, keyed on
// the URL as well as the destination.
//
// Deterministic in both, so a later fetch of the same file from the same place
// finds the previous attempt's partial bytes and resumes via HTTP Range instead
// of starting over. The URL has to be in the key: two loads can map DIFFERENT
// URLs onto the same cache-relative filename, and keyed on the destination alone
// the second would resume from the first's bytes and produce a file that is
// neither. Orphans still can't accumulate — sweep_partials clears every
// ".downloading" at startup. Loads are serialized on the runner's single control
// thread, so two concurrent downloads of one destination can't race the file.
fs::path temp_sibling(const fs::path& dest, const std::string& url) {
    auto p = dest;
    p += "." + short_url_key(url) + ".downloading";
    return p;
}

// Parse the URL's path component (everything after host, up to '?') into a
// string suitable for use as a filename. Best-effort — libcurl's URL parser
// is heavy; we just slice the string. Returns empty on no-path.
std::string url_path(const std::string& url) {
    auto scheme_end = url.find("://");
    auto host_start = (scheme_end == std::string::npos) ? 0 : scheme_end + 3;
    auto path_start = url.find('/', host_start);
    if (path_start == std::string::npos) return {};
    auto query_start = url.find('?', path_start);
    return url.substr(path_start, query_start - path_start);
}

// A failed attempt, plus what the retry ladder needs to decide about it.
struct AttemptFailure {
    FetchError err;
    // Another attempt cannot help: a 4xx the server will repeat, a full disk, or
    // an operator cancellation.
    bool terminal{false};
    // The partial file is unusable — the server ignored the Range request, or
    // rejected it as unsatisfiable — so the next attempt must start at zero.
    bool discard_partial{false};
};

// Single download attempt with optional Range resume. Caller handles retry.
// `from_offset` is the byte offset the server should resume from (0 = full).
[[nodiscard]] std::expected<void, AttemptFailure> download_attempt(
    const std::string& url, const std::map<std::string, std::string>& headers, const fs::path& tmp,
    std::int64_t from_offset, const mass_worker::FetchCancelFn& cancel) {
    auto* curl = curl_easy_init();
    if (!curl) {
        return std::unexpected(AttemptFailure{
            .err = {FetchErrorCode::DownloadFailed, "curl_easy_init failed"}, .terminal = true});
    }

    // std::fopen is the portable Standard C call; MSVC's C4996 "unsafe, use
    // fopen_s" is a non-portable CRT nag (fopen_s changes the signature and
    // doesn't exist on GCC/Clang), so silence it just here under /W4 + /WX.
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4996)
#endif
    std::FILE* fp = std::fopen(tmp.string().c_str(), from_offset > 0 ? "ab" : "wb");
#ifdef _MSC_VER
#pragma warning(pop)
#endif
    if (!fp) {
        curl_easy_cleanup(curl);
        // A temp file we cannot open is a local problem (permissions, a missing
        // dir) that the next attempt would hit identically.
        return std::unexpected(AttemptFailure{
            .err = {FetchErrorCode::IoError, "could not open temp file: " + tmp.string()},
            .terminal = true});
    }

    struct FileCloser {
        void operator()(std::FILE* f) const noexcept {
            if (f) std::fclose(f);  // NOLINT(cppcoreguidelines-owning-memory)
        }
    };
    std::unique_ptr<std::FILE, FileCloser> file(fp);

    WriteCtx ctx{.fp = fp,
                 .hash = nullptr,
                 .cancel = &cancel,
                 .curl = curl,
                 .expect_partial = from_offset > 0};
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_cb);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &ctx);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_FAILONERROR, 1L);  // 4xx/5xx → CURLE_HTTP_RETURNED_ERROR
    curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 1L);
    curl_easy_setopt(curl, CURLOPT_TCP_KEEPALIVE, 1L);
    // Connect budget, plus a stall abort instead of a total transfer timeout —
    // see the constants for the reasoning. Cancellation still comes from the
    // cancel flag; these only catch a connection nobody is going to tell us
    // about.
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, kConnectTimeoutSec);
    curl_easy_setopt(curl, CURLOPT_LOW_SPEED_LIMIT, kStallBytesPerSec);
    curl_easy_setopt(curl, CURLOPT_LOW_SPEED_TIME, kStallSeconds);

    if (from_offset > 0) {
        curl_easy_setopt(curl, CURLOPT_RESUME_FROM_LARGE, static_cast<curl_off_t>(from_offset));
    }

    struct curl_slist* hdr_list = nullptr;
    for (const auto& [k, v] : headers) {
        std::string line = k;
        line += ": ";
        line += v;
        hdr_list = curl_slist_append(hdr_list, line.c_str());
    }
    if (hdr_list) curl_easy_setopt(curl, CURLOPT_HTTPHEADER, hdr_list);

    const CURLcode rc = curl_easy_perform(curl);
    // Read the status BEFORE cleanup. CURLOPT_FAILONERROR folds every 4xx and
    // 5xx into one CURLE_HTTP_RETURNED_ERROR, so this is the only thing that
    // tells a permanently-missing object from a server having a bad minute.
    long http_status = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_status);

    if (hdr_list) curl_slist_free_all(hdr_list);
    // Flush + close NOW: the rename/hash below must see complete bytes.
    file.reset();
    curl_easy_cleanup(curl);

    if (rc != CURLE_OK) {
        if (cancel && cancel()) {
            return std::unexpected(AttemptFailure{
                .err = {FetchErrorCode::DownloadFailed, "download cancelled"}, .terminal = true});
        }
        if (ctx.range_ignored || rc == CURLE_RANGE_ERROR) {
            return std::unexpected(AttemptFailure{
                .err = {FetchErrorCode::DownloadFailed,
                        "server did not honour the Range request (HTTP " +
                            std::to_string(http_status) +
                            "); discarding the partial download and restarting from zero"},
                .discard_partial = true});
        }
        // A short fwrite aborts the transfer as CURLE_WRITE_ERROR; surface
        // disk-full distinctly so the operator isn't chasing a network issue.
        if (ctx.write_errno == ENOSPC) {
            // Terminality still comes from the one policy function — this branch
            // only shapes a message the operator can act on.
            return std::unexpected(AttemptFailure{
                .err = {FetchErrorCode::IoError, "disk full (ENOSPC) writing " + tmp.string()},
                .terminal = !download_failure_is_retryable(http_status, ctx.write_errno)});
        }
        std::string msg = std::string("curl: ") + curl_easy_strerror(rc);
        if (http_status != 0) msg += " (HTTP " + std::to_string(http_status) + ")";
        return std::unexpected(AttemptFailure{
            .err = {FetchErrorCode::DownloadFailed, std::move(msg)},
            .terminal = !download_failure_is_retryable(http_status, ctx.write_errno)});
    }

    // Success, but a resume nobody confirmed. libcurl answers a 416 to a Range
    // request by DECLARING THE FILE ALREADY DOWNLOADED: it returns CURLE_OK
    // having transferred nothing, so the write callback never ran and never saw
    // a 206. Left alone, a truncated or stale partial is then renamed into place
    // as the finished model — silently, whenever f.sha256 is empty. A real 206
    // always carries at least one byte (an empty range is what 416 answers), so
    // "no bytes and no confirmation" is exactly this case. Discard and restart.
    if (from_offset > 0 && !ctx.checked_response) {
        return std::unexpected(AttemptFailure{
            .err = {FetchErrorCode::DownloadFailed,
                    "server did not confirm the resume (HTTP " + std::to_string(http_status) +
                        ", no partial content); discarding the partial download and restarting "
                        "from zero"},
            .discard_partial = true});
    }
    return {};
}

// One full download of `f` into `local`: the retry ladder over
// download_attempt, then the rename into place. `tmp` accumulates partial bytes
// across attempts so a retry resumes instead of starting over. The caller owns
// digest verification.
[[nodiscard]] std::expected<void, FetchError> download_and_place(const ModelFile& f,
                                                                 const fs::path& tmp,
                                                                 const fs::path& local,
                                                                 const FetchCancelFn& cancel) {
    std::error_code ec;
    constexpr int kMaxRetries = 3;
    for (int attempt = 0; attempt < kMaxRetries; ++attempt) {
        std::int64_t resume = 0;
        if (fs::exists(tmp, ec)) {
            resume = static_cast<std::int64_t>(fs::file_size(tmp, ec));
            if (ec) resume = 0;
        }

        auto r = download_attempt(f.url, f.headers, tmp, resume, cancel);
        if (r) break;
        if (cancel && cancel()) return std::unexpected(r.error().err);
        if (r.error().discard_partial) {
            spdlog::warn("{}", r.error().err.message);
            fs::remove(tmp, ec);
        }
        // A 4xx, a full disk, or a cancellation: three attempts and 3s of
        // sleeps would end exactly the same way.
        if (r.error().terminal) return std::unexpected(r.error().err);
        if (attempt + 1 == kMaxRetries) return std::unexpected(r.error().err);
        spdlog::warn("download attempt {} failed: {} — retrying", attempt + 1,
                     r.error().err.message);
        std::this_thread::sleep_for(std::chrono::seconds(1 << attempt));  // 1s, 2s
    }

    // Atomic-ish rename into place.
    fs::rename(tmp, local, ec);
    if (ec) {
        // Fall back to copy+remove if rename fails (rare, e.g. across volumes).
        fs::copy_file(tmp, local, fs::copy_options::overwrite_existing, ec);
        std::error_code rm_ec;
        fs::remove(tmp, rm_ec);  // error_code overload — this thread has no catch
        if (ec) {
            return std::unexpected(
                FetchError{FetchErrorCode::IoError, "finalising download: " + ec.message()});
        }
    }
    return {};
}

}  // namespace

bool download_failure_is_retryable(long http_status, int write_errno) {
    // The disk is full: nothing about the transfer was the problem.
    if (write_errno == ENOSPC) return false;
    // Nothing came back at all (DNS, connect, reset, stall abort).
    if (http_status == 0) return true;
    // The two codes that explicitly mean "come back later".
    if (http_status == 408 || http_status == 429) return true;
    // The rest of 4xx is the request being wrong, not the moment.
    return http_status < 400 || http_status >= 500;
}

Fetcher::Fetcher(fs::path models_dir) : models_dir_(std::move(models_dir)) {}

std::expected<fs::path, FetchError> Fetcher::safe_rel_filename(std::string filename,
                                                               const std::string& url) {
    if (filename.empty() && !url.empty()) {
        filename = url_path(url);
    }
    // Normalize to forward slashes, strip leading.
    std::ranges::replace(filename, '\\', '/');
    filename = trim_left_slashes(std::move(filename));
    if (filename.empty()) {
        return std::unexpected(FetchError{FetchErrorCode::InvalidFilename,
                                          "ModelFile has neither filename nor parseable URL"});
    }

    // Colons never appear in canonical cache keys and enable Windows-only
    // escapes that the checks below miss: "C:evil" is drive-relative (not
    // absolute, so is_absolute() passes) yet models_dir_ / "C:evil" REPLACES
    // the left operand with a C:-rooted path, and NTFS treats "name:stream"
    // as an alternate data stream. Mirror MASS's ValidateRelPath: reject on
    // every platform for deterministic behaviour.
    if (filename.find(':') != std::string::npos) {
        return std::unexpected(
            FetchError{FetchErrorCode::InvalidFilename,
                       "filename contains ':' (drive or stream hazard): " + filename});
    }

    // fs::path::lexically_normal collapses "a/b/../c" to "a/c", but doesn't
    // reject ".." that escapes the root. We do that explicitly below.
    fs::path p(filename);
    p = p.lexically_normal();
    const std::string s = p.generic_string();
    if (s == "." || s.empty()) {
        return std::unexpected(
            FetchError{FetchErrorCode::InvalidFilename, "invalid filename: " + filename});
    }
    if (s.find("..") != std::string::npos) {
        return std::unexpected(FetchError{FetchErrorCode::InvalidFilename,
                                          "filename contains traversal: " + filename});
    }
    if (p.is_absolute()) {
        return std::unexpected(
            FetchError{FetchErrorCode::InvalidFilename, "filename must be relative: " + filename});
    }
    return p;
}

std::expected<void, FetchError> Fetcher::verify_sha256(const fs::path& path,
                                                       const std::string& want_hex) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        return std::unexpected(
            FetchError{FetchErrorCode::IoError, "opening for sha256: " + path.string()});
    }

    EvpCtx ctx(EVP_MD_CTX_new());
    if (!ctx || EVP_DigestInit_ex(ctx.get(), EVP_sha256(), nullptr) != 1) {
        return std::unexpected(FetchError{FetchErrorCode::IoError, "EVP init failed"});
    }

    std::array<char, 64UL * 1024> buf{};
    while (f) {
        f.read(buf.data(), buf.size());
        const auto n = f.gcount();
        if (n > 0) EVP_DigestUpdate(ctx.get(), buf.data(), static_cast<size_t>(n));
    }
    if (f.bad()) {
        return std::unexpected(
            FetchError{FetchErrorCode::IoError, "read error during sha256: " + path.string()});
    }

    unsigned char digest[EVP_MAX_MD_SIZE];
    unsigned int digest_len = 0;
    if (EVP_DigestFinal_ex(ctx.get(), digest, &digest_len) != 1) {
        return std::unexpected(FetchError{FetchErrorCode::IoError, "EVP final failed"});
    }
    const std::string got = to_hex_lower(digest, digest_len);
    if (!ieq(got, want_hex)) {
        return std::unexpected(FetchError{
            FetchErrorCode::Sha256Mismatch,
            "sha256 mismatch for " + path.string() + ": have " + got + " want " + want_hex});
    }
    return {};
}

std::expected<fs::path, FetchError> Fetcher::fetch_one(const ModelFile& f,
                                                       const FetchCancelFn& cancel) const {
    // 1. Loopback shortcut: MASS told us the file is already on this host.
    if (!f.local_path.empty()) {
        std::error_code ec;
        if (!fs::exists(f.local_path, ec)) {
            return std::unexpected(FetchError{FetchErrorCode::LoopbackUnreachable,
                                              "loopback model file unreachable: " + f.local_path});
        }
        if (!f.sha256.empty()) {
            if (auto v = verify_sha256(f.local_path, f.sha256); !v) {
                return std::unexpected(v.error());
            }
        }
        spdlog::debug("loopback model file shared in place: {}", f.local_path);
        return fs::path(f.local_path);
    }

    // 2. Compute the cache-relative path safely.
    auto rel = safe_rel_filename(f.filename, f.url);
    if (!rel) return std::unexpected(rel.error());
    fs::path local = models_dir_ / *rel;

    // 3. Reuse if present (and sha256 matches when supplied). A corrupt
    //    cached file (hash mismatch) is deleted rather than returned as a
    //    permanent error — this load falls through to a fresh download.
    std::error_code ec;
    if (fs::exists(local, ec)) {
        bool reusable = true;
        if (!f.sha256.empty()) {
            if (auto v = verify_sha256(local, f.sha256); !v) {
                if (v.error().code != FetchErrorCode::Sha256Mismatch) {
                    return std::unexpected(v.error());
                }
                spdlog::warn("cached model file corrupt, deleting for refetch: {}",
                             v.error().message);
                fs::remove(local, ec);
                reusable = false;
            }
        }
        if (reusable) {
            spdlog::debug("model file present: {}", local.string());
            return local;
        }
    }

    // 4. Need to download. URL must be present.
    if (f.url.empty()) {
        return std::unexpected(FetchError{
            FetchErrorCode::NoSourceProvided,
            "model file " + rel->generic_string() + " missing both local copy and download URL"});
    }

    fs::create_directories(local.parent_path(), ec);
    if (ec) {
        return std::unexpected(
            FetchError{FetchErrorCode::IoError, "creating cache dir: " + ec.message()});
    }

    // 5. Download (with retries + resume), place, and verify. A post-download
    //    digest mismatch earns exactly one refetch from zero — the same
    //    generosity step 3 gives a corrupt *cached* file, and the same reasoning:
    //    a truncated or scrambled transfer is usually not repeatable. The second
    //    mismatch is the source, not the wire.
    const fs::path tmp = temp_sibling(local, f.url);
    spdlog::info("fetching model file url={} dest={}", f.url, local.string());

    constexpr int kDigestAttempts = 2;
    for (int pass = 0;; ++pass) {
        if (auto d = download_and_place(f, tmp, local, cancel); !d) {
            return std::unexpected(d.error());
        }
        if (f.sha256.empty()) return local;

        auto v = verify_sha256(local, f.sha256);
        if (v) return local;
        if (v.error().code != FetchErrorCode::Sha256Mismatch) return std::unexpected(v.error());

        // A mismatched download must not stay in the cache — it would be
        // "reused" (and re-rejected) forever. Removing it also guarantees the
        // refetch starts from zero: download_and_place resumes from `tmp`, which
        // the rename above consumed.
        std::error_code rm_ec;
        fs::remove(local, rm_ec);
        if (pass + 1 == kDigestAttempts) return std::unexpected(v.error());
        spdlog::warn("downloaded file failed its digest check, refetching from zero: {}",
                     v.error().message);
    }
}

void Fetcher::sweep_partials() const {
    std::error_code ec;
    if (models_dir_.empty() || !fs::is_directory(models_dir_, ec)) return;

    fs::recursive_directory_iterator it(models_dir_, fs::directory_options::skip_permission_denied,
                                        ec);
    if (ec) return;

    for (; it != fs::recursive_directory_iterator(); it.increment(ec)) {
        if (ec) {
            ec.clear();
            continue;
        }
        if (!it->is_regular_file(ec)) continue;
        // Substring, not suffix, so every scheme this repo has shipped is swept:
        // the current "<dest>.<url-key>.downloading", the previous
        // "<dest>.downloading", and the original "<dest>.downloading-<ns>".
        if (it->path().filename().string().find(".downloading") == std::string::npos) {
            continue;
        }
        std::error_code rm_ec;
        if (fs::remove(it->path(), rm_ec)) {
            spdlog::info("removed stale partial download: {}", it->path().string());
        }
    }
}

std::expected<std::map<int, fs::path>, FetchError> Fetcher::fetch_all(
    const std::vector<ModelFile>& files, const FetchCancelFn& cancel) const {
    if (files.empty()) {
        return std::unexpected(
            FetchError{FetchErrorCode::NoSourceProvided, "no model files provided"});
    }
    constexpr int kRolePrimary = 1;  // mass.v1.worker.MODEL_FILE_ROLE_PRIMARY
    // Returned map is keyed by *iteration index* so callers can join back
    // against the original request's parallel arrays. Role-uniqueness +
    // primary-presence are enforced via a separate set so duplicates are
    // still rejected without the map's keys becoming role values.
    std::map<int, fs::path> out;
    std::set<int> seen_roles;
    bool saw_primary = false;
    for (int idx = 0; std::cmp_less(idx, files.size()); ++idx) {
        const auto& f = files[static_cast<std::size_t>(idx)];
        auto path = fetch_one(f, cancel);
        if (!path) return std::unexpected(path.error());
        if (!seen_roles.insert(f.role).second) {
            return std::unexpected(
                FetchError{FetchErrorCode::DuplicateRole,
                           "duplicate model file role: " + std::to_string(f.role)});
        }
        if (f.role == kRolePrimary) saw_primary = true;
        out[idx] = *path;
    }
    if (!saw_primary) {
        return std::unexpected(
            FetchError{FetchErrorCode::NoPrimaryRole, "no PRIMARY model file in request"});
    }
    return out;
}

}  // namespace mass_worker
