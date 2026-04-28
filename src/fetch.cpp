#include "mass_worker/fetch.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <thread>

#include <curl/curl.h>
#include <openssl/evp.h>
#include <spdlog/spdlog.h>

namespace mass_worker {

namespace fs = std::filesystem;

namespace {

// libcurl process-wide init. Idempotent; safe across worker restarts.
struct CurlGlobalInit {
    CurlGlobalInit()  { curl_global_init(CURL_GLOBAL_DEFAULT); }
    ~CurlGlobalInit() { curl_global_cleanup(); }
};
CurlGlobalInit g_curl_init;

// EVP context wrapped in unique_ptr so it can't leak on early return.
struct EvpDeleter { void operator()(EVP_MD_CTX* p) const noexcept { if (p) EVP_MD_CTX_free(p); } };
using EvpCtx = std::unique_ptr<EVP_MD_CTX, EvpDeleter>;

// curl_off_t-friendly write helper that streams to FILE* and updates the
// running sha256. Returns the number of bytes written (libcurl convention).
struct WriteCtx {
    std::FILE* fp;
    EVP_MD_CTX* hash;        // optional; may be null if caller doesn't want streaming hash
    std::atomic<bool>* cancel;
};
size_t write_cb(char* ptr, size_t size, size_t nmemb, void* user) {
    auto* ctx = static_cast<WriteCtx*>(user);
    if (ctx->cancel && ctx->cancel->load(std::memory_order_acquire)) {
        return 0;  // libcurl interprets a short write as an error → abort
    }
    const size_t n = size * nmemb;
    if (ctx->hash) EVP_DigestUpdate(ctx->hash, ptr, n);
    return std::fwrite(ptr, 1, n, ctx->fp);
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
            std::tolower(static_cast<unsigned char>(b[i]))) return false;
    }
    return true;
}

// Strip leading slashes from a forward-slash-normalized path.
std::string trim_left_slashes(std::string s) {
    size_t i = 0;
    while (i < s.size() && s[i] == '/') ++i;
    return s.substr(i);
}

// Returns a tempfile sibling for `dest`: `<dest>.downloading-<unique>`.
// Mirrors the Go worker's `WithTempSuffix(".downloading")` convention. The
// random suffix prevents collision with concurrent downloads of the same
// destination from different requests.
fs::path temp_sibling(const fs::path& dest) {
    using namespace std::chrono;
    const auto ns = duration_cast<nanoseconds>(
        high_resolution_clock::now().time_since_epoch()).count();
    auto p = dest;
    p += ".downloading-" + std::to_string(ns);
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

// Single download attempt with optional Range resume. Caller handles retry.
// `from_offset` is the byte offset the server should resume from (0 = full).
[[nodiscard]] std::expected<void, FetchError>
download_attempt(const std::string&                          url,
                 const std::map<std::string, std::string>&   headers,
                 const fs::path&                             tmp,
                 std::int64_t                                from_offset,
                 std::atomic<bool>&                          cancel) {
    auto* curl = curl_easy_init();
    if (!curl) {
        return std::unexpected(FetchError{FetchErrorCode::DownloadFailed,
                                          "curl_easy_init failed"});
    }

    std::FILE* fp = std::fopen(tmp.string().c_str(),
                               from_offset > 0 ? "ab" : "wb");
    if (!fp) {
        curl_easy_cleanup(curl);
        return std::unexpected(FetchError{FetchErrorCode::IoError,
            "could not open temp file: " + tmp.string()});
    }

    WriteCtx ctx{fp, /*hash=*/nullptr, &cancel};
    curl_easy_setopt(curl, CURLOPT_URL,            url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION,  write_cb);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA,      &ctx);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_FAILONERROR,    1L);  // 4xx/5xx → CURLE_HTTP_RETURNED_ERROR
    curl_easy_setopt(curl, CURLOPT_NOPROGRESS,     1L);
    curl_easy_setopt(curl, CURLOPT_TCP_KEEPALIVE,  1L);
    // 60s connect timeout; no transfer timeout (large GGUFs over slow links
    // legitimately take hours). Cancellation comes from the cancel flag.
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 60L);

    if (from_offset > 0) {
        curl_easy_setopt(curl, CURLOPT_RESUME_FROM_LARGE,
                         static_cast<curl_off_t>(from_offset));
    }

    struct curl_slist* hdr_list = nullptr;
    for (const auto& [k, v] : headers) {
        const std::string line = k + ": " + v;
        hdr_list = curl_slist_append(hdr_list, line.c_str());
    }
    if (hdr_list) curl_easy_setopt(curl, CURLOPT_HTTPHEADER, hdr_list);

    const CURLcode rc = curl_easy_perform(curl);

    if (hdr_list) curl_slist_free_all(hdr_list);
    std::fclose(fp);
    curl_easy_cleanup(curl);

    if (rc != CURLE_OK) {
        if (cancel.load(std::memory_order_acquire)) {
            return std::unexpected(FetchError{FetchErrorCode::DownloadFailed,
                                              "download cancelled"});
        }
        return std::unexpected(FetchError{FetchErrorCode::DownloadFailed,
            std::string("curl: ") + curl_easy_strerror(rc)});
    }
    return {};
}

}  // namespace

Fetcher::Fetcher(fs::path models_dir) : models_dir_(std::move(models_dir)) {}

std::expected<fs::path, FetchError>
Fetcher::safe_rel_filename(std::string filename, const std::string& url) {
    if (filename.empty() && !url.empty()) {
        filename = url_path(url);
    }
    // Normalize to forward slashes, strip leading.
    std::replace(filename.begin(), filename.end(), '\\', '/');
    filename = trim_left_slashes(std::move(filename));
    if (filename.empty()) {
        return std::unexpected(FetchError{FetchErrorCode::InvalidFilename,
            "ModelFile has neither filename nor parseable URL"});
    }

    // fs::path::lexically_normal collapses "a/b/../c" to "a/c", but doesn't
    // reject ".." that escapes the root. We do that explicitly below.
    fs::path p(filename);
    p = p.lexically_normal();
    const std::string s = p.generic_string();
    if (s == "." || s.empty()) {
        return std::unexpected(FetchError{FetchErrorCode::InvalidFilename,
            "invalid filename: " + filename});
    }
    if (s.find("..") != std::string::npos) {
        return std::unexpected(FetchError{FetchErrorCode::InvalidFilename,
            "filename contains traversal: " + filename});
    }
    if (p.is_absolute()) {
        return std::unexpected(FetchError{FetchErrorCode::InvalidFilename,
            "filename must be relative: " + filename});
    }
    return p;
}

std::expected<void, FetchError>
Fetcher::verify_sha256(const fs::path& path, const std::string& want_hex) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        return std::unexpected(FetchError{FetchErrorCode::IoError,
            "opening for sha256: " + path.string()});
    }

    EvpCtx ctx(EVP_MD_CTX_new());
    if (!ctx || EVP_DigestInit_ex(ctx.get(), EVP_sha256(), nullptr) != 1) {
        return std::unexpected(FetchError{FetchErrorCode::IoError,
                                          "EVP init failed"});
    }

    std::array<char, 64 * 1024> buf{};
    while (f) {
        f.read(buf.data(), buf.size());
        const auto n = f.gcount();
        if (n > 0) EVP_DigestUpdate(ctx.get(), buf.data(), static_cast<size_t>(n));
    }
    if (f.bad()) {
        return std::unexpected(FetchError{FetchErrorCode::IoError,
            "read error during sha256: " + path.string()});
    }

    unsigned char digest[EVP_MAX_MD_SIZE];
    unsigned int  digest_len = 0;
    if (EVP_DigestFinal_ex(ctx.get(), digest, &digest_len) != 1) {
        return std::unexpected(FetchError{FetchErrorCode::IoError,
                                          "EVP final failed"});
    }
    const std::string got = to_hex_lower(digest, digest_len);
    if (!ieq(got, want_hex)) {
        return std::unexpected(FetchError{FetchErrorCode::Sha256Mismatch,
            "sha256 mismatch for " + path.string() +
            ": have " + got + " want " + want_hex});
    }
    return {};
}

std::expected<fs::path, FetchError>
Fetcher::fetch_one(const ModelFile& f, std::atomic<bool>& cancel) const {
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

    // 3. Reuse if present (and sha256 matches when supplied).
    std::error_code ec;
    if (fs::exists(local, ec)) {
        if (!f.sha256.empty()) {
            if (auto v = verify_sha256(local, f.sha256); !v) {
                return std::unexpected(v.error());
            }
        }
        spdlog::debug("model file present: {}", local.string());
        return local;
    }

    // 4. Need to download. URL must be present.
    if (f.url.empty()) {
        return std::unexpected(FetchError{FetchErrorCode::NoSourceProvided,
            "model file " + rel->generic_string() +
            " missing both local copy and download URL"});
    }

    fs::create_directories(local.parent_path(), ec);
    if (ec) {
        return std::unexpected(FetchError{FetchErrorCode::IoError,
            "creating cache dir: " + ec.message()});
    }

    // 5. Download with retries + resume. Resume offset comes from the
    //    tempfile's existing size if a previous attempt left bytes behind.
    fs::path tmp = temp_sibling(local);
    spdlog::info("fetching model file url={} dest={}", f.url, local.string());

    constexpr int kMaxRetries = 3;
    for (int attempt = 0; attempt < kMaxRetries; ++attempt) {
        std::int64_t resume = 0;
        if (fs::exists(tmp, ec)) {
            resume = static_cast<std::int64_t>(fs::file_size(tmp, ec));
            if (ec) resume = 0;
        }

        auto r = download_attempt(f.url, f.headers, tmp, resume, cancel);
        if (r) break;
        if (cancel.load(std::memory_order_acquire)) {
            return std::unexpected(r.error());
        }
        if (attempt + 1 == kMaxRetries) return std::unexpected(r.error());
        spdlog::warn("download attempt {} failed: {} — retrying",
                     attempt + 1, r.error().message);
        std::this_thread::sleep_for(std::chrono::seconds(1 << attempt));  // 1s, 2s
    }

    // 6. Atomic-ish rename into place.
    fs::rename(tmp, local, ec);
    if (ec) {
        // Fall back to copy+remove if rename fails (rare, e.g. across volumes).
        fs::copy_file(tmp, local, fs::copy_options::overwrite_existing, ec);
        fs::remove(tmp);
        if (ec) {
            return std::unexpected(FetchError{FetchErrorCode::IoError,
                "finalising download: " + ec.message()});
        }
    }

    // 7. Final sha256 verify (if requested).
    if (!f.sha256.empty()) {
        if (auto v = verify_sha256(local, f.sha256); !v) {
            return std::unexpected(v.error());
        }
    }
    return local;
}

std::expected<std::map<int, fs::path>, FetchError>
Fetcher::fetch_all(const std::vector<ModelFile>& files,
                   std::atomic<bool>&             cancel) const {
    if (files.empty()) {
        return std::unexpected(FetchError{FetchErrorCode::NoSourceProvided,
                                          "no model files provided"});
    }
    constexpr int kRolePrimary = 1;  // mass.v1.worker.MODEL_FILE_ROLE_PRIMARY
    std::map<int, fs::path> out;
    for (const auto& f : files) {
        auto path = fetch_one(f, cancel);
        if (!path) return std::unexpected(path.error());
        if (out.contains(f.role)) {
            return std::unexpected(FetchError{FetchErrorCode::DuplicateRole,
                "duplicate model file role: " + std::to_string(f.role)});
        }
        out[f.role] = *path;
    }
    if (!out.contains(kRolePrimary)) {
        return std::unexpected(FetchError{FetchErrorCode::NoPrimaryRole,
            "no PRIMARY model file in request"});
    }
    return out;
}

}  // namespace mass_worker
