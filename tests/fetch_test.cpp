#include "mass_worker/fetch.hpp"

#include <gtest/gtest.h>

// Sockets for the scripted HTTP origin below. Winsock ahead of anything that
// could pull in WinSock 1, matching main.cpp's ordering.
#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "mass_worker/cache.hpp"

namespace fs = std::filesystem;

namespace {

// The two-line platform gap between BSD sockets and Winsock.
#ifdef _WIN32
using socket_t = SOCKET;  // socklen_t comes from ws2tcpip.h
using iolen_t = int;      // send/recv take an int length on Winsock, size_t on POSIX
constexpr socket_t kBadSocket = INVALID_SOCKET;
void close_socket(socket_t s) {
    if (s != kBadSocket) ::closesocket(s);
}
// One WSAStartup for the whole test binary; Winsock refuses to socket() without.
const bool g_winsock_ready = [] {
    WSADATA d;
    return WSAStartup(MAKEWORD(2, 2), &d) == 0;
}();
#else
using socket_t = int;
using iolen_t = std::size_t;
constexpr socket_t kBadSocket = -1;
void close_socket(socket_t s) {
    if (s != kBadSocket) ::close(s);
}
#endif

// Disposable temp dir for tests. RAII cleanup.
class TempDir {
public:
    TempDir(const TempDir&) = delete;
    TempDir& operator=(const TempDir&) = delete;
    TempDir() {
        std::random_device rd;
        path_ = fs::temp_directory_path() / ("mass-worker-test-" + std::to_string(rd()));
        fs::create_directories(path_);
    }
    ~TempDir() {
        std::error_code ec;
        fs::remove_all(path_, ec);
    }
    [[nodiscard]] const fs::path& path() const { return path_; }

private:
    fs::path path_;
};

// Pre-computed SHA256("hello world\n") — for an 11-byte payload that's
// trivial to write and predictable in tests.
constexpr const char* kHelloSha256 =
    "a948904f2f0f479b8f8197694b30184b0d2ed1c1cd2a1ec0fb85d299a192a447";

void write_file(const fs::path& p, const std::string& content) {
    fs::create_directories(p.parent_path());
    std::ofstream f(p, std::ios::binary);
    f << content;
}

TEST(SafeRelFilenameTest, AcceptsSimplePath) {
    auto r = mass_worker::Fetcher::safe_rel_filename("models/Qwen-7B.gguf", "");
    ASSERT_TRUE(r) << "expected success";
    EXPECT_EQ(r->generic_string(), "models/Qwen-7B.gguf");
}

TEST(SafeRelFilenameTest, NormalizesBackslashes) {
    auto r = mass_worker::Fetcher::safe_rel_filename("models\\Qwen.gguf", "");
    ASSERT_TRUE(r);
    EXPECT_EQ(r->generic_string(), "models/Qwen.gguf");
}

TEST(SafeRelFilenameTest, RejectsTraversal) {
    auto r = mass_worker::Fetcher::safe_rel_filename("../etc/passwd", "");
    ASSERT_FALSE(r);
    EXPECT_EQ(r.error().code, mass_worker::FetchErrorCode::InvalidFilename);
}

TEST(SafeRelFilenameTest, RejectsAbsolutePath) {
    // Use a leading-slash path that survives normalisation but is obviously
    // absolute on POSIX. Windows uses C:\ etc — same treatment via the
    // is_absolute check, but writing the test cross-platform requires a
    // path that's absolute on the host. POSIX-style works on both because
    // fs::path treats leading slashes as root-relative on POSIX and rejects
    // on Windows-style absolutes too via lexically_normal.
    //
    // The simpler invariant: an empty input after stripping leading slashes
    // is rejected, so a path that's _only_ slashes hits InvalidFilename.
    auto r = mass_worker::Fetcher::safe_rel_filename("///", "");
    ASSERT_FALSE(r);
}

// Colons enable Windows-only escapes the other checks miss: "C:evil" is
// drive-relative (passes is_absolute()) yet joining it onto models_dir_
// replaces the left operand with a C:-rooted path, and NTFS parses
// "name:stream" as an alternate data stream. Rejected on every platform.
TEST(SafeRelFilenameTest, RejectsColonAnywhere) {
    for (const auto* bad : {"C:evil", "foo:bar", "note.md:stream", "C:/evil", "dir/C:evil"}) {
        auto r = mass_worker::Fetcher::safe_rel_filename(bad, "");
        ASSERT_FALSE(r) << "expected rejection of " << bad;
        EXPECT_EQ(r.error().code, mass_worker::FetchErrorCode::InvalidFilename) << bad;
    }
}

TEST(SafeRelFilenameTest, FallsBackToUrlPath) {
    auto r =
        mass_worker::Fetcher::safe_rel_filename("", "https://hf.co/repo/resolve/main/Qwen-7B.gguf");
    ASSERT_TRUE(r);
    EXPECT_EQ(r->generic_string(), "repo/resolve/main/Qwen-7B.gguf");
}

TEST(SafeRelFilenameTest, RejectsEmptyBoth) {
    auto r = mass_worker::Fetcher::safe_rel_filename("", "");
    ASSERT_FALSE(r);
}

TEST(VerifySha256Test, MatchesExpected) {
    TempDir td;
    auto p = td.path() / "hello.txt";
    write_file(p, "hello world\n");

    auto r = mass_worker::Fetcher::verify_sha256(p, kHelloSha256);
    EXPECT_TRUE(r) << (r ? "" : r.error().message);
}

TEST(VerifySha256Test, AcceptsUppercaseHex) {
    TempDir td;
    auto p = td.path() / "hello.txt";
    write_file(p, "hello world\n");

    std::string upper(kHelloSha256);
    for (auto& c : upper) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    auto r = mass_worker::Fetcher::verify_sha256(p, upper);
    EXPECT_TRUE(r);
}

TEST(VerifySha256Test, RejectsMismatch) {
    TempDir td;
    auto p = td.path() / "hello.txt";
    write_file(p, "hello world\n");

    auto r = mass_worker::Fetcher::verify_sha256(p, std::string(64, '0'));
    ASSERT_FALSE(r);
    EXPECT_EQ(r.error().code, mass_worker::FetchErrorCode::Sha256Mismatch);
}

TEST(VerifySha256Test, ReportsMissingFile) {
    TempDir td;
    auto r = mass_worker::Fetcher::verify_sha256(td.path() / "nope.bin", kHelloSha256);
    ASSERT_FALSE(r);
    EXPECT_EQ(r.error().code, mass_worker::FetchErrorCode::IoError);
}

TEST(FetcherTest, LoopbackPathReusedInPlace) {
    TempDir td;
    auto loopback = td.path() / "loopback.gguf";
    write_file(loopback, "hello world\n");

    mass_worker::Fetcher fetcher(td.path() / "cache");
    mass_worker::ModelFile f{
        .filename = "ignored.gguf",
        .url = "",
        .sha256 = kHelloSha256,
        .local_path = loopback.string(),
        .role = 1,  // PRIMARY
    };
    auto r = fetcher.fetch_all({f}, {});
    ASSERT_TRUE(r) << (r ? "" : r.error().message);
    EXPECT_EQ((*r)[0], loopback);
}

TEST(FetcherTest, LoopbackMissingFileFails) {
    TempDir td;
    mass_worker::Fetcher fetcher(td.path() / "cache");
    mass_worker::ModelFile f{
        .local_path = (td.path() / "nope.gguf").string(),
        .role = 1,
    };
    auto r = fetcher.fetch_all({f}, {});
    ASSERT_FALSE(r);
    EXPECT_EQ(r.error().code, mass_worker::FetchErrorCode::LoopbackUnreachable);
}

TEST(FetcherTest, ExistingCachedFileReusedWhenShaMatches) {
    TempDir td;
    auto cache = td.path() / "cache";
    auto cached = cache / "model.gguf";
    write_file(cached, "hello world\n");

    mass_worker::Fetcher fetcher(cache);
    mass_worker::ModelFile f{
        .filename = "model.gguf",
        .sha256 = kHelloSha256,
        .role = 1,
    };
    auto r = fetcher.fetch_all({f}, {});
    ASSERT_TRUE(r);
    EXPECT_EQ((*r)[0], cached);
}

TEST(FetcherTest, MissingFileNoUrlIsError) {
    TempDir td;
    mass_worker::Fetcher fetcher(td.path() / "cache");
    mass_worker::ModelFile f{
        .filename = "model.gguf",
        .role = 1,
    };
    auto r = fetcher.fetch_all({f}, {});
    ASSERT_FALSE(r);
    EXPECT_EQ(r.error().code, mass_worker::FetchErrorCode::NoSourceProvided);
}

TEST(FetcherTest, NoPrimaryRoleIsError) {
    TempDir td;
    auto loopback = td.path() / "f.gguf";
    write_file(loopback, "x");

    mass_worker::Fetcher fetcher(td.path() / "cache");
    mass_worker::ModelFile f{
        .local_path = loopback.string(),
        .role = 2,  // MMPROJ — no PRIMARY in request
    };
    auto r = fetcher.fetch_all({f}, {});
    ASSERT_FALSE(r);
    EXPECT_EQ(r.error().code, mass_worker::FetchErrorCode::NoPrimaryRole);
}

TEST(FetcherTest, CancelledDownloadAbortsWithoutRetries) {
    TempDir td;
    mass_worker::Fetcher fetcher(td.path() / "cache");
    mass_worker::ModelFile f{
        .filename = "model.gguf",
        .url = "http://127.0.0.1:1/model.gguf",  // refused fast; cancel wins
        .role = 1,
    };
    auto r = fetcher.fetch_all({f}, [] { return true; });
    ASSERT_FALSE(r);
    EXPECT_EQ(r.error().code, mass_worker::FetchErrorCode::DownloadFailed);
    // A cancelled fetch reports as cancelled and skips the retry ladder.
    EXPECT_NE(r.error().message.find("cancelled"), std::string::npos);
}

TEST(FetcherTest, CorruptCachedFileIsDeletedForRefetch) {
    TempDir td;
    auto cache = td.path() / "cache";
    auto cached = cache / "model.gguf";
    write_file(cached, "corrupted bytes");

    mass_worker::Fetcher fetcher(cache);
    mass_worker::ModelFile f{
        .filename = "model.gguf",
        .sha256 = kHelloSha256,  // doesn't match "corrupted bytes"
        .role = 1,
    };
    // No URL, so after deleting the corrupt file the refetch has no source —
    // the NoSourceProvided error (not Sha256Mismatch) proves the fall-through.
    auto r = fetcher.fetch_all({f}, {});
    ASSERT_FALSE(r);
    EXPECT_EQ(r.error().code, mass_worker::FetchErrorCode::NoSourceProvided);
    EXPECT_FALSE(fs::exists(cached)) << "corrupt cached file must be deleted";
}

TEST(FetcherTest, SweepPartialsRemovesOnlyDownloadArtifacts) {
    TempDir td;
    write_file(td.path() / "model.gguf", "keep");
    write_file(td.path() / "a.gguf.downloading", "partial");
    write_file(td.path() / "sub" / "b.gguf.downloading-123456", "legacy partial");

    mass_worker::Fetcher fetcher(td.path());
    fetcher.sweep_partials();

    EXPECT_TRUE(fs::exists(td.path() / "model.gguf"));
    EXPECT_FALSE(fs::exists(td.path() / "a.gguf.downloading"));
    EXPECT_FALSE(fs::exists(td.path() / "sub" / "b.gguf.downloading-123456"));
}

// --- Retry policy: which download failures are worth another attempt --------

struct RetryCase {
    long http_status;
    int write_errno;
    bool retryable;
    const char* why;
};

class DownloadRetryPolicy : public ::testing::TestWithParam<RetryCase> {};

TEST_P(DownloadRetryPolicy, ClassifiesTheFailure) {
    const auto& c = GetParam();
    EXPECT_EQ(mass_worker::download_failure_is_retryable(c.http_status, c.write_errno), c.retryable)
        << c.why;
}

INSTANTIATE_TEST_SUITE_P(
    Failures, DownloadRetryPolicy,
    ::testing::Values(
        // No status at all: DNS, connect, reset, or the low-speed stall abort.
        RetryCase{0, 0, true, "a transport failure before any status is transient"},
        RetryCase{200, 0, true,
                  "a 200 that broke mid-body is a broken transfer, not a bad request"},
        RetryCase{206, 0, true, "partial content that broke mid-body: resume and retry"},
        // 4xx is the request being wrong, and it will be wrong again. Retrying
        // only burns three attempts and 3s of sleeps per file.
        RetryCase{400, 0, false, "a malformed request stays malformed"},
        RetryCase{401, 0, false, "missing credentials do not appear on their own"},
        RetryCase{403, 0, false, "a rejected token stays rejected"},
        RetryCase{404, 0, false, "a missing object stays missing"},
        RetryCase{410, 0, false, "gone is gone"},
        RetryCase{451, 0, false, "any other 4xx is the request, not the moment"},
        // The two 4xx codes that explicitly mean "come back later".
        RetryCase{408, 0, true, "request timeout invites another attempt"},
        RetryCase{429, 0, true, "too many requests invites another attempt"},
        RetryCase{500, 0, true, "a server-side fault may be transient"},
        RetryCase{502, 0, true, "bad gateway may be transient"},
        RetryCase{503, 0, true, "service unavailable may be transient"},
        // A full disk is local and terminal whatever the server said.
        RetryCase{0, ENOSPC, false, "a full disk is not a network problem"},
        RetryCase{200, ENOSPC, false, "disk full outranks a healthy status"},
        RetryCase{503, ENOSPC, false, "disk full outranks a retryable status"},
        // ENOSPC is the only write errno singled out, deliberately.
        RetryCase{0, EIO, true, "a transient I/O error gets another go"}));

// --- Download, place, verify ------------------------------------------------

// Names of every "*.downloading*" partial under `dir`.
std::vector<std::string> partial_names(const fs::path& dir) {
    std::vector<std::string> out;
    std::error_code ec;
    if (!fs::is_directory(dir, ec)) return out;
    for (fs::recursive_directory_iterator it(dir, ec), end; it != end; it.increment(ec)) {
        if (ec) break;
        const std::string name = it->path().filename().string();
        if (name.find(".downloading") != std::string::npos) out.push_back(name);
    }
    return out;
}

// A scripted HTTP/1.1 origin, one response per request, served sequentially on a
// background thread. The fetch layer can only be driven over HTTP — this build's
// libcurl compiles in http/https and nothing else, so there is no local URL
// scheme to point a test at — and the retry/resume rules are not observable any
// other way: whether a 404 is retried, whether a mismatch is refetched, and
// above all whether an ignored Range corrupts the file.
class ScriptedOrigin {
public:
    struct Reply {
        int status{200};
        std::string body;
        // Serve fewer bytes than Content-Length announces, then close: the
        // client sees a truncated transfer and keeps the partial bytes.
        bool truncate{false};
        // false → answer a Range request with 200 and the WHOLE body, the way a
        // server without range support does.
        bool honour_range{true};
        // Answer a Range request with the WHOLE body, a 200, AND a Content-Range
        // header. libcurl accepts that combination (its own resume check only
        // fires when Content-Range is absent), so this is the shape that reaches
        // the worker's own 206 check.
        bool fake_content_range{false};
    };

    // EXPECT rather than ASSERT: the fatal form returns, which a constructor
    // cannot do. A failure here leaves port_ at 0 and the test fails on the
    // fetch, with these expectations naming the reason.
    explicit ScriptedOrigin(std::vector<Reply> script)
        : script_(std::move(script)), listen_(::socket(AF_INET, SOCK_STREAM, 0)) {
        EXPECT_NE(listen_, kBadSocket) << "socket() failed";
        int on = 1;
        ::setsockopt(listen_, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<const char*>(&on),
                     sizeof(on));
        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        addr.sin_port = 0;  // any free port
        EXPECT_EQ(::bind(listen_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)), 0);
        EXPECT_EQ(::listen(listen_, 8), 0);
        socklen_t len = sizeof(addr);
        EXPECT_EQ(::getsockname(listen_, reinterpret_cast<sockaddr*>(&addr), &len), 0);
        port_ = ntohs(addr.sin_port);
        thread_ = std::thread([this] { serve(); });
    }

    // Join BEFORE closing: closing a listening socket does not reliably wake a
    // thread already blocked in accept(), so the serve loop polls with select()
    // and exits on its own once stopping_ is set.
    ~ScriptedOrigin() {
        stopping_.store(true);
        if (thread_.joinable()) thread_.join();
        close_socket(listen_);
    }

    ScriptedOrigin(const ScriptedOrigin&) = delete;
    ScriptedOrigin& operator=(const ScriptedOrigin&) = delete;

    [[nodiscard]] std::string url(const std::string& path) const {
        return "http://127.0.0.1:" + std::to_string(port_) + path;
    }
    // Requests served so far, and the Range offset each one asked for (-1 = no
    // Range header) — the retry ladder is counted straight off these.
    [[nodiscard]] std::vector<long long> range_offsets() const {
        std::scoped_lock lk(mu_);
        return offsets_;
    }
    [[nodiscard]] std::size_t requests() const {
        std::scoped_lock lk(mu_);
        return offsets_.size();
    }

private:
    void serve() {
        while (!stopping_.load()) {
            // select() rather than a bare accept() so the loop comes back to
            // check stopping_ — see the destructor.
            fd_set readable;
            FD_ZERO(&readable);
            FD_SET(listen_, &readable);
            timeval tv{.tv_sec = 0, .tv_usec = 50'000};  // 50ms
            if (::select(static_cast<int>(listen_ + 1), &readable, nullptr, nullptr, &tv) <= 0) {
                continue;
            }
            socket_t conn = ::accept(listen_, nullptr, nullptr);
            if (conn == kBadSocket) continue;
            std::string req;
            char buf[2048];
            while (req.find("\r\n\r\n") == std::string::npos) {
                const auto n = ::recv(conn, buf, static_cast<iolen_t>(sizeof(buf)), 0);
                if (n <= 0) break;
                req.append(buf, static_cast<std::size_t>(n));
            }
            const long long offset = parse_range(req);
            Reply reply;
            {
                std::scoped_lock lk(mu_);
                offsets_.push_back(offset);
                // Past the end of the script, the last reply repeats.
                const std::size_t i = std::min(offsets_.size() - 1, script_.size() - 1);
                reply = script_[i];
            }
            const std::string out = render(reply, offset);
            ::send(conn, out.data(), static_cast<iolen_t>(out.size()), 0);
            close_socket(conn);
        }
    }

    static long long parse_range(const std::string& req) {
        const auto at = req.find("Range: bytes=");
        if (at == std::string::npos) return -1;
        return std::strtoll(req.c_str() + at + std::string("Range: bytes=").size(), nullptr, 10);
    }

    static std::string render(const Reply& reply, long long offset) {
        // A resume offset past the end is what a real origin answers 416 to.
        if (offset > 0 && std::cmp_greater_equal(offset, reply.body.size())) {
            return "HTTP/1.1 416 Range Not Satisfiable\r\nContent-Length: 0\r\n"
                   "Connection: close\r\n\r\n";
        }
        const bool partial = offset > 0 && reply.honour_range && reply.status == 200;
        std::string body = reply.body;
        if (partial) body = body.substr(static_cast<std::size_t>(offset));

        std::string head = "HTTP/1.1 ";
        if (reply.status != 200) {
            head += std::to_string(reply.status) + " Error\r\n";
        } else {
            head += partial ? "206 Partial Content\r\n" : "200 OK\r\n";
        }
        head += "Content-Length: " + std::to_string(body.size()) + "\r\n";
        // fake_content_range sends this header WITHOUT the 206 — the combination
        // libcurl accepts and only the worker's own check rejects.
        if (offset > 0 && (partial || reply.fake_content_range)) {
            head += "Content-Range: bytes " + std::to_string(offset) + "-" +
                    std::to_string(reply.body.size() - 1) + "/" +
                    std::to_string(reply.body.size()) + "\r\n";
        }
        head += "Connection: close\r\n\r\n";
        // Truncated: the announced length stands, but half the bytes arrive.
        if (reply.truncate) body.resize(body.size() / 2);
        return head + body;
    }

    mutable std::mutex mu_;
    std::vector<long long> offsets_;
    std::vector<Reply> script_;
    socket_t listen_{kBadSocket};
    int port_{0};
    std::atomic<bool> stopping_{false};
    std::thread thread_;
};

TEST(FetcherTest, DownloadsPlacesAndVerifiesDigest) {
    TempDir td;
    const auto cache = td.path() / "cache";
    ScriptedOrigin origin({{.body = "hello world\n"}});

    mass_worker::Fetcher fetcher(cache);
    mass_worker::ModelFile f{
        .filename = "model.gguf",
        .url = origin.url("/model.gguf"),
        .sha256 = kHelloSha256,
        .role = 1,
    };
    auto r = fetcher.fetch_all({f}, {});
    ASSERT_TRUE(r) << (r ? "" : r.error().message);
    EXPECT_EQ((*r)[0], cache / "model.gguf");
    EXPECT_EQ(origin.requests(), 1u);
    // The rename consumes the partial; nothing is left behind on success.
    EXPECT_TRUE(partial_names(cache).empty());
}

TEST(FetcherTest, NotFoundIsTerminalAndNotRetried) {
    TempDir td;
    ScriptedOrigin origin({{.status = 404}});

    mass_worker::Fetcher fetcher(td.path() / "cache");
    mass_worker::ModelFile f{
        .filename = "model.gguf",
        .url = origin.url("/missing.gguf"),
        .role = 1,
    };
    auto r = fetcher.fetch_all({f}, {});
    ASSERT_FALSE(r);
    EXPECT_EQ(r.error().code, mass_worker::FetchErrorCode::DownloadFailed);
    EXPECT_NE(r.error().message.find("404"), std::string::npos) << r.error().message;
    // The point of the fix: one attempt, not three attempts and 3s of sleeps.
    EXPECT_EQ(origin.requests(), 1u);
}

TEST(FetcherTest, TooManyRequestsIsRetried) {
    TempDir td;
    const auto cache = td.path() / "cache";
    ScriptedOrigin origin({{.status = 429}, {.body = "hello world\n"}});

    mass_worker::Fetcher fetcher(cache);
    mass_worker::ModelFile f{
        .filename = "model.gguf",
        .url = origin.url("/model.gguf"),
        .sha256 = kHelloSha256,
        .role = 1,
    };
    auto r = fetcher.fetch_all({f}, {});
    ASSERT_TRUE(r) << (r ? "" : r.error().message);
    EXPECT_EQ(origin.requests(), 2u) << "429 must be retried, unlike the rest of 4xx";
}

TEST(FetcherTest, DigestMismatchAfterDownloadRefetchesOnceThenFails) {
    TempDir td;
    const auto cache = td.path() / "cache";
    ScriptedOrigin origin({{.body = "not hello world at all"}});  // never matches

    mass_worker::Fetcher fetcher(cache);
    mass_worker::ModelFile f{
        .filename = "model.gguf",
        .url = origin.url("/model.gguf"),
        .sha256 = kHelloSha256,
        .role = 1,
    };
    auto r = fetcher.fetch_all({f}, {});
    ASSERT_FALSE(r);
    EXPECT_EQ(r.error().code, mass_worker::FetchErrorCode::Sha256Mismatch);
    // The download plus exactly one refetch — the generosity a corrupt *cached*
    // file already got. Both from zero: neither request carries a Range.
    EXPECT_EQ(origin.range_offsets(), (std::vector<long long>{-1, -1}));
    // Nothing corrupt is left to be "reused" (and re-rejected) forever.
    EXPECT_FALSE(fs::exists(cache / "model.gguf"));
    EXPECT_TRUE(partial_names(cache).empty());
}

// A truncated transfer leaves a partial, the retry asks to resume from it, and
// the server does not honour the Range. The partial has to be thrown away: kept,
// it makes every remaining attempt ask for the same unsatisfiable resume, so the
// download fails permanently until the next startup sweep — and in the
// Content-Range-but-not-206 variant below, the whole object gets appended to it.
//
// Parameterised over the two shapes a non-honouring server takes, because
// libcurl only catches the first one for us.
class RangeRefused : public ::testing::TestWithParam<bool> {};

TEST_P(RangeRefused, RestartsFromZeroInsteadOfResumingForever) {
    const bool fake_content_range = GetParam();
    TempDir td;
    const auto cache = td.path() / "cache";
    const std::string body = "hello world\n";
    ScriptedOrigin origin({
        {.body = body, .truncate = true},  // partial bytes, then close
        // Refuses the resume: either plainly (libcurl raises CURLE_RANGE_ERROR)
        // or with a Content-Range it accepts (only our own 206 check catches it).
        {.body = body, .honour_range = false, .fake_content_range = fake_content_range},
        {.body = body},  // from zero: clean
    });

    mass_worker::Fetcher fetcher(cache);
    mass_worker::ModelFile f{
        .filename = "model.gguf",
        .url = origin.url("/model.gguf"),
        .role = 1,  // deliberately NO sha256 — the corruption would be invisible
    };
    auto r = fetcher.fetch_all({f}, {});
    ASSERT_TRUE(r) << (r ? "" : r.error().message);
    // Attempt 2 resumed, attempt 3 started over because the resume wasn't honoured.
    const auto offsets = origin.range_offsets();
    ASSERT_EQ(offsets.size(), 3u) << "expected truncate, resume, restart";
    EXPECT_EQ(offsets[0], -1);
    EXPECT_GT(offsets[1], 0) << "the retry must resume from the partial bytes";
    EXPECT_EQ(offsets[2], -1) << "after an ignored Range the next attempt starts at zero";
    // The delivered file is exactly the object, not the partial plus the object.
    const auto placed = (*r)[0];
    auto digest = mass_worker::Fetcher::verify_sha256(placed, kHelloSha256);
    EXPECT_TRUE(digest) << "a resumed download must not concatenate: "
                        << (digest ? "" : digest.error().message);
    EXPECT_EQ(fs::file_size(placed), body.size());
    EXPECT_TRUE(partial_names(cache).empty());
}

INSTANTIATE_TEST_SUITE_P(WithoutAndWithContentRange, RangeRefused, ::testing::Bool());

// The nastiest shape of an unhonoured resume. libcurl answers a 416 to a Range
// request by declaring the file ALREADY DOWNLOADED: CURLE_OK, nothing
// transferred. So a partial left over from a larger object — here 20 bytes of a
// 40-byte body, against an origin now serving 12 — gets renamed into place as
// the finished model. With no sha256 to catch it (deliberately none here, and
// often none in practice) the worker would load a 20-byte "model".
TEST(FetcherTest, UnconfirmedResumeIsDiscardedInsteadOfPassingAsComplete) {
    TempDir td;
    const auto cache = td.path() / "cache";
    const std::string served = "hello world\n";
    ScriptedOrigin origin({
        // A big object cut off mid-transfer: leaves a partial LONGER than the
        // object the next attempts are offered, so the resume is unsatisfiable.
        {.body = std::string(40, 'x'), .truncate = true},
        {.body = served},
    });

    mass_worker::Fetcher fetcher(cache);
    mass_worker::ModelFile f{
        .filename = "model.gguf",
        .url = origin.url("/model.gguf"),
        .role = 1,  // no sha256: nothing else can catch a bad file
    };
    auto r = fetcher.fetch_all({f}, {});
    ASSERT_TRUE(r) << (r ? "" : r.error().message);

    const auto offsets = origin.range_offsets();
    ASSERT_EQ(offsets.size(), 3u);
    EXPECT_EQ(offsets[0], -1);
    EXPECT_EQ(offsets[1], 20) << "the retry resumes from the 20 partial bytes";
    EXPECT_EQ(offsets[2], -1) << "an unconfirmed resume drops the partial and starts over";
    // What landed is the object, not the 20-byte carcass of the previous one.
    const auto placed = (*r)[0];
    EXPECT_EQ(fs::file_size(placed), served.size());
    EXPECT_TRUE(mass_worker::Fetcher::verify_sha256(placed, kHelloSha256));
    EXPECT_TRUE(partial_names(cache).empty());
}

// Two loads can map DIFFERENT urls onto the same cache-relative filename. Keyed
// on the destination alone, the second would resume from the first's partial
// bytes and produce a file that is neither.
TEST(FetcherTest, PartialFileNameIsKeyedOnTheUrl) {
    auto name_for_url = [](const std::string& url) {
        TempDir td;
        const auto cache = td.path() / "cache";
        mass_worker::Fetcher fetcher(cache);
        mass_worker::ModelFile f{.filename = "model.gguf", .url = url, .role = 1};
        // Refused instantly, and cancelling makes it terminal — one attempt, no
        // retry sleeps, and the temp file it opened stays behind to be read.
        auto r = fetcher.fetch_all({f}, [] { return true; });
        EXPECT_FALSE(r);
        auto names = partial_names(cache);
        return names.empty() ? std::string{} : names.front();
    };

    const std::string a = name_for_url("http://127.0.0.1:1/repo-a/model.gguf");
    const std::string b = name_for_url("http://127.0.0.1:1/repo-b/model.gguf");
    ASSERT_FALSE(a.empty());
    ASSERT_FALSE(b.empty());
    EXPECT_NE(a, b) << "same partial name for two different urls: " << a;
    // Still swept at startup: sweep_partials matches ".downloading" anywhere.
    EXPECT_NE(a.find(".downloading"), std::string::npos) << a;
}

TEST(FetcherTest, DuplicateRoleIsError) {
    TempDir td;
    auto a = td.path() / "a.gguf";
    auto b = td.path() / "b.gguf";
    write_file(a, "1");
    write_file(b, "2");

    mass_worker::Fetcher fetcher(td.path() / "cache");
    auto r = fetcher.fetch_all(
        {
            {.local_path = a.string(), .role = 1},
            {.local_path = b.string(), .role = 1},  // duplicate PRIMARY
        },
        {});
    ASSERT_FALSE(r);
    EXPECT_EQ(r.error().code, mass_worker::FetchErrorCode::DuplicateRole);
}

// --- Cache ---

TEST(CacheTest, ListGgufSkipsTempfilesAndNonGguf) {
    TempDir td;
    write_file(td.path() / "model-a.gguf", "x");
    write_file(td.path() / "sub" / "model-b.GGUF", "x");  // case-insens
    write_file(td.path() / "model-c.txt", "x");           // wrong ext
    write_file(td.path() / ".downloading-x.gguf", "x");   // tempfile

    mass_worker::Cache cache(td.path());
    auto files = cache.list_gguf();
    std::ranges::sort(files);
    ASSERT_EQ(files.size(), 2u);
    EXPECT_EQ(files[0], "model-a.gguf");
    EXPECT_EQ(files[1], "sub/model-b.GGUF");
}

TEST(CacheTest, SafePathRejectsTraversal) {
    mass_worker::Cache cache(fs::path("/some/cache"));
    EXPECT_FALSE(cache.safe_cache_path("../etc/passwd"));
    EXPECT_FALSE(cache.safe_cache_path("a/../../b"));
    EXPECT_FALSE(cache.safe_cache_path(""));
    // ':' — drive-relative / NTFS stream hazard, same as the fetch path.
    EXPECT_FALSE(cache.safe_cache_path("C:evil"));
    EXPECT_FALSE(cache.safe_cache_path("gguf/name:stream.gguf"));
}

TEST(CacheTest, SafePathAcceptsCleanRelative) {
    mass_worker::Cache cache(fs::path("/some/cache"));
    auto r = cache.safe_cache_path("models/Qwen.gguf");
    ASSERT_TRUE(r);
    EXPECT_EQ(r->generic_string(), "/some/cache/models/Qwen.gguf");
}

TEST(CacheTest, DeleteRemovesUnloadedAndSkipsLoaded) {
    TempDir td;
    write_file(td.path() / "a.gguf", "x");
    write_file(td.path() / "b.gguf", "y");

    mass_worker::Cache cache(td.path());
    std::unordered_set<fs::path> loaded;
    loaded.insert(td.path() / "b.gguf");

    cache.delete_files({"a.gguf", "b.gguf", "../escape.gguf"}, loaded);

    EXPECT_FALSE(fs::exists(td.path() / "a.gguf"));
    EXPECT_TRUE(fs::exists(td.path() / "b.gguf"));  // skipped because loaded
}

TEST(CacheTest, DeletePrunesEmptyParents) {
    TempDir td;
    write_file(td.path() / "deep" / "deeper" / "x.gguf", "x");

    mass_worker::Cache cache(td.path());
    cache.delete_files({"deep/deeper/x.gguf"}, {});

    EXPECT_FALSE(fs::exists(td.path() / "deep" / "deeper"));
    EXPECT_FALSE(fs::exists(td.path() / "deep"));
    EXPECT_TRUE(fs::exists(td.path()));  // root not pruned
}

}  // namespace
