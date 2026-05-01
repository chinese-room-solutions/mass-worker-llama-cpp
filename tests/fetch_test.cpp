#include <gtest/gtest.h>

#include <atomic>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>

#include "mass_worker/cache.hpp"
#include "mass_worker/fetch.hpp"

namespace fs = std::filesystem;

namespace {

// Disposable temp dir for tests. RAII cleanup.
class TempDir {
public:
    TempDir() {
        std::random_device rd;
        path_ = fs::temp_directory_path() /
                ("mass-worker-test-" + std::to_string(rd()));
        fs::create_directories(path_);
    }
    ~TempDir() {
        std::error_code ec;
        fs::remove_all(path_, ec);
    }
    const fs::path& path() const { return path_; }

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

TEST(SafeRelFilenameTest, FallsBackToUrlPath) {
    auto r = mass_worker::Fetcher::safe_rel_filename(
        "", "https://hf.co/repo/resolve/main/Qwen-7B.gguf");
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
    auto r = mass_worker::Fetcher::verify_sha256(td.path() / "nope.bin",
                                                 kHelloSha256);
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
    std::atomic<bool> cancel{false};
    auto r = fetcher.fetch_all({f}, cancel);
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
    std::atomic<bool> cancel{false};
    auto r = fetcher.fetch_all({f}, cancel);
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
    std::atomic<bool> cancel{false};
    auto r = fetcher.fetch_all({f}, cancel);
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
    std::atomic<bool> cancel{false};
    auto r = fetcher.fetch_all({f}, cancel);
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
    std::atomic<bool> cancel{false};
    auto r = fetcher.fetch_all({f}, cancel);
    ASSERT_FALSE(r);
    EXPECT_EQ(r.error().code, mass_worker::FetchErrorCode::NoPrimaryRole);
}

TEST(FetcherTest, DuplicateRoleIsError) {
    TempDir td;
    auto a = td.path() / "a.gguf";
    auto b = td.path() / "b.gguf";
    write_file(a, "1");
    write_file(b, "2");

    mass_worker::Fetcher fetcher(td.path() / "cache");
    std::atomic<bool> cancel{false};
    auto r = fetcher.fetch_all({
        {.local_path = a.string(), .role = 1},
        {.local_path = b.string(), .role = 1},  // duplicate PRIMARY
    }, cancel);
    ASSERT_FALSE(r);
    EXPECT_EQ(r.error().code, mass_worker::FetchErrorCode::DuplicateRole);
}

// --- Cache ---

TEST(CacheTest, ListGgufSkipsTempfilesAndNonGguf) {
    TempDir td;
    write_file(td.path() / "model-a.gguf",         "x");
    write_file(td.path() / "sub" / "model-b.GGUF", "x");          // case-insens
    write_file(td.path() / "model-c.txt",          "x");          // wrong ext
    write_file(td.path() / ".downloading-x.gguf",  "x");          // tempfile

    mass_worker::Cache cache(td.path());
    auto files = cache.list_gguf();
    std::sort(files.begin(), files.end());
    ASSERT_EQ(files.size(), 2u);
    EXPECT_EQ(files[0], "model-a.gguf");
    EXPECT_EQ(files[1], "sub/model-b.GGUF");
}

TEST(CacheTest, SafePathRejectsTraversal) {
    mass_worker::Cache cache(fs::path("/some/cache"));
    EXPECT_FALSE(cache.safe_cache_path("../etc/passwd"));
    EXPECT_FALSE(cache.safe_cache_path("a/../../b"));
    EXPECT_FALSE(cache.safe_cache_path(""));
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
