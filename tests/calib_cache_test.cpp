#include "mass_worker/calib_cache.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "llama.h"

namespace {

namespace fs = std::filesystem;

using mass_worker::calib_cache_key;
using mass_worker::calib_cache_lookup;
using mass_worker::calib_cache_store;
using mass_worker::CalibEntry;
using mass_worker::kCalibCacheMaxEntries;

class CalibCacheTest : public ::testing::Test {
protected:
    void SetUp() override {
        const auto* info = ::testing::UnitTest::GetInstance()->current_test_info();
        dir_ = fs::temp_directory_path() / (std::string("calib_cache_test_") + info->name());
        fs::remove_all(dir_);
        fs::create_directories(dir_);
        file_ = dir_ / ".calibration-cache";
    }
    void TearDown() override {
        std::error_code ec;
        fs::remove_all(dir_, ec);
    }

    fs::path dir_;
    fs::path file_;
};

TEST_F(CalibCacheTest, LookupOnMissingFileIsMiss) {
    EXPECT_FALSE(calib_cache_lookup(file_, "some-key").has_value());
}

TEST_F(CalibCacheTest, StoreThenLookupRoundTrips) {
    const CalibEntry entry{.graph_secs = 1.75, .slot_deltas = {123456789, 42}};
    calib_cache_store(file_, "key-a", entry);

    const auto got = calib_cache_lookup(file_, "key-a");
    ASSERT_TRUE(got.has_value());
    EXPECT_DOUBLE_EQ(got->graph_secs, 1.75);
    EXPECT_EQ(got->slot_deltas, (std::vector<std::int64_t>{123456789, 42}));

    EXPECT_FALSE(calib_cache_lookup(file_, "key-b").has_value());
}

TEST_F(CalibCacheTest, KeysWithSpacesRoundTrip) {
    // Model paths contain spaces; the key is the line's tail so they must
    // survive the round trip verbatim.
    const std::string key = "/models/My Model v2.gguf|123|456|ctx=4096|devs=gpu:0,cpu:0";
    calib_cache_store(file_, key, {.graph_secs = 0.5, .slot_deltas = {}});
    const auto got = calib_cache_lookup(file_, key);
    ASSERT_TRUE(got.has_value());
    EXPECT_DOUBLE_EQ(got->graph_secs, 0.5);
    EXPECT_TRUE(got->slot_deltas.empty());
}

TEST_F(CalibCacheTest, StoreReplacesExistingKey) {
    calib_cache_store(file_, "key-a", {.graph_secs = 1.0, .slot_deltas = {1}});
    calib_cache_store(file_, "key-a", {.graph_secs = 2.0, .slot_deltas = {2}});

    const auto got = calib_cache_lookup(file_, "key-a");
    ASSERT_TRUE(got.has_value());
    EXPECT_DOUBLE_EQ(got->graph_secs, 2.0);
    EXPECT_EQ(got->slot_deltas, (std::vector<std::int64_t>{2}));

    // Exactly one line remains.
    std::ifstream in(file_);
    std::string line;
    int lines = 0;
    while (std::getline(in, line)) ++lines;
    EXPECT_EQ(lines, 1);
}

TEST_F(CalibCacheTest, OldestEntriesArePrunedAtCap) {
    for (std::size_t i = 0; i < kCalibCacheMaxEntries + 5; ++i) {
        calib_cache_store(file_, "key-" + std::to_string(i),
                          {.graph_secs = 1.0, .slot_deltas = {}});
    }
    EXPECT_FALSE(calib_cache_lookup(file_, "key-0").has_value());
    EXPECT_FALSE(calib_cache_lookup(file_, "key-4").has_value());
    EXPECT_TRUE(calib_cache_lookup(file_, "key-5").has_value());
    EXPECT_TRUE(
        calib_cache_lookup(file_, "key-" + std::to_string(kCalibCacheMaxEntries + 4)).has_value());
}

TEST_F(CalibCacheTest, CorruptLinesAreSkippedAndDroppedOnStore) {
    {
        std::ofstream out(file_);
        out << "not a number at all\n";
        out << "1.5\n";  // truncated: no delta count
        out << "2.5 1 999 valid-key\n";
    }
    const auto got = calib_cache_lookup(file_, "valid-key");
    ASSERT_TRUE(got.has_value());
    EXPECT_DOUBLE_EQ(got->graph_secs, 2.5);
    EXPECT_EQ(got->slot_deltas, (std::vector<std::int64_t>{999}));

    // A store rewrites the file with only parseable entries.
    calib_cache_store(file_, "another-key", {.graph_secs = 3.0, .slot_deltas = {}});
    std::ifstream in(file_);
    std::string line;
    int lines = 0;
    while (std::getline(in, line)) ++lines;
    EXPECT_EQ(lines, 2);
}

TEST_F(CalibCacheTest, StoreCreatesMissingParentDirectories) {
    // A worker co-hosted with the gateway loads absolute paths and never
    // fetches, so models_dir may not exist when the first store runs.
    const fs::path nested = dir_ / "models" / "sub" / ".calibration-cache";
    calib_cache_store(nested, "key-a", {.graph_secs = 1.0, .slot_deltas = {7}});
    const auto got = calib_cache_lookup(nested, "key-a");
    ASSERT_TRUE(got.has_value());
    EXPECT_EQ(got->slot_deltas, (std::vector<std::int64_t>{7}));
}

TEST_F(CalibCacheTest, NonPositiveSecondsAreNeverStored) {
    calib_cache_store(file_, "key-a", {.graph_secs = 0.0, .slot_deltas = {}});
    calib_cache_store(file_, "key-b", {.graph_secs = -1.0, .slot_deltas = {}});
    EXPECT_FALSE(fs::exists(file_));
}

TEST(CalibCacheKey, EveryInputParticipates) {
    const llama_context_params base_params = llama_context_default_params();
    const std::vector<std::string> base_devs = {"gpu:0"};
    const fs::path base_path = "/models/m.gguf";
    const std::string base = calib_cache_key(base_path, base_params, base_devs);

    // Same inputs → same key (the cache would never hit otherwise).
    EXPECT_EQ(base, calib_cache_key(base_path, base_params, base_devs));

    EXPECT_NE(base, calib_cache_key("/models/other.gguf", base_params, base_devs));
    EXPECT_NE(base, calib_cache_key(base_path, base_params, {"gpu:0", "gpu:1"}));
    EXPECT_NE(base, calib_cache_key(base_path, base_params, {}));

    auto p = base_params;
    p.n_ctx = base_params.n_ctx + 1;
    EXPECT_NE(base, calib_cache_key(base_path, p, base_devs));

    p = base_params;
    p.n_batch = base_params.n_batch + 1;
    EXPECT_NE(base, calib_cache_key(base_path, p, base_devs));

    p = base_params;
    p.n_ubatch = base_params.n_ubatch + 1;
    EXPECT_NE(base, calib_cache_key(base_path, p, base_devs));

    p = base_params;
    p.n_threads_batch = base_params.n_threads_batch + 1;
    EXPECT_NE(base, calib_cache_key(base_path, p, base_devs));

    p = base_params;
    p.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    EXPECT_NE(base, calib_cache_key(base_path, p, base_devs));

    p = base_params;
    p.type_k = GGML_TYPE_Q8_0;
    EXPECT_NE(base, calib_cache_key(base_path, p, base_devs));
}

}  // namespace
