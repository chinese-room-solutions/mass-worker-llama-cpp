#include "mass_worker/config.hpp"

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>

namespace {

namespace fs = std::filesystem;
using mass_worker::WorkerConfig;

fs::path temp_dir(const std::string& tag) {
    const fs::path d = fs::temp_directory_path() /
                       ("mass-config-test-" + tag + "-" +
                        std::to_string(::testing::UnitTest::GetInstance()->random_seed()));
    fs::remove_all(d);
    fs::create_directories(d);
    return d;
}

TEST(WorkerConfig, SaveThenLoadRoundTrips) {
    const fs::path dir = temp_dir("rt");

    WorkerConfig cfg;
    cfg.name = "llama-rig-1";
    cfg.models_dir = "/var/lib/mass-worker-llama-cpp/models";
    cfg.gpu_backend = "vulkan";
    cfg.log_level = "info";
    cfg.log_file = "/var/log/mass-worker-llama-cpp.log";
    cfg.vram_headroom_pct = 80;
    ASSERT_TRUE(mass_worker::save_config(dir.string(), cfg));

    auto loaded = mass_worker::load_config(dir.string());
    ASSERT_TRUE(loaded.has_value());
    EXPECT_EQ(loaded->name, "llama-rig-1");
    EXPECT_EQ(loaded->models_dir, "/var/lib/mass-worker-llama-cpp/models");
    EXPECT_EQ(loaded->gpu_backend, "vulkan");
    EXPECT_EQ(loaded->log_level, "info");
    EXPECT_EQ(loaded->log_file, "/var/log/mass-worker-llama-cpp.log");
    EXPECT_EQ(loaded->vram_headroom_pct, 80);

    fs::remove_all(dir);
}

TEST(WorkerConfig, UnsetFieldsStayUnsetAcrossRoundTrip) {
    const fs::path dir = temp_dir("partial");

    // Only two fields set; the rest must remain nullopt after a round-trip so
    // the launch-time merge falls through to flags/defaults for them.
    WorkerConfig cfg;
    cfg.name = "partial";
    cfg.models_dir = "models";
    ASSERT_TRUE(mass_worker::save_config(dir.string(), cfg));

    auto loaded = mass_worker::load_config(dir.string());
    ASSERT_TRUE(loaded.has_value());
    EXPECT_EQ(loaded->name, "partial");
    EXPECT_EQ(loaded->models_dir, "models");
    EXPECT_FALSE(loaded->gpu_backend.has_value());
    EXPECT_FALSE(loaded->log_level.has_value());
    EXPECT_FALSE(loaded->log_file.has_value());
    EXPECT_FALSE(loaded->vram_headroom_pct.has_value());

    fs::remove_all(dir);
}

TEST(WorkerConfig, MissingFileLoadsAllUnset) {
    const fs::path dir = temp_dir("missing");
    auto loaded = mass_worker::load_config(dir.string());
    ASSERT_TRUE(loaded.has_value());  // absent is not an error
    EXPECT_FALSE(loaded->name.has_value());
    EXPECT_FALSE(loaded->models_dir.has_value());
    EXPECT_FALSE(loaded->vram_headroom_pct.has_value());
    fs::remove_all(dir);
}

TEST(WorkerConfig, MalformedLineReturnsNullopt) {
    const fs::path dir = temp_dir("corrupt");
    // A non-comment line with no '=' is malformed — load must refuse rather
    // than silently ignore the file, so a launch fails loudly instead of
    // dropping operator intent.
    std::ofstream(mass_worker::config_path(dir.string()))
        << "name=ok\nthis line has no equals sign\n";
    EXPECT_FALSE(mass_worker::load_config(dir.string()).has_value());
    fs::remove_all(dir);
}

TEST(WorkerConfig, NonIntegerVramReturnsNullopt) {
    const fs::path dir = temp_dir("badvram");
    std::ofstream(mass_worker::config_path(dir.string())) << "vram_headroom_pct=lots\n";
    EXPECT_FALSE(mass_worker::load_config(dir.string()).has_value());
    fs::remove_all(dir);
}

TEST(WorkerConfig, CommentsAndBlankLinesIgnored) {
    const fs::path dir = temp_dir("comments");
    std::ofstream(mass_worker::config_path(dir.string()))
        << "# a comment\n\n  # indented comment\nname=rig\n";
    auto loaded = mass_worker::load_config(dir.string());
    ASSERT_TRUE(loaded.has_value());
    EXPECT_EQ(loaded->name, "rig");
    EXPECT_FALSE(loaded->models_dir.has_value());
    fs::remove_all(dir);
}

TEST(WorkerConfig, ConfigPathIsUnderDataDir) {
    const std::string p = mass_worker::config_path("/data/mass");
    EXPECT_NE(p.find("config.conf"), std::string::npos);
    EXPECT_NE(p.find("mass"), std::string::npos);
}

}  // namespace
