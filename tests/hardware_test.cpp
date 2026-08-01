#include "mass_worker/hardware.hpp"

#include <gtest/gtest.h>

#include <algorithm>

namespace {

TEST(HardwareTest, AlwaysReportsCpuFirst) {
    mass_worker::Hardware hw;
    const auto& devs = hw.devices();
    ASSERT_FALSE(devs.empty());
    EXPECT_EQ(devs[0].id, "cpu:0");
    EXPECT_EQ(devs[0].type, mass_worker::DeviceType::Cpu);
    EXPECT_GT(devs[0].total_memory_mb, 0);
}

TEST(HardwareTest, GpuDeviceIdsAreSequential) {
    mass_worker::Hardware hw;
    int seen_gpus = 0;
    for (const auto& d : hw.devices()) {
        if (d.type != mass_worker::DeviceType::Gpu) continue;
        EXPECT_EQ(d.id, "gpu:" + std::to_string(seen_gpus));
        ++seen_gpus;
    }
    // No assert on count — host-dependent. Just verify the IDs are dense
    // and start at 0 if any GPUs are present.
    SUCCEED();
}

TEST(HardwareTest, StatsMatchDeviceList) {
    mass_worker::Hardware hw;
    const auto stats = hw.stats();
    ASSERT_EQ(stats.size(), hw.devices().size());

    // Stat IDs should match the device list 1:1, in order.
    for (std::size_t i = 0; i < stats.size(); ++i) {
        EXPECT_EQ(stats[i].id, hw.devices()[i].id);
        EXPECT_GE(stats[i].used_memory_mb, 0);
        EXPECT_GE(stats[i].total_memory_mb, 0);
        EXPECT_LE(stats[i].used_memory_mb, stats[i].total_memory_mb);
    }
}

}  // namespace
