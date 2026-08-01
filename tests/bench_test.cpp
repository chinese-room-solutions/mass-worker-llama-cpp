#include "mass_worker/bench.hpp"

#include <gtest/gtest.h>

#include "mass_worker/hardware.hpp"

namespace {

// A device MASS never advertised must come back as a typed error — not a
// BenchResult full of 0.0s that MASS would read as a measurement. The real
// micro-benches are too heavy for CI; the error seam is the contract here.
TEST(BenchTest, UnknownDeviceIsErrorNotZeroResult) {
    mass_worker::Hardware hw;
    auto r = mass_worker::bench_one(hw, "gpu:999");
    ASSERT_FALSE(r);
    EXPECT_EQ(r.error().code, mass_worker::BenchErrorCode::UnknownDevice);
    EXPECT_NE(r.error().message.find("gpu:999"), std::string::npos);
}

TEST(BenchTest, EmptyDeviceIdIsError) {
    mass_worker::Hardware hw;
    auto r = mass_worker::bench_one(hw, "");
    ASSERT_FALSE(r);
    EXPECT_EQ(r.error().code, mass_worker::BenchErrorCode::UnknownDevice);
}

}  // namespace
