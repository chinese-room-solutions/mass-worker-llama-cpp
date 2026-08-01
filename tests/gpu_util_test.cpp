#include "mass_worker/gpu_util.hpp"

#include <gtest/gtest.h>

#include <string>
#include <string_view>

namespace {

using mass_worker::busy_ns_to_util_pct;
using mass_worker::classify_gpu_vendor;
using mass_worker::DrmClientBusy;
using mass_worker::GpuVendor;
using mass_worker::parse_drm_fdinfo;

// --- Vendor classification --------------------------------------------------

struct VendorCase {
    std::string_view name;
    GpuVendor want;
};

class VendorTest : public testing::TestWithParam<VendorCase> {};

TEST_P(VendorTest, ClassifiesByBrandToken) {
    EXPECT_EQ(classify_gpu_vendor(GetParam().name), GetParam().want);
}

INSTANTIATE_TEST_SUITE_P(
    Vendors, VendorTest,
    testing::Values(VendorCase{"NVIDIA GeForce RTX 4090", GpuVendor::Nvidia},
                    VendorCase{"Tesla T4", GpuVendor::Nvidia},
                    VendorCase{"Intel(R) Iris(R) Xe Graphics (ADL GT2)", GpuVendor::Intel},
                    VendorCase{"AMD Radeon RX 7900 XTX", GpuVendor::Amd},
                    VendorCase{"Radeon RX 580", GpuVendor::Amd},
                    VendorCase{"Apple M3 Max", GpuVendor::Apple},
                    VendorCase{"Some Unknown Accelerator", GpuVendor::Unknown}));

// --- busy-ns -> util% -------------------------------------------------------

struct UtilCase {
    std::int64_t busy_delta;
    std::int64_t elapsed;
    double want;
};

class UtilPctTest : public testing::TestWithParam<UtilCase> {};

TEST_P(UtilPctTest, ConvertsAndClamps) {
    EXPECT_DOUBLE_EQ(busy_ns_to_util_pct(GetParam().busy_delta, GetParam().elapsed),
                     GetParam().want);
}

INSTANTIATE_TEST_SUITE_P(
    Conversions, UtilPctTest,
    testing::Values(UtilCase{0, 1'000'000, 0.0}, UtilCase{500'000, 1'000'000, 50.0},
                    UtilCase{1'000'000, 1'000'000, 100.0},
                    // Cross-engine busy can exceed wall-clock — clamp to 100, not 150.
                    UtilCase{1'500'000, 1'000'000, 100.0},
                    // Clock didn't advance: avoid divide-by-zero, report 0.
                    UtilCase{500'000, 0, 0.0}));

// --- DRM fdinfo parsing -----------------------------------------------------

TEST(DrmFdinfoTest, SumsEngineBusyAndReadsClientId) {
    const std::string text =
        "drm-driver:\ti915\n"
        "drm-client-id:\t17\n"
        "drm-pdev:\t0000:00:02.0\n"
        "drm-total-system0:\t3544 KiB\n"
        "drm-engine-render:\t222456 ns\n"
        "drm-engine-copy:\t1000 ns\n"
        "drm-engine-video:\t0 ns\n"
        "drm-engine-capacity-video:\t2\n"  // capacity, NOT busy — must skip
        "drm-engine-video-enhance:\t44 ns\n";

    const DrmClientBusy b = parse_drm_fdinfo(text);
    EXPECT_EQ(b.client_id, 17);
    EXPECT_TRUE(b.has_engine);
    // 222456 + 1000 + 0 + 44; the capacity line (2) must not be summed in.
    EXPECT_EQ(b.busy_ns, 223500);
}

TEST(DrmFdinfoTest, NoEngineLinesReportsHasEngineFalse) {
    const std::string text =
        "drm-driver:\tnvidia\n"
        "drm-client-id:\t5\n";  // proprietary NVIDIA: no drm-engine-* lines
    const DrmClientBusy b = parse_drm_fdinfo(text);
    EXPECT_EQ(b.client_id, 5);
    EXPECT_FALSE(b.has_engine);
    EXPECT_EQ(b.busy_ns, 0);
}

TEST(DrmFdinfoTest, MissingClientIdYieldsNegativeOne) {
    const std::string text = "drm-engine-render:\t10 ns\n";
    const DrmClientBusy b = parse_drm_fdinfo(text);
    EXPECT_EQ(b.client_id, -1);
    EXPECT_TRUE(b.has_engine);
    EXPECT_EQ(b.busy_ns, 10);
}

TEST(DrmFdinfoTest, EmptyInputIsInert) {
    const DrmClientBusy b = parse_drm_fdinfo("");
    EXPECT_EQ(b.client_id, -1);
    EXPECT_FALSE(b.has_engine);
    EXPECT_EQ(b.busy_ns, 0);
}

}  // namespace
