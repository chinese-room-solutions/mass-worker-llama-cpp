#include "mass_worker/wizard.hpp"

#include <gtest/gtest.h>

#include <filesystem>
#include <string>

#include "mass_worker/service.hpp"  // path_is_writable (the probe the reason uses)

#ifndef _WIN32
#include <unistd.h>  // geteuid (root-skip guard)
#endif

namespace {

namespace fs = std::filesystem;

using mass_worker::elevation_reason;

// A directory we can definitely write (so the reason must NOT mention it).
fs::path writable_dir() {
    const fs::path d = fs::temp_directory_path() / "mass-elev-test" /
                       std::to_string(::testing::UnitTest::GetInstance()->random_seed());
    fs::create_directories(d);
    return d;
}

// --- PhaseCentring: the phase's row centring belongs to its page --------------

// Nothing is centred until a phase page opens. The linear (dumb-terminal)
// fallback never opens a page, so its step list stays flush with the column-0
// transcript above it; and the centring is gone again once the page closes.
TEST(PhaseCentring, OffWithoutAPageAndOffAgainAfterOne) {
    using mass_worker::phase_centred;
    using mass_worker::PhaseCentring;

    EXPECT_FALSE(phase_centred());
    {
        const PhaseCentring page(true);
        EXPECT_TRUE(phase_centred());
    }
    EXPECT_FALSE(phase_centred());
}

// --- elevation_reason: the "why root is needed" explanation ------------------

TEST(ElevationReason, AlwaysNamesTheServiceEvenWithNoDirs) {
    // Registering the machine-wide service is always privileged, so the reason is
    // never empty — even when no directories are passed.
    const std::string why = elevation_reason({});
    EXPECT_NE(why.find("machine-wide service"), std::string::npos);
    EXPECT_EQ(why.find("writing the"), std::string::npos);  // no dir clause
}

TEST(ElevationReason, OmitsWritableDirs) {
    const fs::path d = writable_dir();
    const std::string why = elevation_reason({{"install directory", d.string()}});
    EXPECT_NE(why.find("machine-wide service"), std::string::npos);
    // A dir the user can already write isn't a reason for elevation.
    EXPECT_EQ(why.find(d.string()), std::string::npos);
    EXPECT_EQ(why.find("writing the"), std::string::npos);
    fs::remove_all(d);
}

TEST(ElevationReason, SkipsEmptyPaths) {
    // An empty path (e.g. an unset data dir) must not produce a phantom clause.
    const std::string why = elevation_reason({{"data directory", ""}});
    EXPECT_EQ(why.find("writing the"), std::string::npos);
}

#ifndef _WIN32
TEST(ElevationReason, NamesAnUnwritableDir) {
    if (::geteuid() == 0) GTEST_SKIP() << "running as root; nothing is unwritable";
    // /proc is a real, never-writable path on Linux; on macOS /var/lib doesn't
    // exist but its nearest ancestor (/var) isn't user-writable either. Use the
    // system data dir, which the unprivileged user can't write.
    const std::string sysdir = mass_worker::service_data_dir();
    if (mass_worker::path_is_writable(sysdir)) {
        GTEST_SKIP() << "system data dir unexpectedly writable here";
    }
    const std::string why = elevation_reason({{"data directory", sysdir}});
    EXPECT_NE(why.find("machine-wide service"), std::string::npos);
    EXPECT_NE(why.find("writing the"), std::string::npos);
    EXPECT_NE(why.find("data directory"), std::string::npos);
    EXPECT_NE(why.find(sysdir), std::string::npos);
}

TEST(ElevationReason, JoinsMultipleUnwritableDirsWithAnd) {
    if (::geteuid() == 0) GTEST_SKIP() << "running as root; nothing is unwritable";
    const std::string a = mass_worker::service_data_dir();
    const std::string b = mass_worker::default_install_dir();
    if (mass_worker::path_is_writable(a) || mass_worker::path_is_writable(b)) {
        GTEST_SKIP() << "a system dir unexpectedly writable here";
    }
    const std::string why = elevation_reason({{"install directory", b}, {"data directory", a}});
    // Both named, joined with " and " (the two-item conjunction).
    EXPECT_NE(why.find("install directory"), std::string::npos);
    EXPECT_NE(why.find("data directory"), std::string::npos);
    EXPECT_NE(why.find(" and "), std::string::npos);
}

TEST(ElevationReason, MixedWritableAndNot_OnlyNamesTheUnwritable) {
    if (::geteuid() == 0) GTEST_SKIP() << "running as root; nothing is unwritable";
    const fs::path ok = writable_dir();
    const std::string locked = mass_worker::service_data_dir();
    if (mass_worker::path_is_writable(locked)) {
        GTEST_SKIP() << "system data dir unexpectedly writable here";
    }
    const std::string why =
        elevation_reason({{"install directory", ok.string()}, {"data directory", locked}});
    EXPECT_EQ(why.find(ok.string()), std::string::npos);  // writable omitted
    EXPECT_NE(why.find(locked), std::string::npos);       // unwritable named
    // Single unwritable item → no " and " between dir entries.
    EXPECT_EQ(why.find(") and "), std::string::npos);
    fs::remove_all(ok);
}
#endif  // !_WIN32

}  // namespace
