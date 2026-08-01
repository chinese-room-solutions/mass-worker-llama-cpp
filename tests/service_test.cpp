#include <gtest/gtest.h>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#ifndef _WIN32
#include <unistd.h>  // geteuid (root-skip guards in the writability tests)
#endif

#include "mass_worker/exit_codes.hpp"
#include "mass_worker/payload.hpp"
#include "mass_worker/service.hpp"

namespace {

namespace fs = std::filesystem;

using mass_worker::service_args;
using mass_worker::ServiceConfig;

// A fully-populated config the render functions can be exercised against.
ServiceConfig sample_config() {
    ServiceConfig cfg;
    cfg.exe_path = "/opt/mass-worker/bin/mass-worker-llama-cpp";
    cfg.models_dir = "/var/lib/mass/models";
    cfg.name = "rig-1";
    cfg.log_level = "debug";
    cfg.log_file = "/var/log/mass-worker.log";
    cfg.vram_headroom_pct = 80;
    return cfg;
}

bool contains(const std::vector<std::string>& v, const std::string& s) {
    return std::ranges::find(v, s) != v.end();
}

// The flag immediately following `flag` in the args vector, or "" if absent.
std::string value_after(const std::vector<std::string>& v, const std::string& flag) {
    for (std::size_t i = 0; i + 1 < v.size(); ++i) {
        if (v[i] == flag) return v[i + 1];
    }
    return {};
}

// --- service_args (shared across all backends) ------------------------------

TEST(ServiceArgs, ForwardsEveryPopulatedFlagAsPairs) {
    const auto args = service_args(sample_config());

    EXPECT_EQ(value_after(args, "--name"), "rig-1");
    EXPECT_EQ(value_after(args, "--models-dir"), "/var/lib/mass/models");
    EXPECT_EQ(value_after(args, "--log-level"), "debug");
    EXPECT_EQ(value_after(args, "--log-file"), "/var/log/mass-worker.log");
    EXPECT_EQ(value_after(args, "--vram-headroom-pct"), "80");
}

TEST(ServiceArgs, NeverForwardsConnectionSecrets) {
    // The rendered service definition (SCM binPath / unit / plist) is
    // world-readable; the token and its companion connection flags must never
    // appear there. The service reads them from the 0600 credentials file.
    const auto args = service_args(sample_config());
    EXPECT_FALSE(contains(args, "--token"));
    EXPECT_FALSE(contains(args, "--mass-url"));
    EXPECT_FALSE(contains(args, "--ca-file"));
}

TEST(ServiceArgs, OmitsEmptyOptionsButAlwaysEmitsVramHeadroom) {
    ServiceConfig cfg;
    cfg.exe_path = "/usr/bin/mass-worker-llama-cpp";
    cfg.vram_headroom_pct = 50;
    // Everything else empty.

    const auto args = service_args(cfg);

    EXPECT_FALSE(contains(args, "--token"));
    EXPECT_FALSE(contains(args, "--ca-file"));
    EXPECT_FALSE(contains(args, "--name"));
    EXPECT_FALSE(contains(args, "--mass-url"));
    EXPECT_FALSE(contains(args, "--log-file"));
    // vram-headroom is never empty (it's an int), so it's always forwarded.
    EXPECT_EQ(value_after(args, "--vram-headroom-pct"), "50");
}

TEST(ServiceArgs, NeverIncludesTheExecutablePath) {
    const auto args = service_args(sample_config());
    EXPECT_FALSE(contains(args, "/opt/mass-worker/bin/mass-worker-llama-cpp"));
}

TEST(ServiceArgs, ForwardsTheDataDirSoTheWorkerFindsItsCredentials) {
    // The worker loads its config + 0600 credentials from --data-dir. A
    // non-default install (User scope under $HOME, or an operator's --data-dir)
    // must forward that location, or the worker reads the compiled-in system dir,
    // finds nothing, and dials the default MASS URL instead of the configured one.
    ServiceConfig cfg = sample_config();
    cfg.data_dir = "/home/op/.local/share/mass-worker-llama-cpp";
    EXPECT_EQ(value_after(service_args(cfg), "--data-dir"),
              "/home/op/.local/share/mass-worker-llama-cpp");
}

TEST(ServiceArgs, ResolvesAnEmptyDataDirToTheDefaultRatherThanForwardingEmpty) {
    // An empty data_dir means "the per-OS default"; forwarding an empty value
    // would make the worker's own --data-dir parse to "", which then resolves
    // differently. Forward the resolved default so both sides agree.
    ServiceConfig cfg = sample_config();
    cfg.data_dir = "";
    EXPECT_EQ(value_after(service_args(cfg), "--data-dir"), mass_worker::service_data_dir());
}

// --- Path finalization (shared) ---------------------------------------------

TEST(ServicePaths, DataDirIsNamedAfterTheService) {
    const std::string dir = mass_worker::service_data_dir();
    EXPECT_FALSE(dir.empty());
    // Ends with the service name regardless of the per-OS base.
    EXPECT_NE(dir.find("mass-worker-llama-cpp"), std::string::npos);
}

TEST(ServicePaths, AbsolutizesRelativeModelsDirUnderDataDir) {
    // The default (empty data_dir) resolves to the System-scope dir, which only
    // root can create; in production finalize runs inside the elevated
    // --install-service. Unprivileged test runs can only cover it when the dir
    // already exists or is creatable.
    std::error_code ec;
    fs::create_directories(mass_worker::service_data_dir(), ec);
    if (ec) GTEST_SKIP() << "system data dir needs root: " << ec.message();

    ServiceConfig cfg;
    cfg.exe_path = mass_worker::current_executable_path();
    cfg.models_dir = "models";  // relative — the problematic default
    cfg.log_file = "";          // empty — also gets a default

    ASSERT_TRUE(mass_worker::finalize_service_paths(cfg));

    EXPECT_NE(cfg.models_dir.find(mass_worker::service_data_dir()), std::string::npos);
    EXPECT_FALSE(cfg.log_file.empty());
    // The log file is now absolute, never the bare relative default.
    EXPECT_NE(cfg.log_file, "mass-worker-llama-cpp.log");
}

TEST(ServicePaths, HonorsExplicitDataDir) {
    // When cfg.data_dir is set (--data-dir / the wizard's data-dir prompt), a
    // relative models-dir absolutizes under THAT dir, not the per-OS default.
    const fs::path tmp = fs::temp_directory_path() / "mass-svc-paths-test" /
                         std::to_string(::testing::UnitTest::GetInstance()->random_seed());
    fs::remove_all(tmp);

    ServiceConfig cfg;
    cfg.exe_path = mass_worker::current_executable_path();
    cfg.data_dir = tmp.string();
    cfg.models_dir = "models";  // relative
    cfg.log_file = "";

    ASSERT_TRUE(mass_worker::finalize_service_paths(cfg));
    EXPECT_NE(cfg.models_dir.find(tmp.string()), std::string::npos);
    EXPECT_TRUE(fs::exists(tmp));  // the data dir was created

    fs::remove_all(tmp);
}

TEST(ServicePaths, LeavesAbsolutePathsUntouched) {
    // Hermetic: an explicit data_dir keeps finalize away from the root-only
    // System-scope default; the assertions only concern the absolute paths.
    const fs::path tmp = fs::temp_directory_path() / "mass-svc-abs-test" /
                         std::to_string(::testing::UnitTest::GetInstance()->random_seed());
    fs::remove_all(tmp);

    ServiceConfig cfg;
    cfg.data_dir = tmp.string();
#ifdef _WIN32
    cfg.models_dir = "D:\\data\\models";
    cfg.log_file = "D:\\logs\\worker.log";
#else
    cfg.models_dir = "/srv/models";
    cfg.log_file = "/srv/worker.log";
#endif
    ASSERT_TRUE(mass_worker::finalize_service_paths(cfg));
#ifdef _WIN32
    EXPECT_EQ(cfg.models_dir, "D:\\data\\models");
    EXPECT_EQ(cfg.log_file, "D:\\logs\\worker.log");
#else
    EXPECT_EQ(cfg.models_dir, "/srv/models");
    EXPECT_EQ(cfg.log_file, "/srv/worker.log");
#endif
    fs::remove_all(tmp);
}

// --- Install staging --------------------------------------------------------

TEST(InstallDir, DefaultIsNamedAfterTheServiceAndDistinctFromData) {
    const std::string install = mass_worker::default_install_dir();
    EXPECT_FALSE(install.empty());
    EXPECT_NE(install.find("mass-worker-llama-cpp"), std::string::npos);
    // Code/deps and mutable state live in different trees.
    EXPECT_NE(install, mass_worker::service_data_dir());
}

#ifndef _WIN32
// The per-user dirs (User scope). The data dir honours $XDG_DATA_HOME; the
// install dir is the fixed ~/.local/lib analogue of /usr/lib (NOT XDG-driven, to
// keep program code out of the data root). Pure getenv-driven functions, no
// privileged side effects. Save/restore env so cases don't bleed.
TEST(UserDirs, DataHonorsXdgDataHome) {
    const char* prev = std::getenv("XDG_DATA_HOME");
    const std::string saved = prev ? prev : "";
    setenv("XDG_DATA_HOME", "/tmp/xdgtest-data", 1);
    EXPECT_EQ(mass_worker::user_data_dir(), "/tmp/xdgtest-data/mass-worker-llama-cpp");
    if (prev)
        setenv("XDG_DATA_HOME", saved.c_str(), 1);
    else
        unsetenv("XDG_DATA_HOME");
}

#ifdef __linux__
TEST(UserDirs, InstallIsUnderLocalLibNotTheDataRoot) {
    // Program code lands in ~/.local/lib (the /usr/lib analogue), never under the
    // XDG data root — even when XDG_DATA_HOME is set, the install must not move
    // there (that's the data dir's home).
    setenv("XDG_DATA_HOME", "/tmp/xdgtest-data", 1);
    const std::string ui = mass_worker::user_install_dir();
    EXPECT_NE(ui.find("/.local/lib/mass-worker-llama-cpp"), std::string::npos);
    EXPECT_EQ(ui.find("/tmp/xdgtest-data"), std::string::npos);
    unsetenv("XDG_DATA_HOME");
}
#endif

TEST(UserDirs, InstallAndDataAreDistinctAndNotSystem) {
    unsetenv("XDG_DATA_HOME");
    const std::string ui = mass_worker::user_install_dir();
    const std::string ud = mass_worker::user_data_dir();
    EXPECT_NE(ui, ud);
    EXPECT_NE(ui, mass_worker::default_install_dir());
    EXPECT_NE(ud, mass_worker::service_data_dir());
}
#endif  // !_WIN32

// --- path_is_writable (drives the "why elevation is needed" explanation) -----

TEST(PathWritable, ExistingWritableDirIsWritable) {
    const fs::path dir = fs::temp_directory_path() / "mass-wr-test" /
                         std::to_string(::testing::UnitTest::GetInstance()->random_seed());
    fs::create_directories(dir);
    EXPECT_TRUE(mass_worker::path_is_writable(dir.string()));
    fs::remove_all(dir);
}

TEST(PathWritable, NonexistentTargetUnderWritableParentIsWritable) {
    // The install dir often doesn't exist yet — writability is judged by the
    // nearest existing ancestor (the dir it would be created under).
    const fs::path base = fs::temp_directory_path() / "mass-wr-test" /
                          std::to_string(::testing::UnitTest::GetInstance()->random_seed());
    fs::create_directories(base);
    const fs::path target = base / "does" / "not" / "exist" / "yet";
    EXPECT_FALSE(fs::exists(target));
    EXPECT_TRUE(mass_worker::path_is_writable(target.string()));
    fs::remove_all(base);
}

TEST(PathWritable, EmptyPathIsNotWritable) {
    EXPECT_FALSE(mass_worker::path_is_writable(""));
}

#ifndef _WIN32
TEST(PathWritable, ReadOnlyDirIsNotWritable) {
    // Skip as root: uid 0 bypasses the permission bits, so the probe would
    // (correctly) report writable and this case can't be exercised.
    if (::geteuid() == 0) GTEST_SKIP() << "running as root; permission bits bypassed";
    const fs::path dir = fs::temp_directory_path() / "mass-wr-ro" /
                         std::to_string(::testing::UnitTest::GetInstance()->random_seed());
    fs::create_directories(dir);
    fs::permissions(dir, fs::perms::owner_read | fs::perms::owner_exec, fs::perm_options::replace);
    EXPECT_FALSE(mass_worker::path_is_writable(dir.string()));
    // A not-yet-created child of a read-only dir is likewise not creatable.
    EXPECT_FALSE(mass_worker::path_is_writable((dir / "child").string()));
    fs::permissions(dir, fs::perms::owner_all, fs::perm_options::replace);  // so cleanup works
    fs::remove_all(dir);
}

TEST(PathWritable, SystemDirsTypicallyNotWritableUnprivileged) {
    // The whole point: the default System install/data dirs are NOT writable by an
    // unprivileged user, which is what makes elevation necessary. Skip as root.
    if (::geteuid() == 0) GTEST_SKIP() << "running as root; system dirs are writable";
    // /var/lib and /opt (or their nearest existing ancestor) shouldn't be
    // user-writable on a normal box; assert at least one default reports so rather
    // than pinning an exact path (environments vary).
    const bool any_locked = !mass_worker::path_is_writable(mass_worker::service_data_dir()) ||
                            !mass_worker::path_is_writable(mass_worker::default_install_dir());
    EXPECT_TRUE(any_locked);
}
#endif  // !_WIN32

TEST(InstallDir, StagedExePathIsUnderInstallDir) {
#ifdef _WIN32
    const std::string dir = "C:\\opt\\mass";
#else
    const std::string dir = "/opt/mass";
#endif
    const std::string staged = mass_worker::staged_exe_path(dir);
    EXPECT_EQ(staged.find(dir), 0u);  // begins with the install dir
    // The staged binary is always the WORKER (mass-worker-llama-cpp[.exe]),
    // independent of which binary did the staging — so the installer
    // (mass-worker-setup) registers a worker, not a copy of itself.
    EXPECT_NE(staged.find("mass-worker-llama-cpp"), std::string::npos);
    const std::string leaf = fs::path(staged).filename().string();
#ifdef _WIN32
    EXPECT_EQ(leaf, "mass-worker-llama-cpp.exe");
#else
    EXPECT_EQ(leaf, "mass-worker-llama-cpp");
#endif
}

TEST(Stage, CopiesBinaryAndLibrariesIntoInstallDir) {
    // Stage this test binary's own directory into a temp install dir and confirm
    // the executable lands there. The test binary sits beside the worker's
    // shared libraries in the build tree, so this exercises the real copy path.
    const fs::path dst = fs::temp_directory_path() / "mass-stage-test" /
                         std::to_string(::testing::UnitTest::GetInstance()->random_seed());
    fs::remove_all(dst);

    auto staged = mass_worker::stage_install(dst.string());
    ASSERT_TRUE(staged.has_value()) << (staged ? "" : staged.error().message);
    EXPECT_EQ(*staged, mass_worker::staged_exe_path(dst.string()));
    EXPECT_TRUE(fs::exists(*staged));
    EXPECT_EQ(fs::path(*staged).parent_path(), dst);

    fs::remove_all(dst);
}

TEST(Stage, SelfCopyIntoOwnDirIsANoOp) {
    // Staging into the directory the binary already lives in must not error
    // (an in-place reinstall/upgrade) — it just returns the existing path.
    const fs::path own = fs::path(mass_worker::current_executable_path()).parent_path();
    auto staged = mass_worker::stage_install(own.string());
    ASSERT_TRUE(staged.has_value()) << (staged ? "" : staged.error().message);
    EXPECT_EQ(*staged, mass_worker::staged_exe_path(own.string()));
    EXPECT_TRUE(fs::exists(*staged));
}

TEST(Stage, RemoveDeletesStagedDirectory) {
    // Stage into a temp dir, then remove it — the inverse round-trip.
    const fs::path dst = fs::temp_directory_path() / "mass-unstage-test" /
                         std::to_string(::testing::UnitTest::GetInstance()->random_seed());
    fs::remove_all(dst);

    auto staged = mass_worker::stage_install(dst.string());
    ASSERT_TRUE(staged.has_value()) << (staged ? "" : staged.error().message);
    ASSERT_TRUE(fs::exists(dst));

    bool self_skipped = true;
    auto removed = mass_worker::remove_staged_install(dst.string(), self_skipped);
    ASSERT_TRUE(removed.has_value()) << (removed ? "" : removed.error().message);
    EXPECT_FALSE(self_skipped);
    EXPECT_GT(*removed, 0u);
    EXPECT_FALSE(fs::exists(dst));
}

TEST(Stage, RemoveMissingDirIsNotAnError) {
    const fs::path gone = fs::temp_directory_path() / "mass-never-existed" /
                          std::to_string(::testing::UnitTest::GetInstance()->random_seed());
    fs::remove_all(gone);

    bool self_skipped = true;
    auto removed = mass_worker::remove_staged_install(gone.string(), self_skipped);
    ASSERT_TRUE(removed.has_value()) << (removed ? "" : removed.error().message);
    EXPECT_EQ(*removed, 0u);
    EXPECT_FALSE(self_skipped);
}

TEST(Stage, RemoveRefusesOwnDirectory) {
    // Removing the directory the running binary lives in must be refused — else
    // an uninstall from the installed copy would delete the live executable.
    const fs::path own = fs::path(mass_worker::current_executable_path()).parent_path();
    bool self_skipped = false;
    auto removed = mass_worker::remove_staged_install(own.string(), self_skipped);
    ASSERT_TRUE(removed.has_value()) << (removed ? "" : removed.error().message);
    EXPECT_TRUE(self_skipped);
    EXPECT_EQ(*removed, 0u);
    EXPECT_TRUE(fs::exists(own));  // still there
}

// --- Appended payload (self-extracting installer) ---------------------------

TEST(Payload, PlainBinaryHasNoAppendedPayload) {
    // The test binary is not a packaged installer — no trailer, so the reader
    // reports no payload and extraction refuses rather than corrupting a dir.
    EXPECT_FALSE(mass_worker::has_appended_payload());

    const fs::path dst = fs::temp_directory_path() / "mass-nopayload-test";
    auto r = mass_worker::extract_appended_payload(dst.string());
    EXPECT_FALSE(r.has_value());
}

TEST(InstallRecord, PathIsUnderTheDefaultDataDir) {
    const fs::path rec = mass_worker::install_record_path();
    EXPECT_EQ(rec.parent_path(), fs::path(mass_worker::service_data_dir()));
    EXPECT_EQ(rec.filename().string(), "install.record");
}

TEST(InstallRecord, RoundTripsInstallAndDataDirs) {
    // Skip if a real install record exists — don't clobber a developer's actual
    // install on this machine. (The record lives at a fixed machine location;
    // there's no test seam to redirect it, and adding one just for the test
    // would be abstraction for a hypothetical.)
    if (mass_worker::load_install_record().has_value()) {
        GTEST_SKIP() << "an install record already exists; skipping to avoid clobbering it";
    }

    const mass_worker::InstallRecord in{
        .install_dir = "D:/Programs/MASS Workers/mass-worker-llama-cpp",
        .data_dir = "D:/Programs/MASS Workers Data/mass-worker-llama-cpp",
    };
    // Writing needs the default data dir to be creatable (admin for the real
    // ProgramData on a locked-down box); skip rather than fail if it isn't.
    if (!mass_worker::save_install_record(in)) {
        GTEST_SKIP() << "default data dir not writable in this environment";
    }

    const auto out = mass_worker::load_install_record();
    ASSERT_TRUE(out.has_value());
    EXPECT_EQ(out->install_dir, in.install_dir);
    EXPECT_EQ(out->data_dir, in.data_dir);
    EXPECT_EQ(out->scope, mass_worker::ServiceScope::System);  // default scope

    mass_worker::remove_install_record();
    EXPECT_FALSE(mass_worker::load_install_record().has_value());
}

#ifndef _WIN32
TEST(InstallRecord, UserScopeRoundTripsScopeAndLoadsFromUserPath) {
    // A User-scope record persists scope=user and lands under the per-user data
    // dir (writable without root), so this actually runs in CI. It must round-trip
    // the scope so a re-run's Remove targets the user service. Guard against
    // clobbering a real record.
    if (mass_worker::load_install_record().has_value()) {
        GTEST_SKIP() << "an install record already exists; skipping to avoid clobbering it";
    }
    const mass_worker::InstallRecord in{
        .install_dir = mass_worker::user_install_dir(),
        .data_dir = mass_worker::user_data_dir(),
        .scope = mass_worker::ServiceScope::User,
    };
    if (!mass_worker::save_install_record(in, mass_worker::ServiceScope::User)) {
        GTEST_SKIP() << "per-user data dir not writable in this environment";
    }
    const auto out = mass_worker::load_install_record();
    ASSERT_TRUE(out.has_value());
    EXPECT_EQ(out->install_dir, in.install_dir);
    EXPECT_EQ(out->data_dir, in.data_dir);
    EXPECT_EQ(out->scope, mass_worker::ServiceScope::User);  // the key assertion

    mass_worker::remove_install_record(mass_worker::ServiceScope::User);
    EXPECT_FALSE(mass_worker::load_install_record().has_value());
}
#endif  // !_WIN32

TEST(InstallRecord, AbsentRecordLoadsAsNullopt) {
    // After removal there is no record; load reports absence rather than an
    // empty-but-present record. Guarded the same way as the round-trip.
    if (mass_worker::load_install_record().has_value()) {
        GTEST_SKIP() << "an install record already exists; skipping";
    }
    EXPECT_FALSE(mass_worker::load_install_record().has_value());
}

// --- Per-OS definition rendering (host-OS only) -----------------------------

#ifdef __linux__
TEST(SystemdUnit, RendersExecStartWithQuotedExeAndForwardedFlags) {
    const std::string unit = mass_worker::render_systemd_unit(sample_config());

    EXPECT_NE(unit.find("[Service]"), std::string::npos);
    EXPECT_NE(unit.find("Restart=on-failure"), std::string::npos);
    EXPECT_NE(unit.find("WantedBy=multi-user.target"), std::string::npos);
    EXPECT_NE(unit.find("After=network-online.target"), std::string::npos);
    // ExecStart quotes the exe. No connection secrets in the unit — the unit
    // file is world-readable; the service reads the 0600 credentials file.
    EXPECT_NE(unit.find("ExecStart=\"/opt/mass-worker/bin/mass-worker-llama-cpp\""),
              std::string::npos);
    EXPECT_EQ(unit.find("--token"), std::string::npos);
    EXPECT_EQ(unit.find("--mass-url"), std::string::npos);
    // WorkingDirectory is the machine data dir (/var/lib/mass-worker-llama-cpp),
    // not the binary dir — relative paths must land somewhere writable.
    EXPECT_NE(unit.find("WorkingDirectory=/var/lib/mass-worker-llama-cpp"), std::string::npos);
}

TEST(SystemdUnit, DoesNotRestartOnTheFatalExitCode) {
    // Restart=on-failure alone would turn a deterministically fatal failure (a
    // revoked join token) into an endless 5s restart loop. The unit must pin
    // the worker's fatal code so systemd leaves it failed instead.
    const std::string unit = mass_worker::render_systemd_unit(sample_config());
    EXPECT_NE(unit.find("RestartPreventExitStatus=" + std::to_string(mass_worker::kExitFatal)),
              std::string::npos)
        << unit;
    // A retryable failure must still be restarted.
    EXPECT_EQ(unit.find("RestartPreventExitStatus=" + std::to_string(mass_worker::kExitFailure)),
              std::string::npos);
}

TEST(SystemdUnit, QuotesArgumentsContainingSpaces) {
    ServiceConfig cfg = sample_config();
    cfg.models_dir = "/var/lib/mass models";
    const std::string unit = mass_worker::render_systemd_unit(cfg);
    EXPECT_NE(unit.find("\"--models-dir\" \"/var/lib/mass models\""), std::string::npos);
}

TEST(SystemdUnit, UserScopeTargetsDefaultTargetAndUserWorkdir) {
    ServiceConfig cfg = sample_config();
    cfg.scope = mass_worker::ServiceScope::User;
    cfg.data_dir = "/home/op/.local/state/mass-worker-llama-cpp";
    const std::string unit = mass_worker::render_systemd_unit(cfg);
    // A user unit reaches default.target (there is no multi-user.target in the
    // per-user manager) and works under the user data dir, not /var/lib.
    EXPECT_NE(unit.find("WantedBy=default.target"), std::string::npos);
    EXPECT_EQ(unit.find("WantedBy=multi-user.target"), std::string::npos);
    EXPECT_NE(unit.find("WorkingDirectory=/home/op/.local/state/mass-worker-llama-cpp"),
              std::string::npos);
}
#elifdef __APPLE__
TEST(LaunchdPlist, RendersLabelProgramArgumentsAndKeepAlive) {
    const std::string plist = mass_worker::render_launchd_plist(sample_config());

    EXPECT_NE(plist.find("<key>Label</key>"), std::string::npos);
    EXPECT_NE(plist.find("com.chinese-room-solutions.mass-worker-llama-cpp"), std::string::npos);
    EXPECT_NE(plist.find("<key>RunAtLoad</key>"), std::string::npos);
    // KeepAlive must be the dictionary form: the plain <true/> respawns the
    // worker even after a clean exit(0), so a requested stop comes right back.
    EXPECT_NE(plist.find("<key>KeepAlive</key>\n  <dict>\n    <key>SuccessfulExit</key>\n"
                         "    <false/>\n  </dict>"),
              std::string::npos)
        << plist;
    // exe is the first ProgramArguments string.
    EXPECT_NE(plist.find("<string>/opt/mass-worker/bin/mass-worker-llama-cpp</string>"),
              std::string::npos);
    // No connection secrets in the plist — it is world-readable; the service
    // reads the 0600 credentials file instead.
    EXPECT_EQ(plist.find("<string>--token</string>"), std::string::npos);
    // WorkingDirectory = bin dir so bundled .dylib resolve; stderr captured for
    // Metal init logs.
    EXPECT_NE(plist.find("<key>WorkingDirectory</key>"), std::string::npos);
    EXPECT_NE(plist.find("<key>StandardErrorPath</key>"), std::string::npos);
}

TEST(LaunchdPlist, XmlEscapesArgumentValues) {
    ServiceConfig cfg = sample_config();
    cfg.name = "rig&<one>";
    const std::string plist = mass_worker::render_launchd_plist(cfg);
    EXPECT_NE(plist.find("rig&amp;&lt;one&gt;"), std::string::npos);
    EXPECT_EQ(plist.find("rig&<one>"), std::string::npos);  // no raw entities
}
#elif defined(_WIN32)
TEST(ServiceBinpath, QuotesExeAndForwardsFlagsWithSentinel) {
    const std::string bin = mass_worker::render_service_binpath(sample_config());

    // Exe is quoted (it may contain "Program Files").
    EXPECT_NE(bin.find("\"/opt/mass-worker/bin/mass-worker-llama-cpp\""), std::string::npos);
    // No connection secrets in the binPath — the SCM stores it world-readable;
    // the service reads the 0600 credentials file instead.
    EXPECT_EQ(bin.find("--token"), std::string::npos);
    EXPECT_EQ(bin.find("--mass-url"), std::string::npos);
    // The SCM-launch sentinel must be present so the started process takes the
    // ServiceMain path.
    EXPECT_NE(bin.find("--run-as-service"), std::string::npos);
}

TEST(ServiceBinpath, IsServiceLaunchArgMatchesSentinel) {
    EXPECT_TRUE(mass_worker::is_service_launch_arg("--run-as-service"));
    EXPECT_FALSE(mass_worker::is_service_launch_arg("--mass-url"));
    EXPECT_FALSE(mass_worker::is_service_launch_arg(nullptr));
}

TEST(ServiceSentinel, MatchesWholeArgvTokensOnly) {
    using mass_worker::command_line_has_token;
    constexpr const wchar_t* kTok = L"--run-as-service";

    // The real shape: quoted exe + forwarded flags + trailing sentinel.
    EXPECT_TRUE(command_line_has_token(
        L"\"C:\\Program Files\\mass\\worker.exe\" --models-dir C:\\m --run-as-service", kTok));
    // Quoting around the token is transparent to CommandLineToArgvW.
    EXPECT_TRUE(command_line_has_token(L"worker.exe \"--run-as-service\"", kTok));

    // An argument merely CONTAINING the sentinel must not match — the old
    // wcsstr substring scan false-positived on all of these.
    EXPECT_FALSE(command_line_has_token(L"worker.exe --name rig---run-as-service-01", kTok));
    EXPECT_FALSE(
        command_line_has_token(L"worker.exe --log-file \"C:\\logs\\--run-as-service.log\"", kTok));
    EXPECT_FALSE(command_line_has_token(L"worker.exe --run-as-services", kTok));

    // argv[0] is the program path, not an argument.
    EXPECT_FALSE(command_line_has_token(L"--run-as-service", kTok));
    // Degenerate inputs.
    EXPECT_FALSE(command_line_has_token(L"", kTok));
    EXPECT_FALSE(command_line_has_token(nullptr, kTok));
}

TEST(ServiceSentinel, RoundTripsThroughRenderedBinpath) {
    // The binPath render and the SCM-launch detection must agree: what
    // render_service_binpath stores is exactly what GetCommandLineW hands
    // running_as_service() when the SCM starts us.
    const std::string bin = mass_worker::render_service_binpath(sample_config());
    const std::wstring wide(bin.begin(), bin.end());  // sample paths are ASCII
    EXPECT_TRUE(mass_worker::command_line_has_token(wide.c_str(), L"--run-as-service"));
}

TEST(ServiceBinpath, TrailingBackslashDoesNotEscapeClosingQuote) {
    // A Windows install dir ending in a backslash (e.g. C:\mass\) must not let
    // the trailing backslash escape the closing quote in the SCM binPath — the
    // CommandLineToArgvW rule is that backslashes before a quote are doubled.
    mass_worker::ServiceConfig cfg = sample_config();
    cfg.exe_path = "C:\\Program Files\\mass\\mass-worker-llama-cpp.exe";
    const std::string bin = mass_worker::render_service_binpath(cfg);

    // The exe must be wrapped in a balanced quote pair: a quote, the path with
    // each backslash preserved (none precede a quote here, so they stay single),
    // then a closing quote followed by a space before the next argument.
    EXPECT_NE(bin.find("\"C:\\Program Files\\mass\\mass-worker-llama-cpp.exe\" "),
              std::string::npos);
    // Sanity: equal number of quotes overall (every field balanced).
    const auto quotes = std::count(bin.begin(), bin.end(), '"');
    EXPECT_EQ(quotes % 2, 0);
}
#endif

}  // namespace
