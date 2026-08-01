#include "mass_worker/credentials.hpp"

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <sstream>

namespace {

namespace fs = std::filesystem;
using mass_worker::Credentials;

// A unique temp dir per test, cleaned up on teardown.
fs::path temp_dir(const std::string& tag) {
    const fs::path d = fs::temp_directory_path() /
                       ("mass-creds-test-" + tag + "-" +
                        std::to_string(::testing::UnitTest::GetInstance()->random_seed()));
    fs::remove_all(d);
    fs::create_directories(d);
    return d;
}

std::string read_all(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// An enrolled record (worker_id + secret) round-trips, and the removed
// worker_token key never appears on disk.
TEST(Credentials, EnrolledRoundTrips) {
    const fs::path dir = temp_dir("rt");

    Credentials creds{
        .mass_url = "https://mass.example:3455",
        .worker_id = "wrk_abc123",
        .worker_secret = "sec-xyz789",  // gitleaks:allow (dummy fixture)
        .join_token = {},
        .ca_file = {},
        .name = "rig-1",
    };
    ASSERT_TRUE(mass_worker::write_credentials(dir.string(), creds, "-----BEGIN CERT-----\nx\n"));

    // A CA was supplied, so ca_file is now set to the written ca.pem.
    EXPECT_FALSE(creds.ca_file.empty());
    EXPECT_TRUE(fs::exists(creds.ca_file));

    auto loaded = mass_worker::load_credentials(dir.string());
    ASSERT_TRUE(loaded.has_value());
    EXPECT_EQ(loaded->mass_url, "https://mass.example:3455");
    EXPECT_EQ(loaded->worker_id, "wrk_abc123");
    EXPECT_EQ(loaded->worker_secret, "sec-xyz789");  // gitleaks:allow (dummy fixture)
    EXPECT_TRUE(loaded->join_token.empty());
    EXPECT_EQ(loaded->name, "rig-1");
    EXPECT_EQ(loaded->ca_file, creds.ca_file);
    EXPECT_TRUE(mass_worker::enrolled(*loaded));

    // The retired shared-token field must never be serialized.
    const std::string body = read_all(mass_worker::credentials_path(dir.string()));
    EXPECT_EQ(body.find("worker_token"), std::string::npos);

    fs::remove_all(dir);
}

// A pre-enrollment record carries a join token but no identity; the token IS
// written (the worker needs it to enroll on first connect).
TEST(Credentials, PreEnrollmentPersistsJoinToken) {
    const fs::path dir = temp_dir("preenroll");
    Credentials creds{
        .mass_url = "http://localhost:3455",
        .worker_id = {},
        .worker_secret = {},
        .join_token = "join-abc",  // gitleaks:allow (dummy fixture)
        .ca_file = {},
        .name = "n",
    };
    ASSERT_TRUE(mass_worker::write_credentials(dir.string(), creds, ""));
    EXPECT_FALSE(mass_worker::enrolled(creds));

    auto loaded = mass_worker::load_credentials(dir.string());
    ASSERT_TRUE(loaded.has_value());
    EXPECT_EQ(loaded->join_token, "join-abc");  // gitleaks:allow (dummy fixture)
    EXPECT_TRUE(loaded->worker_id.empty());
    EXPECT_TRUE(loaded->worker_secret.empty());
    EXPECT_FALSE(mass_worker::enrolled(*loaded));
    fs::remove_all(dir);
}

// persist_enrollment rewrites a pre-enrollment record into an enrolled one: the
// identity lands, the join token is dropped from disk, and the ca_file path is
// preserved (the ca.pem is not re-copied).
TEST(Credentials, PersistEnrollmentDropsJoinToken) {
    const fs::path dir = temp_dir("persist");

    // Installer seeds the pre-enrollment record with a join token + CA.
    Credentials seed{
        .mass_url = "http://localhost:3455",
        .worker_id = {},
        .worker_secret = {},
        .join_token = "join-once",  // gitleaks:allow (dummy fixture)
        .ca_file = {},
        .name = "rig-2",
    };
    ASSERT_TRUE(mass_worker::write_credentials(dir.string(), seed, "-----BEGIN CERT-----\ny\n"));
    ASSERT_FALSE(seed.ca_file.empty());
    const std::string ca_path = seed.ca_file;

    // Worker enrolls: persist the MASS-minted identity, keep mass_url/name/ca.
    Credentials enrolled_creds{
        .mass_url = seed.mass_url,
        .worker_id = "wrk_minted",
        .worker_secret = "sec-minted",  // gitleaks:allow (dummy fixture)
        .join_token = "join-once",      // gitleaks:allow — should be dropped on write
        .ca_file = ca_path,
        .name = seed.name,
    };
    ASSERT_TRUE(mass_worker::persist_enrollment(dir.string(), enrolled_creds));

    auto loaded = mass_worker::load_credentials(dir.string());
    ASSERT_TRUE(loaded.has_value());
    EXPECT_TRUE(mass_worker::enrolled(*loaded));
    EXPECT_EQ(loaded->worker_id, "wrk_minted");
    EXPECT_EQ(loaded->worker_secret, "sec-minted");  // gitleaks:allow (dummy fixture)
    EXPECT_TRUE(loaded->join_token.empty());
    EXPECT_EQ(loaded->ca_file, ca_path);
    EXPECT_TRUE(fs::exists(ca_path));

    // No join token anywhere in the persisted body.
    const std::string body = read_all(mass_worker::credentials_path(dir.string()));
    EXPECT_EQ(body.find("join-once"), std::string::npos);
    EXPECT_EQ(body.find("join_token"), std::string::npos);

    fs::remove_all(dir);
}

TEST(Credentials, NoCALeavesCaFileEmptyOnFreshSave) {
    const fs::path dir = temp_dir("noca");

    Credentials creds{
        .mass_url = "http://localhost:3455",
        .worker_id = "w",
        .worker_secret = "s",  // gitleaks:allow (dummy fixture)
        .join_token = {},
        .ca_file = {},  // fresh save, no CA → stays empty
        .name = "n",
    };
    ASSERT_TRUE(mass_worker::write_credentials(dir.string(), creds, /*ca_pem=*/""));
    EXPECT_TRUE(creds.ca_file.empty());
    EXPECT_FALSE(fs::exists(dir / "ca.pem"));

    auto loaded = mass_worker::load_credentials(dir.string());
    ASSERT_TRUE(loaded.has_value());
    EXPECT_TRUE(loaded->ca_file.empty());

    fs::remove_all(dir);
}

TEST(Credentials, LoadMissingReturnsNullopt) {
    const fs::path dir = temp_dir("missing");
    EXPECT_FALSE(mass_worker::load_credentials(dir.string()).has_value());
    fs::remove_all(dir);
}

TEST(Credentials, NoURLTreatedAsAbsent) {
    const fs::path dir = temp_dir("nourl");
    // A record without mass_url can't drive a connect → load must refuse it.
    std::ofstream(mass_worker::credentials_path(dir.string())) << "worker_id=x\nworker_secret=s\n";
    EXPECT_FALSE(mass_worker::load_credentials(dir.string()).has_value());
    fs::remove_all(dir);
}

TEST(Credentials, CredentialsPathIsUnderDataDir) {
    const std::string p = mass_worker::credentials_path("/data/mass");
    EXPECT_NE(p.find("credentials"), std::string::npos);
    EXPECT_NE(p.find("mass"), std::string::npos);
}

}  // namespace
