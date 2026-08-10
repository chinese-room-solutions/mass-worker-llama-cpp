#include "mass_worker/mass_client.hpp"

#include <grpcpp/support/status_code_enum.h>
#include <gtest/gtest.h>
#include <spdlog/spdlog.h>

#include <string>
#include <vector>

namespace {

struct ParseUrlCase {
    std::string name;
    std::string url;
    std::string want_target;
    bool want_tls;
};

class ParseUrlTest : public ::testing::TestWithParam<ParseUrlCase> {
protected:
    // parse_url warns on stripped components; keep test output quiet.
    void SetUp() override { spdlog::set_level(spdlog::level::off); }
    void TearDown() override { spdlog::set_level(spdlog::level::info); }
};

TEST_P(ParseUrlTest, TargetAndTls) {
    const auto& c = GetParam();
    const auto got = mass_worker::parse_url(c.url);
    EXPECT_EQ(got.target, c.want_target);
    EXPECT_EQ(got.use_tls, c.want_tls);
}

INSTANTIATE_TEST_SUITE_P(
    Table, ParseUrlTest,
    ::testing::Values(
        // Scheme variants.
        ParseUrlCase{"bare_host", "localhost", "localhost", false},
        ParseUrlCase{"host_port_no_scheme", "localhost:3455", "localhost:3455", false},
        ParseUrlCase{"http_host_port", "http://mass.local:3455", "mass.local:3455", false},
        ParseUrlCase{"https_host_port", "https://mass.local:3455", "mass.local:3455", true},
        ParseUrlCase{"https_host_only", "https://mass.local", "mass.local", true},
        // Scheme match is case-sensitive and whole-prefix: an unknown scheme
        // is left alone (insecure default), not half-stripped.
        ParseUrlCase{"unknown_scheme_kept", "grpc-host:3455", "grpc-host:3455", false},
        // Trailing slash — common copy-paste artifact, stripped silently.
        ParseUrlCase{"trailing_slash", "http://mass.local:3455/", "mass.local:3455", false},
        ParseUrlCase{"tls_trailing_slash", "https://mass.local/", "mass.local", true},
        // Paths / query / fragment can't ride a gRPC dial target — stripped
        // (with a warning) instead of producing a bogus "host/prefix" target.
        ParseUrlCase{"path", "https://mass.local:3455/prefix", "mass.local:3455", true},
        ParseUrlCase{"deep_path_and_query", "http://mass.local/api/v1?x=1", "mass.local", false},
        ParseUrlCase{"query_without_path", "http://mass.local:3455?x=1", "mass.local:3455", false},
        // Userinfo is stripped too; credentials travel via --token.
        ParseUrlCase{"userinfo", "http://user:pass@mass.local:3455", "mass.local:3455", false},
        ParseUrlCase{"userinfo_and_path", "https://op@mass.local/x", "mass.local", true}),
    [](const ::testing::TestParamInfo<ParseUrlCase>& tpi) { return tpi.param.name; });

// --- enrollment_failure_message: what the operator is told to go fix --------

struct EnrollmentMessageCase {
    std::string name;
    bool had_token;
    grpc::StatusCode code;
    std::string server_message;
    std::vector<std::string> want_contains;
    std::vector<std::string> want_absent;
};

class EnrollmentMessageTest : public ::testing::TestWithParam<EnrollmentMessageCase> {};

TEST_P(EnrollmentMessageTest, Wording) {
    const auto& c = GetParam();
    const std::string got =
        mass_worker::enrollment_failure_message(c.had_token, c.code, c.server_message);
    for (const auto& want : c.want_contains) {
        EXPECT_NE(got.find(want), std::string::npos)
            << "expected substring \"" << want << "\" in: " << got;
    }
    for (const auto& unwanted : c.want_absent) {
        EXPECT_EQ(got.find(unwanted), std::string::npos)
            << "unexpected substring \"" << unwanted << "\" in: " << got;
    }
}

// The message MASS really sent the night this bug was found: a broken schema
// reported as INTERNAL. The worker held the diagnosis and still blamed a
// missing join token — the regression these cases pin down.
const char* const kSchemaFailure =
    "enrolling worker: persisting worker: inserting worker: SQL logic error: no such table: "
    "workers (1)";

INSTANTIATE_TEST_SUITE_P(
    Table, EnrollmentMessageTest,
    ::testing::Values(
        // UNAUTHENTICATED is the only code that means "credentials" — there the
        // token hypothesis is the right one, split by whether we held a token.
        EnrollmentMessageCase{"unauthenticated_without_token",
                              false,
                              grpc::StatusCode::UNAUTHENTICATED,
                              "worker enrollment requires a join token (authorization bearer)",
                              {"join token", "--token", "MASS_JOIN_TOKEN",
                               "worker enrollment requires a join token", "(code 16)"},
                              {"invalid or expired"}},
        EnrollmentMessageCase{"unauthenticated_with_token",
                              true,
                              grpc::StatusCode::UNAUTHENTICATED,
                              "invalid or expired join token",
                              {"join token", "invalid or expired", "(code 16)"},
                              {"--token", "MASS_JOIN_TOKEN"}},
        EnrollmentMessageCase{"permission_denied_with_token",
                              true,
                              grpc::StatusCode::PERMISSION_DENIED,
                              "forbidden",
                              {"join token", "invalid or expired", "forbidden", "(code 7)"},
                              {}},
        // Everything else is the server's own failure: report it verbatim and
        // never mention a token, whether or not one was supplied.
        EnrollmentMessageCase{"internal_without_token",
                              false,
                              grpc::StatusCode::INTERNAL,
                              kSchemaFailure,
                              {"no such table: workers", "SQL logic error", "(code 13)"},
                              {"token", "--token", "MASS_JOIN_TOKEN"}},
        EnrollmentMessageCase{"internal_with_token",
                              true,
                              grpc::StatusCode::INTERNAL,
                              kSchemaFailure,
                              {"no such table: workers", "(code 13)"},
                              {"token"}},
        // A refused register frame: the wire-protocol lists don't intersect.
        // The token is irrelevant, so it must not be mentioned even though one
        // was held; the server's message names both lists.
        EnrollmentMessageCase{
            "failed_precondition_protocol_mismatch",
            true,
            grpc::StatusCode::FAILED_PRECONDITION,
            "worker speaks protocol versions [1], MASS speaks [2 3]",
            {"rejected the registration", "protocol versions [1]", "[2 3]", "(code 9)"},
            {"token"}},
        // No message at all: a bare transport close. Report the code and stop —
        // any hypothesis would be invention.
        EnrollmentMessageCase{"empty_message_transport_close",
                              false,
                              grpc::StatusCode::UNAVAILABLE,
                              "",
                              {"closed the stream without replying", "(code 14)"},
                              {"token", ": ("}},
        EnrollmentMessageCase{"empty_message_with_token",
                              true,
                              grpc::StatusCode::UNAVAILABLE,
                              "",
                              {"closed the stream without replying", "(code 14)"},
                              {"token"}}),
    [](const ::testing::TestParamInfo<EnrollmentMessageCase>& tpi) { return tpi.param.name; });

// --- connect_error_is_fatal: exit vs. back off and retry --------------------

struct ConnectFatalCase {
    std::string name;
    grpc::StatusCode code;
    bool want_fatal;
};

class ConnectFatalTest : public ::testing::TestWithParam<ConnectFatalCase> {};

TEST_P(ConnectFatalTest, Classification) {
    const auto& c = GetParam();
    EXPECT_EQ(mass_worker::connect_error_is_fatal(c.code), c.want_fatal);
}

INSTANTIATE_TEST_SUITE_P(
    Table, ConnectFatalTest,
    ::testing::Values(
        // Only an operator can supply a valid token or a well-formed request;
        // reconnecting with the same inputs would loop forever.
        ConnectFatalCase{"unauthenticated", grpc::StatusCode::UNAUTHENTICATED, true},
        ConnectFatalCase{"permission_denied", grpc::StatusCode::PERMISSION_DENIED, true},
        ConnectFatalCase{"invalid_argument", grpc::StatusCode::INVALID_ARGUMENT, true},
        // A refused registration: no wire-protocol version in common, or no
        // such runtime installed. Only a different binary (or server) clears it.
        ConnectFatalCase{"failed_precondition", grpc::StatusCode::FAILED_PRECONDITION, true},
        // Server-side conditions: fixable while the worker is up, so it should
        // still be there (backing off) when they are.
        ConnectFatalCase{"internal", grpc::StatusCode::INTERNAL, false},
        ConnectFatalCase{"unavailable", grpc::StatusCode::UNAVAILABLE, false},
        ConnectFatalCase{"deadline_exceeded", grpc::StatusCode::DEADLINE_EXCEEDED, false},
        ConnectFatalCase{"unknown", grpc::StatusCode::UNKNOWN, false},
        ConnectFatalCase{"resource_exhausted", grpc::StatusCode::RESOURCE_EXHAUSTED, false}),
    [](const ::testing::TestParamInfo<ConnectFatalCase>& tpi) { return tpi.param.name; });

}  // namespace
