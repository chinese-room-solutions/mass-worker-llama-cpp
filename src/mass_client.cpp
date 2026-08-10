#include "mass_worker/mass_client.hpp"

#include <grpcpp/create_channel.h>
#include <grpcpp/security/credentials.h>
#include <spdlog/fmt/fmt.h>
#include <spdlog/spdlog.h>

#include <stdexcept>
#include <string>
#include <string_view>

namespace mass_worker {

ParsedUrl parse_url(const std::string& url) {
    constexpr std::string_view kHttp = "http://";
    constexpr std::string_view kHttps = "https://";

    std::string_view rest = url;
    // No scheme — assume insecure to match the Go default behaviour.
    bool use_tls = false;
    if (rest.starts_with(kHttps)) {
        rest.remove_prefix(kHttps.size());
        use_tls = true;
    } else if (rest.starts_with(kHttp)) {
        rest.remove_prefix(kHttp.size());
    }

    // A gRPC dial target is host[:port] only. Strip anything it can't express
    // rather than dialing a bogus target (https://host/prefix would otherwise
    // try to resolve "host/prefix"). Warn so a mistyped URL is diagnosable;
    // a bare trailing "/" is a harmless copy-paste artifact, no warning.
    if (const auto cut = rest.find_first_of("/?#"); cut != std::string_view::npos) {
        const std::string_view dropped = rest.substr(cut);
        rest = rest.substr(0, cut);
        if (dropped != "/") {
            spdlog::warn(
                "mass-url: ignoring \"{}\" — gRPC targets are "
                "host:port only (url={})",
                dropped, url);
        }
    }
    if (const auto at = rest.rfind('@'); at != std::string_view::npos) {
        rest = rest.substr(at + 1);
        // Don't echo the dropped prefix: userinfo may carry a credential.
        spdlog::warn(
            "mass-url: ignoring userinfo before '@' — pass the token "
            "via --token, not the URL");
    }
    return {std::string(rest), use_tls};
}

namespace {

// ": <message> (code N)", or just " (code N)" when the status carried no
// message — so a caller can append it to any sentence without punctuation
// dangling off an empty message.
std::string status_detail(grpc::StatusCode code, std::string_view server_message) {
    const auto n = static_cast<int>(code);
    return server_message.empty() ? fmt::format(" (code {})", n)
                                  : fmt::format(": {} (code {})", server_message, n);
}

}  // namespace

std::string enrollment_failure_message(bool had_token, grpc::StatusCode code,
                                       std::string_view server_message) {
    const std::string detail = status_detail(code, server_message);

    if (code == grpc::StatusCode::UNAUTHENTICATED || code == grpc::StatusCode::PERMISSION_DENIED) {
        if (had_token) {
            return "enrollment failed: MASS rejected the join token (invalid or expired?)" + detail;
        }
        return "enrollment failed: MASS requires a join token and none was provided (pass --token "
               "or MASS_JOIN_TOKEN)" +
               detail;
    }
    if (code == grpc::StatusCode::FAILED_PRECONDITION) {
        // The register frame itself was refused (no common wire-protocol
        // version, no such runtime installed) — nothing to do with the token,
        // and the server message names what doesn't line up.
        return "enrollment failed: MASS rejected the registration" + detail;
    }
    if (server_message.empty()) {
        return "enrollment failed: MASS closed the stream without replying" + detail;
    }
    return "enrollment failed: MASS rejected the enrollment" + detail;
}

bool connect_error_is_fatal(grpc::StatusCode code) {
    switch (code) {
        case grpc::StatusCode::UNAUTHENTICATED:
        case grpc::StatusCode::PERMISSION_DENIED:
        case grpc::StatusCode::INVALID_ARGUMENT:
        case grpc::StatusCode::FAILED_PRECONDITION:
            return true;
        default:
            return false;
    }
}

MassClient::MassClient(MassClientConfig cfg) : cfg_(std::move(cfg)) {
    auto [target, use_tls] = parse_url(cfg_.mass_url);

    grpc::ChannelArguments args;
    // Match Go runner's behaviour: no app-level deadline on the connect stream
    // (the heartbeat liveness check is what detects dead connections). gRPC's
    // internal HTTP/2 keepalive handles the transport.
    args.SetInt(GRPC_ARG_KEEPALIVE_TIME_MS, 30 * 1000);
    args.SetInt(GRPC_ARG_KEEPALIVE_TIMEOUT_MS, 10 * 1000);
    args.SetInt(GRPC_ARG_KEEPALIVE_PERMIT_WITHOUT_CALLS, 1);

    std::shared_ptr<grpc::ChannelCredentials> creds;
    if (use_tls) {
        grpc::SslCredentialsOptions ssl_opts;
        if (!cfg_.ca_pem.empty()) {
            ssl_opts.pem_root_certs = cfg_.ca_pem;
        }
        creds = grpc::SslCredentials(ssl_opts);
    } else {
        creds = grpc::InsecureChannelCredentials();
    }

    channel_ = grpc::CreateCustomChannel(target, creds, args);
    stub_ = mass::v1::worker::WorkerHub::NewStub(channel_);
    spdlog::debug("mass client created (target={}, tls={})", target, use_tls);
}

MassClient::Connection MassClient::open_connect_stream() {
    auto ctx = std::make_unique<grpc::ClientContext>();
    if (!cfg_.auth_token.empty()) {
        ctx->AddMetadata("authorization", "Bearer " + cfg_.auth_token);
    }
    // An enrolled worker also identifies itself by its server-assigned ID; an
    // enrolling worker omits it (MASS mints one and returns it in WorkerEnrolled).
    if (!cfg_.worker_id.empty()) {
        ctx->AddMetadata("x-mass-worker-id", cfg_.worker_id);
    }
    auto stream = stub_->Connect(ctx.get());
    return {std::move(ctx), std::move(stream)};
}

}  // namespace mass_worker
