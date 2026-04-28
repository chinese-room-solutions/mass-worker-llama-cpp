#include "mass_worker/mass_client.hpp"

#include <chrono>
#include <stdexcept>

#include <grpcpp/create_channel.h>
#include <grpcpp/security/credentials.h>
#include <spdlog/spdlog.h>

namespace mass_worker {

namespace {

// Strip "http://" or "https://" prefix and return (host:port, use_tls).
struct ParsedUrl {
    std::string target;
    bool use_tls;
};
ParsedUrl parse_url(const std::string& url) {
    constexpr std::string_view kHttp  = "http://";
    constexpr std::string_view kHttps = "https://";
    if (url.starts_with(kHttps)) {
        return {url.substr(kHttps.size()), /*use_tls=*/true};
    }
    if (url.starts_with(kHttp)) {
        return {url.substr(kHttp.size()), /*use_tls=*/false};
    }
    // No scheme — assume insecure to match the Go default behaviour.
    return {url, /*use_tls=*/false};
}

}  // namespace

MassClient::MassClient(MassClientConfig cfg) : cfg_(std::move(cfg)) {
    auto [target, use_tls] = parse_url(cfg_.mass_url);

    grpc::ChannelArguments args;
    // Match Go runner's behaviour: no app-level deadline on the connect stream
    // (the heartbeat liveness check is what detects dead connections). gRPC's
    // internal HTTP/2 keepalive handles the transport.
    args.SetInt(GRPC_ARG_KEEPALIVE_TIME_MS,            30 * 1000);
    args.SetInt(GRPC_ARG_KEEPALIVE_TIMEOUT_MS,         10 * 1000);
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

    channel_ = grpc::CreateCustomChannel(target, std::move(creds), args);
    stub_    = mass::v1::worker::WorkerHub::NewStub(channel_);
    spdlog::debug("mass client created (target={}, tls={})", target, use_tls);
}

MassClient::Connection MassClient::open_connect_stream() {
    auto ctx = std::make_unique<grpc::ClientContext>();
    if (!cfg_.auth_token.empty()) {
        ctx->AddMetadata("authorization", "Bearer " + cfg_.auth_token);
    }
    auto stream = stub_->Connect(ctx.get());
    return {std::move(ctx), std::move(stream)};
}

bool MassClient::wait_for_connection(std::chrono::milliseconds timeout) const {
    auto state = channel_->GetState(/*try_to_connect=*/true);
    if (state == GRPC_CHANNEL_READY) return true;
    auto deadline = std::chrono::system_clock::now() + timeout;
    return channel_->WaitForStateChange(state, deadline) &&
           channel_->GetState(false) == GRPC_CHANNEL_READY;
}

}  // namespace mass_worker
