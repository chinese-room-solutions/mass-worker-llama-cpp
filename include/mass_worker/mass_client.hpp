#pragma once

#include <chrono>
#include <memory>
#include <string>

#include <grpcpp/channel.h>
#include <grpcpp/client_context.h>
#include <grpcpp/support/sync_stream.h>

#include "worker/worker.grpc.pb.h"

namespace mass_worker {

// Connection settings. mass_url accepts http://host:port or https://host:port;
// the scheme determines whether TLS is used. ca_pem (optional) is a PEM bundle
// for verifying self-signed MASS server certs — if empty and the URL is https,
// the system trust store is used.
struct MassClientConfig {
    std::string mass_url;
    std::string auth_token;  // optional bearer; sent as metadata "authorization: Bearer <token>"
    std::string ca_pem;      // optional CA bundle (file contents, not path); empty → system roots
};

// MassClient owns a long-lived grpc::Channel to MASS plus a worker-hub stub
// built on top of it. Cheap to construct, cheap to copy the inner stub_;
// the channel itself is reused across reconnects (gRPC manages the underlying
// HTTP/2 connection lifecycle internally — TCP gets dialed lazily on first
// RPC and reused for the channel's lifetime).
class MassClient {
public:
    explicit MassClient(MassClientConfig cfg);

    MassClient(const MassClient&) = delete;
    MassClient& operator=(const MassClient&) = delete;

    using Stream = grpc::ClientReaderWriter<
        mass::v1::worker::WorkerMessage, mass::v1::worker::HubMessage>;

    // Open a fresh Connect() bidi stream. Returns the stream + the
    // ClientContext that owns its deadlines/cancellation. Caller keeps the
    // context alive for the stream's lifetime; cancel via context->TryCancel().
    struct Connection {
        std::unique_ptr<grpc::ClientContext> context;
        std::unique_ptr<Stream> stream;
    };
    [[nodiscard]] Connection open_connect_stream();

    // Wait up to `timeout` for the underlying channel to become connected.
    // Useful as a liveness check before sending the Register frame.
    bool wait_for_connection(std::chrono::milliseconds timeout) const;

    [[nodiscard]] const std::string& mass_url() const { return cfg_.mass_url; }

private:
    MassClientConfig cfg_;
    std::shared_ptr<grpc::Channel> channel_;
    std::unique_ptr<mass::v1::worker::WorkerHub::Stub> stub_;
};

}  // namespace mass_worker
