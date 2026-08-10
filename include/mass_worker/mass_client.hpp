#pragma once

#include <grpcpp/channel.h>
#include <grpcpp/client_context.h>
#include <grpcpp/support/status_code_enum.h>
#include <grpcpp/support/sync_stream.h>

#include <memory>
#include <string>
#include <string_view>

#include "worker/worker.grpc.pb.h"

namespace mass_worker {

// Connection settings. mass_url accepts http://host:port or https://host:port;
// the scheme determines whether TLS is used. ca_pem (optional) is a PEM bundle
// for verifying self-signed MASS server certs — if empty and the URL is https,
// the system trust store is used.
struct MassClientConfig {
    std::string mass_url;
    // Bearer credential sent as metadata "authorization: Bearer <auth_token>".
    // Enrolling: the one-time join token. Enrolled: the per-worker secret.
    std::string auth_token;
    // Server-assigned worker identity, sent as metadata "x-mass-worker-id:
    // <worker_id>" on every enrolled connect. Empty while enrolling (no identity
    // yet — MASS assigns one and returns it in WorkerEnrolled).
    std::string worker_id;
    std::string ca_pem;  // optional CA bundle (file contents, not path); empty → system roots
};

// The gRPC dial target derived from a MASS URL: the scheme selects TLS, the
// target is host[:port] only. Anything a gRPC dial target cannot express —
// userinfo (user:pass@), a path, query, or fragment — is stripped with a
// warning instead of silently producing a bogus target; a bare trailing "/"
// is stripped silently (harmless copy-paste artifact). Exposed for testing.
struct ParsedUrl {
    std::string target;
    bool use_tls;
};
[[nodiscard]] ParsedUrl parse_url(const std::string& url);

// Why an enrollment attempt died, phrased for the operator. The gRPC status is
// the diagnosis, never the mere presence of a join token: MASS answers every
// credential problem — no token, invalid or expired token, unknown or revoked
// worker, wrong per-worker secret — with UNAUTHENTICATED, and every server-side
// failure (a failed insert, a broken schema) with INTERNAL or another code
// carrying the real cause. So only the credential codes earn a token
// hypothesis; anything else is reported exactly as MASS phrased it, so a server
// fault never sends the operator hunting for a token they don't need. An empty
// message means the stream simply closed, leaving only the code to report.
// Pure; exposed for testing.
[[nodiscard]] std::string enrollment_failure_message(bool had_token, grpc::StatusCode code,
                                                     std::string_view server_message);

// Whether a Connect-stream failure with this status is worth reconnecting for.
// Credential errors, malformed requests and a refused registration (no common
// wire-protocol version, no such runtime) need an operator to change something
// the worker cannot change by retrying, so they end the process; every other
// code describes a server-side condition that may well be repaired while the
// worker is up, so the worker backs off and retries instead of exiting. Pure;
// exposed for testing.
[[nodiscard]] bool connect_error_is_fatal(grpc::StatusCode code);

// MassClient owns a long-lived grpc::Channel to MASS plus a worker-hub stub
// built on top of it. Cheap to construct, cheap to copy the inner stub_;
// the channel itself is reused across reconnects (gRPC manages the underlying
// HTTP/2 connection lifecycle internally — TCP gets dialed lazily on first
// RPC and reused for the channel's lifetime).
class MassClient {
public:
    explicit MassClient(MassClientConfig cfg);

    ~MassClient() = default;

    MassClient(const MassClient&) = delete;
    MassClient& operator=(const MassClient&) = delete;

    using Stream =
        grpc::ClientReaderWriter<mass::v1::worker::WorkerMessage, mass::v1::worker::HubMessage>;

    // Open a fresh Connect() bidi stream. Returns the stream + the
    // ClientContext that owns its deadlines/cancellation. Caller keeps the
    // context alive for the stream's lifetime; cancel via context->TryCancel().
    struct Connection {
        std::unique_ptr<grpc::ClientContext> context;
        std::unique_ptr<Stream> stream;
    };
    [[nodiscard]] Connection open_connect_stream();

    [[nodiscard]] const std::string& mass_url() const { return cfg_.mass_url; }

private:
    MassClientConfig cfg_;
    std::shared_ptr<grpc::Channel> channel_;
    std::unique_ptr<mass::v1::worker::WorkerHub::Stub> stub_;
};

}  // namespace mass_worker
