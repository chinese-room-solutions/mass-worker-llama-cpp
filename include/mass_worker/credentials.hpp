#pragma once

#include <optional>
#include <string>

namespace mass_worker {

// Persisted connection credentials: the MASS URL, per-worker identity, and
// optional CA the worker dials with. The installer (or the interactive wizard)
// seeds the file with a one-time JOIN token; on its first-ever connect the
// worker enrolls, MASS mints a per-worker identity (worker_id + secret), and the
// worker rewrites this file with that identity and drops the join token. Every
// later launch auto-loads the identity and reconnects with no flags.
//
// Two on-disk shapes, distinguished by which fields are present:
//   - pre-enrollment: mass_url + join_token (+ name, ca) — written by the
//     installer, consumed once.
//   - enrolled:       mass_url + worker_id + worker_secret (+ name, ca) — the
//     join token is gone, and worker_secret authenticates every reconnect.
// join_token is transient: it is loaded from disk but NEVER written back once an
// identity exists.

struct Credentials {
    std::string mass_url;       // http(s)://host:port of the MASS server
    std::string worker_id;      // server-assigned identity; empty until enrolled
    std::string worker_secret;  // per-worker bearer secret; empty until enrolled
    std::string
        join_token;       // one-time enrollment token; transient, never persisted after enrollment
    std::string ca_file;  // path to a written ca.pem, or empty
    std::string name;     // worker name
};

// enrolled() is true once MASS has minted this worker's identity — both the
// worker_id and its secret are present. An enrolled worker reconnects with the
// stored secret; a non-enrolled one must enroll with its join_token.
[[nodiscard]] inline bool enrolled(const Credentials& c) {
    return !c.worker_id.empty() && !c.worker_secret.empty();
}

// The on-disk path of the credentials file under data_dir.
[[nodiscard]] std::string credentials_path(const std::string& data_dir);

// Load previously-persisted credentials, or nullopt when none exist (the worker
// then runs with its CLI-provided connection settings). A file missing the
// mass_url is treated as absent — partial records never drive a connect.
[[nodiscard]] std::optional<Credentials> load_credentials(const std::string& data_dir);

// Write credentials to data_dir (credentials file 0600, atomic temp+rename;
// ca.pem 0600 when ca_pem is non-empty). On success creds.ca_file is set to the
// written ca.pem path when ca_pem was supplied, else left as passed (so an
// enrolled rewrite can preserve the ca.pem the installer already wrote). The
// join_token is serialized ONLY for a pre-enrollment record (the installer's
// one-time token the worker consumes on first connect); once the record is
// enrolled (worker_id + secret present) the token is dropped, so a spent token
// never lingers on disk. Returns false on write failure.
[[nodiscard]] bool write_credentials(const std::string& data_dir, Credentials& creds,
                                     const std::string& ca_pem);

// persist_enrollment rewrites the credentials file after a successful enrollment:
// it stores the MASS-minted worker_id + secret, keeps mass_url/name/ca, and drops
// the join token. Reuses any ca.pem already written (does not re-copy the CA).
// The write is atomic (temp+rename). Returns false on write failure — the caller
// MUST treat that as fatal: proceeding without a persisted identity would orphan
// the server-side record on the next restart.
[[nodiscard]] bool persist_enrollment(const std::string& data_dir, const Credentials& enrolled);

}  // namespace mass_worker
