#include "mass_worker/credentials.hpp"

#include <spdlog/spdlog.h>

#include <filesystem>
#include <fstream>
#include <sstream>

#include "mass_worker/fsutil.hpp"

namespace mass_worker {

namespace {

namespace fs = std::filesystem;

constexpr const char* kCredsFile = "credentials";
constexpr const char* kCAFile = "ca.pem";

}  // namespace

std::string credentials_path(const std::string& data_dir) {
    return (fs::path(data_dir) / kCredsFile).string();
}

bool write_credentials(const std::string& data_dir, Credentials& creds, const std::string& ca_pem) {
    std::error_code ec;
    fs::create_directories(data_dir, ec);
    if (ec) {
        spdlog::error("credentials: cannot create data dir {}: {}", data_dir, ec.message());
        return false;
    }

    // Write the CA next to the credentials when its contents are supplied, and
    // point the credentials at it. Empty ca_pem means "don't (re)write a CA":
    // creds.ca_file is left as passed, so persist_enrollment can preserve the
    // ca.pem the installer already wrote. A fresh save with no CA passes both
    // ca_pem and creds.ca_file empty.
    if (!ca_pem.empty()) {
        const fs::path ca = fs::path(data_dir) / kCAFile;
        if (!fsutil::write_private_file(ca.string(), ca_pem)) {
            spdlog::error("credentials: failed writing CA file {}", ca.string());
            return false;
        }
        creds.ca_file = ca.string();
    }

    // Minimal line-based key=value record — no parser dependency. Values never
    // contain newlines (ids/secrets/tokens/URLs/paths). The join token is
    // written only pre-enrollment: an enrolled record (worker_id + secret
    // present) drops it, so a one-time token never lingers on disk.
    std::ostringstream body;
    body << "mass_url=" << creds.mass_url << '\n'
         << "worker_id=" << creds.worker_id << '\n'
         << "worker_secret=" << creds.worker_secret << '\n';
    if (!enrolled(creds)) {
        body << "join_token=" << creds.join_token << '\n';
    }
    body << "ca_file=" << creds.ca_file << '\n' << "name=" << creds.name << '\n';

    const fs::path path = fs::path(data_dir) / kCredsFile;
    if (!fsutil::write_private_file(path.string(), body.str())) {
        spdlog::error("credentials: failed writing {}", path.string());
        return false;
    }
    return true;
}

bool persist_enrollment(const std::string& data_dir, const Credentials& enrolled_creds) {
    // The CA was already written when the installer/wizard first saved the
    // credentials; the enrolled record reuses that ca_file path verbatim, so
    // pass an empty ca_pem to skip re-copying it.
    Credentials out = enrolled_creds;
    out.join_token.clear();  // belt-and-suspenders: write_credentials also drops it
    return write_credentials(data_dir, out, /*ca_pem=*/"");
}

std::optional<Credentials> load_credentials(const std::string& data_dir) {
    const fs::path path = fs::path(data_dir) / kCredsFile;
    std::ifstream f(path, std::ios::binary);
    if (!f) return std::nullopt;

    Credentials creds;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        const auto eq = line.find('=');
        if (eq == std::string::npos) continue;
        const std::string key = line.substr(0, eq);
        std::string value = line.substr(eq + 1);
        if (!value.empty() && value.back() == '\r') value.pop_back();  // CRLF
        if (key == "mass_url")
            creds.mass_url = value;
        else if (key == "worker_id")
            creds.worker_id = value;
        else if (key == "worker_secret")
            creds.worker_secret = value;
        else if (key == "join_token")
            creds.join_token = value;
        else if (key == "ca_file")
            creds.ca_file = value;
        else if (key == "name")
            creds.name = value;
    }

    // A record without a mass_url can't drive a connection — treat as absent so
    // the worker falls back to CLI settings rather than dialing nothing.
    if (creds.mass_url.empty()) {
        return std::nullopt;
    }
    return creds;
}

}  // namespace mass_worker
