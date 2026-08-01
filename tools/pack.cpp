// pack — build the self-extracting worker installer.
//
// Concatenates a host executable (mass-worker-setup) with a compressed payload
// (the worker binary + its runtime libraries) and a fixed trailer, producing one
// self-contained installer. The format is defined in mass_worker/payload.hpp,
// written by append_payload() (src/payload_pack.cpp) and read back by
// extract_appended_payload() at install time.
//
// Usage:
//   pack --host <setup-exe> --out <installer-exe> <file>...
//
// A build-time CLI, so it reports to stderr rather than through the logger:
// `make package` runs it before anything is installed or configured.

#include <iostream>
#include <string>
#include <vector>

#include "mass_worker/payload.hpp"

int main(int argc, char** argv) {
    std::string host;
    std::string out;
    std::vector<std::string> payload;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--host" && i + 1 < argc) {
            host = argv[++i];
        } else if (a == "--out" && i + 1 < argc) {
            out = argv[++i];
        } else {
            payload.push_back(a);
        }
    }
    if (host.empty() || out.empty() || payload.empty()) {
        std::cerr << "usage: pack --host <setup-exe> --out <installer-exe> <file>...\n";
        return 2;
    }

    if (auto r = mass_worker::append_payload(host, out, payload); !r) {
        std::cerr << "pack: " << r.error().message << "\n";
        return 1;
    }
    std::cout << "pack: wrote " << out << " (" << payload.size() << " payload entries)\n";
    return 0;
}
