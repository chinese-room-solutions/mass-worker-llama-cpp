#include "mass_worker/version.hpp"

namespace mass_worker {

std::string version_detail() {
    // "<ver> (<git>, <backend>)" — git omitted when this wasn't built from a
    // checkout (tarball/release build), backend always present so a deployed
    // binary reports which GPU path it carries.
    std::string s = kVersion;
    s += " (";
    if (kGitDescribe[0] != '\0') {
        s += kGitDescribe;
        s += ", ";
    }
    s += kGpuBackend;
    s += ')';
    return s;
}

std::string banner_version() {
    // The short version shown under the setup banner. Prefer the live
    // `git describe --tags --dirty --always` string (e.g. "2b9031a-dirty" on an
    // untagged checkout, "0.1.0-3-gabc1234" once tagged) so a dev build shows its
    // real commit/dirty state, mirroring the MASS installer. Fall back to the
    // compiled-in semantic version only when this wasn't built from a checkout
    // (a tarball/release build, where kGitDescribe is empty).
    if (kGitDescribe[0] != '\0') return kGitDescribe;
    return kVersion;
}

std::string version_string() {
    // The --version / log line leads with the program name; the banner uses
    // version_detail() instead (the name is already its heading).
    return "mass-worker-llama-cpp " + version_detail();
}

}  // namespace mass_worker
