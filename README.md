# mass-worker-llama-cpp

[![CI](https://github.com/chinese-room-solutions/mass-worker-llama-cpp/actions/workflows/ci.yml/badge.svg)](https://github.com/chinese-room-solutions/mass-worker-llama-cpp/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/chinese-room-solutions/mass-worker-llama-cpp)](https://github.com/chinese-room-solutions/mass-worker-llama-cpp/releases/latest)
[![License: FSL-1.1-ALv2](https://img.shields.io/badge/License-FSL--1.1--ALv2-blue.svg)](LICENSE.md)

Native C++ MASS worker for the **llama.cpp** inference runtime.

Generated proto sources land in `build/proto/` — they're never checked in.
The wire-format source of truth is the sibling
[`mass-proto`](../mass-proto) repo.

## Dependencies

Managed via **vcpkg** in manifest mode. Required:

- gRPC ≥ 1.60 (gRPC C++ client; talks to MASS's ConnectRPC server in
  gRPC-compatibility mode)
- protobuf ≥ 5.0
- spdlog (logging)
- CLI11 (flag parsing)
- libcurl (model file downloads)
- OpenSSL (TLS, sha256 verification)
- GoogleTest (test framework)

llama.cpp itself is vendored as a git submodule — built from source so we
control GPU backend flags.

## Build

### Prerequisites

1. **CMake ≥ 3.24**, a C++20 compiler (MSVC 19.30+, GCC 11+, Clang 14+),
   `git`, `python` (for vcpkg).
2. **vcpkg**:
   ```bash
   git clone https://github.com/microsoft/vcpkg ~/vcpkg
   ~/vcpkg/bootstrap-vcpkg.sh    # or .bat on Windows
   export VCPKG_ROOT=~/vcpkg     # add to shell profile
   ```
3. **mass-proto** checked out next to this repo (or point
   `-DMASS_PROTO_DIR` elsewhere):
   ```bash
   cd ..
   git clone https://github.com/chinese-room-solutions/mass-proto
   ```
4. **llama.cpp submodule**:
   ```bash
   cd mass-worker-llama-cpp
   git submodule update --init --recursive
   ```

### Configure + build

The `Makefile` wraps CMake with backend presets. `make build` auto-selects
the right backend for the host — **Metal on macOS** (the only path on Apple
Silicon), **Vulkan everywhere else** (cross-vendor GPU):

```bash
make build          # host default: Metal on macOS, Vulkan on Linux/Windows
make build-metal    # force Metal  (build/)
make build-vulkan   # force Vulkan (build/)
make build-cuda     # force CUDA   (build-cuda/, NVIDIA-optimized)
```

Or invoke CMake directly (vcpkg toolchain is picked up from `$VCPKG_ROOT`):

```bash
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

### Backend matrix

llama.cpp's compute backend determines which hardware the worker can use.
`make build` picks **Metal on macOS** and **Vulkan elsewhere** — Vulkan is
the cross-platform default because it's the only backend that covers
NVIDIA + AMD + Intel + integrated GPUs in a single binary.

| Backend     | Vendor coverage                       | `make build` default | Build prerequisite                        |
| ----------- | ------------------------------------- | -------------------- | ----------------------------------------- |
| **Vulkan**  | NVIDIA + AMD + Intel + iGPUs (~all)   | **non-macOS**        | [Vulkan SDK](https://vulkan.lunarg.com)   |
| **CUDA**    | NVIDIA only, max performance          | off (opt-in)         | NVIDIA CUDA Toolkit                        |
| **Metal**   | Apple Silicon (macOS only)            | **macOS**            | Xcode                                      |
| **CPU**     | Always available, fallback            | always               | (none)                                     |

Override per build:

```bash
-DMASS_WORKER_LLAMA_VULKAN=OFF    # opt out of Vulkan (CPU-only build)
-DMASS_WORKER_LLAMA_CUDA=ON       # add CUDA backend (NVIDIA-optimized)
-DMASS_WORKER_LLAMA_METAL=ON      # macOS / Apple Silicon
```

**Which backend should you use?**

- **Vulkan** (default off macOS) — recommended for almost everyone on
  Linux/Windows. Works on every modern GPU, runs at 70–90% of the
  vendor-native backend's speed.
- **Metal** (default on macOS) — required on macOS Apple Silicon (no other
  GPU option); `make build` selects it automatically there.
- **CUDA** — turn this on if you have NVIDIA hardware and want maximum
  throughput. Bigger binary, NVIDIA-only.
- **CPU-only** — for environments without GPU drivers (CI, headless boxes).

Backends can coexist in one binary at the cost of larger binary size and
needing every backend's SDK installed at build time. In practice, ship
purpose-built binaries per platform (Vulkan default, CUDA opt-in for
NVIDIA users) rather than one mega-binary.

### GPU utilization reporting

The worker reports each GPU's live load to MASS (the "GPU load" gauge). The
source is per-vendor, and **none of them needs elevated privilege** — no
`setcap`, no root, no group membership:

| Vendor        | Source                                   | Notes |
|---------------|------------------------------------------|-------|
| Intel, AMD    | DRM `drm-engine-*` in `/proc/self/fdinfo` | The worker reads its *own* GPU clients, so it reports its own load (per-client, not whole-GPU). |
| NVIDIA        | NVML (`libnvidia-ml`, runtime-loaded)     | Ships with the driver; never a build dependency. |
| Apple         | IOKit `IOAccelerator` "Device Utilization %" | macOS only. |

If a source is unavailable (e.g. an exotic driver that doesn't populate
`drm-engine-*`), the worker reports `0` rather than failing — the gauge
degrades gracefully. Note: the Linux i915 *perf* interface
(`perf_event_open`) is deliberately **not** used; it requires real root even
with `CAP_PERFMON`, whereas DRM fdinfo gives the same busy-nanosecond
counters with no privilege.

### Test

```bash
make test    # builds, then runs ctest
```

### Package

`make package` builds the terminal install wizard (`mass-worker-setup`) with
the worker and its runtime libraries as payload, wrapped double-clickable —
`.AppImage` on Linux, `.app` on macOS, plain `.exe` on Windows — into `dist/`.
Prebuilt installers ship on the
[releases page](https://github.com/chinese-room-solutions/mass-worker-llama-cpp/releases).

## Install

On macOS and Linux, fetch and run the installer in one line:

```bash
curl -fsSL https://raw.githubusercontent.com/chinese-room-solutions/mass-worker-llama-cpp/main/install.sh | sh
```

This downloads the installer for your platform from the latest release,
verifies its checksum, and opens the setup wizard. On macOS it also avoids
Gatekeeper's "Open Anyway" dance — `curl` sets no quarantine attribute, so the
installer runs straight away, unlike a browser download of the `.app`.

Windows users download `mass-worker-setup_windows_amd64.exe` (or the
double-clickable installer) from the
[releases page](https://github.com/chinese-room-solutions/mass-worker-llama-cpp/releases/latest).

## Run

```bash
./build/bin/mass-worker-llama-cpp \
    --mass-url http://localhost:3455 \
    --token <join-token> \
    --models-dir ./models \
    --name my-machine
```

`--token` is a one-time **join token** from MASS, used only on the worker's
first-ever connect: the worker enrolls, MASS mints a per-worker secret, and the
worker persists its identity (`worker_id` + secret, 0600) under the data dir.
Every later launch reconnects with that stored identity and needs no token — so
`--token` is only for the initial enrollment (the installer normally supplies
it). It is optional: a no-auth MASS enrolls a bare stream with no token, while a
MASS with authentication rejects the enrollment unless the join token is present.
There is no shared operator token.

`--help` lists the remaining flags (`--ca-file` for self-signed MASS TLS,
`--log-level`, `--log-file`, `--vram-headroom-pct`, …).

## License

[FSL-1.1-ALv2](LICENSE.md) — source-available: use, modify, and redistribute
freely for anything except a competing product or service; each release
converts to Apache-2.0 two years after publication.
