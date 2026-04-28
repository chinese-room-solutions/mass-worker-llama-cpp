# mass-worker-llama-cpp

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
3. **mass-proto** checked out next to this repo:
   ```bash
   cd ..
   git clone https://github.com/chinese-room-solutions/mass-proto
   ```
4. **llama.cpp submodule** (one-time):
   ```bash
   cd mass-worker-llama-cpp
   git submodule add https://github.com/ggerganov/llama.cpp third_party/llama.cpp
   git submodule update --init --recursive
   ```

### Configure + build

```bash
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

### Backend matrix

llama.cpp's compute backend determines which hardware the worker can use.
We **default to Vulkan** because it's the only backend that covers
NVIDIA + AMD + Intel + integrated GPUs in a single binary.

| Backend     | Vendor coverage                       | Default | Build prerequisite                        |
| ----------- | ------------------------------------- | ------- | ----------------------------------------- |
| **Vulkan**  | NVIDIA + AMD + Intel + iGPUs (~all)   | **ON**  | [Vulkan SDK](https://vulkan.lunarg.com)   |
| **CUDA**    | NVIDIA only, max performance          | off     | NVIDIA CUDA Toolkit                        |
| **Metal**   | Apple Silicon (macOS only)            | off     | Xcode                                      |
| **CPU**     | Always available, fallback            | always  | (none)                                     |

Override per build:

```bash
-DMASS_WORKER_LLAMA_VULKAN=OFF    # opt out of Vulkan (CPU-only build)
-DMASS_WORKER_LLAMA_CUDA=ON       # add CUDA backend (NVIDIA-optimized)
-DMASS_WORKER_LLAMA_METAL=ON      # macOS / Apple Silicon
```

**Which backend should you use?**

- **Default (Vulkan)** — recommended for almost everyone. Works on every
  modern GPU, runs at 70–90% of the vendor-native backend's speed.
- **CUDA** — turn this on if you have NVIDIA hardware and want maximum
  throughput. Bigger binary, NVIDIA-only.
- **Metal** — required on macOS Apple Silicon (no other GPU option).
- **CPU-only** — for environments without GPU drivers (CI, headless boxes).

Backends can coexist in one binary at the cost of larger binary size and
needing every backend's SDK installed at build time. In practice, ship
purpose-built binaries per platform (Vulkan default, CUDA opt-in for
NVIDIA users) rather than one mega-binary.

### Test

```bash
ctest --test-dir build --output-on-failure -j
```

## Run

```bash
./build/src/mass-worker-llama \
    --mass-url http://localhost:3455 \
    --token <auth-token> \
    --models-dir ./models \
    --name my-machine
```
