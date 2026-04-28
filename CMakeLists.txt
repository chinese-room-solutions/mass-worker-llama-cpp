cmake_minimum_required(VERSION 3.24)

# Toolchain file (vcpkg) must be set before project(). Resolved from env var
# VCPKG_ROOT or the canonical $HOME/.vcpkg / C:\vcpkg locations. Override on
# the command line: -DCMAKE_TOOLCHAIN_FILE=<path>.
if(NOT DEFINED CMAKE_TOOLCHAIN_FILE AND DEFINED ENV{VCPKG_ROOT})
    set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
        CACHE STRING "vcpkg toolchain file")
endif()

project(mass-worker-llama-cpp
    VERSION 0.1.0
    DESCRIPTION "MASS worker — llama.cpp runtime, native C++"
    LANGUAGES CXX C
)

# Note: do *not* set a project-wide CMAKE_CXX_STANDARD. llama.cpp targets
# C++17 internally and uses u8"..." literals that change type to char8_t in
# C++20+ (and stop converting to const char* implicitly). Each of our own
# targets requests cxx_std_23 via target_compile_features so the standard
# is scoped to mass_worker_lib + tests only — we need C++23 for std::expected.
set(CMAKE_CXX_EXTENSIONS OFF)

# Compile commands DB for clangd / IDE tooling.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Co-locate all binaries (.exe + .dll) under build/bin so Windows DLL
# resolution finds llama.dll / ggml*.dll next to every .exe — including
# test binaries — without having to mess with PATH. The Visual Studio
# multi-config generator appends /<Config>/ automatically.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# Have vcpkg copy every transitive dependency DLL next to our binaries on
# Windows. Without this, gRPC / abseil / protobuf / etc. fail to load with
# STATUS_DLL_NOT_FOUND (0xc0000135) at test/smoke run time.
if(WIN32)
    set(VCPKG_APPLOCAL_DEPS ON)
    set(X_VCPKG_APPLOCAL_DEPS_INSTALL ON)
endif()

# Warnings — keep tight, treat as a real signal not noise. -Werror gated
# behind option so CI can flip it without forcing it on local dev.
option(MASS_WORKER_WERROR "Treat warnings as errors" OFF)

if(MSVC)
    add_compile_options(/W4 /permissive- /Zc:preprocessor)
    if(MASS_WORKER_WERROR)
        add_compile_options(/WX)
    endif()
else()
    add_compile_options(-Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wsign-conversion)
    if(MASS_WORKER_WERROR)
        add_compile_options(-Werror)
    endif()
endif()

# --- Dependencies (via vcpkg manifest) ---
find_package(Protobuf CONFIG REQUIRED)
find_package(gRPC CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)
find_package(CLI11 CONFIG REQUIRED)
find_package(CURL CONFIG REQUIRED)
find_package(OpenSSL REQUIRED)

# --- Generated proto sources from sibling mass-proto/ repo ---
# Sources live in ../mass-proto/proto/. Generated .pb.h/.cc + grpc.pb.h/.cc
# land in ${CMAKE_BINARY_DIR}/proto/ so they don't pollute the source tree.
# Override MASS_PROTO_DIR if mass-proto is checked out somewhere else.
set(MASS_PROTO_DIR "${CMAKE_SOURCE_DIR}/../mass-proto" CACHE PATH
    "Path to the mass-proto repo (contains proto/service.proto)")

if(NOT EXISTS "${MASS_PROTO_DIR}/proto/service.proto")
    message(FATAL_ERROR
        "mass-proto sources not found at ${MASS_PROTO_DIR}/proto/. "
        "Clone https://github.com/chinese-room-solutions/mass-proto "
        "next to this repo, or pass -DMASS_PROTO_DIR=<path>.")
endif()

include(cmake/Proto.cmake)
mass_generate_proto(
    PROTO_DIR "${MASS_PROTO_DIR}/proto"
    PROTOS    service.proto worker/worker.proto
    OUT_DIR   "${CMAKE_BINARY_DIR}/proto"
    OUT_VAR   MASS_PROTO_SOURCES
)

add_library(mass_proto STATIC ${MASS_PROTO_SOURCES})
target_include_directories(mass_proto PUBLIC "${CMAKE_BINARY_DIR}/proto")
target_link_libraries(mass_proto
    PUBLIC
        protobuf::libprotobuf
        gRPC::grpc++
)
# Generated proto headers leak into every consumer; suppress the noisy
# warnings they trip (size_t→int conversions etc). Don't apply project
# warning flags to generated code we can't fix.
if(MSVC)
    target_compile_options(mass_proto PRIVATE /W0)
    target_compile_options(mass_proto PUBLIC  /wd4267 /wd4244 /wd4251)
else()
    target_compile_options(mass_proto PRIVATE -w)
endif()

# --- llama.cpp (vendored as submodule under third_party/llama.cpp) ---
# Add via add_subdirectory so we control build flags. GPU backends are opt-in.
# GPU backend selection — see README "Backend matrix" for the full story.
# Vulkan is the default because it covers NVIDIA + AMD + Intel + iGPUs in
# one binary at ~70-90% of native-backend performance. CUDA is opt-in for
# users who want maximum NVIDIA throughput; Metal is the only path on
# Apple Silicon. Backends can be combined but binary size grows with each.
option(MASS_WORKER_LLAMA_VULKAN "Build llama.cpp with Vulkan support (cross-vendor GPU)" ON)
option(MASS_WORKER_LLAMA_CUDA   "Build llama.cpp with CUDA support (NVIDIA only)"        OFF)
option(MASS_WORKER_LLAMA_METAL  "Build llama.cpp with Metal support (macOS only)"        OFF)

set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER   OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS    OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL           OFF CACHE BOOL "" FORCE)
# We need the `common` library for chat-template + reasoning-content parsing
# (common_chat_templates_init / common_chat_parse). It's off by default when
# llama.cpp is consumed via add_subdirectory; turn it on so our targets can
# link against the `common` target.
set(LLAMA_BUILD_COMMON   ON  CACHE BOOL "" FORCE)
set(GGML_CUDA ${MASS_WORKER_LLAMA_CUDA} CACHE BOOL "" FORCE)
set(GGML_METAL ${MASS_WORKER_LLAMA_METAL} CACHE BOOL "" FORCE)
set(GGML_VULKAN ${MASS_WORKER_LLAMA_VULKAN} CACHE BOOL "" FORCE)

if(EXISTS "${CMAKE_SOURCE_DIR}/third_party/llama.cpp/CMakeLists.txt")
    add_subdirectory(third_party/llama.cpp EXCLUDE_FROM_ALL)
    # mtmd lives under tools/ and is normally gated by LLAMA_BUILD_TOOLS.
    # We want the library without dragging in the rest of the tools (cli,
    # server, etc.) — so add the mtmd subdir directly. Two vars normally
    # set inside llama.cpp's own scope have to be promoted here because
    # they aren't PARENT_SCOPE-exported by the upstream CMakeLists.
    if(EXISTS "${CMAKE_SOURCE_DIR}/third_party/llama.cpp/tools/mtmd/CMakeLists.txt")
        set(LLAMA_INSTALL_VERSION 0.0.0 CACHE INTERNAL "")
        set(LLAMA_TOOLS_INSTALL OFF CACHE INTERNAL "")
        add_subdirectory(third_party/llama.cpp/tools/mtmd EXCLUDE_FROM_ALL)
    else()
        message(FATAL_ERROR "third_party/llama.cpp/tools/mtmd not found")
    endif()
else()
    message(WARNING
        "third_party/llama.cpp not found. Run:\n"
        "    git submodule add https://github.com/ggerganov/llama.cpp third_party/llama.cpp\n"
        "    git submodule update --init --recursive\n"
        "Build will fail when targets that depend on llama are added.")
endif()

# --- Core worker library + binary ---
add_subdirectory(src)

# --- Tests ---
option(MASS_WORKER_BUILD_TESTS "Build unit tests" ON)
if(MASS_WORKER_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# --- Install ---
install(TARGETS mass-worker-llama
    RUNTIME DESTINATION bin
)
