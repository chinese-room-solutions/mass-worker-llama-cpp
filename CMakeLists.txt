cmake_minimum_required(VERSION 3.24)

# Static-linking the vcpkg dependencies (fmt/curl/openssl/protobuf/gRPC/abseil/…)
# into our binaries — so the worker ships as one exe plus only the llama/ggml
# DLLs, not ~20 support DLLs — is selected by the vcpkg triplet, which must be
# passed on the configure command line (the Makefile's build targets do this):
#   -DVCPKG_TARGET_TRIPLET=x64-windows-static-md-rel
#   -DVCPKG_OVERLAY_TRIPLETS=<repo>/triplets
# The "-md" keeps the dynamic MSVC runtime (matching llama.cpp); the "-rel"
# overlay (triplets/) builds Release-only deps (skips the unused Debug half).
# llama/ggml build via add_subdirectory (not vcpkg) so they stay DLLs regardless.
# Setting the triplet here (before the toolchain) is unreliable — vcpkg's
# manifest install doesn't pick up a CMakeLists-set overlay triplet — so it's a
# command-line concern, not baked into this file.

# Toolchain file (vcpkg) must be set before project(). Resolved from env var
# VCPKG_ROOT or the canonical $HOME/.vcpkg / C:\vcpkg locations. Override on
# the command line: -DCMAKE_TOOLCHAIN_FILE=<path>.
if(NOT DEFINED CMAKE_TOOLCHAIN_FILE AND DEFINED ENV{VCPKG_ROOT})
    set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
        CACHE STRING "vcpkg toolchain file")
endif()

project(mass-worker-llama-cpp
    VERSION 0.3.1
    DESCRIPTION "MASS worker — llama.cpp runtime, native C++"
    LANGUAGES CXX C
)

# Note: do *not* set a project-wide CMAKE_CXX_STANDARD. llama.cpp targets
# C++17 internally and uses u8"..." literals that change type to char8_t in
# C++20+ (and stop converting to const char* implicitly). Each of our own
# targets requests cxx_std_23 via target_compile_features so the standard
# is scoped to mass_worker_lib + tests only — we need C++23 for std::expected.
set(CMAKE_CXX_EXTENSIONS OFF)

# RPATH for the staged worker — scoped to the mass-worker-llama-cpp target in
# src/CMakeLists.txt (mass_worker_xplat_rpath), NOT globally, so the vendored
# llama.cpp/ggml targets and the test binaries keep their normal build-tree
# RPATH. See the comment there for the why ($ORIGIN/@loader_path because
# stage_install lays libraries out flat beside the binary).

# Speed up rebuilds (notably the vendored llama.cpp/ggml tree) by routing the
# compiler through a cache when one is available — ccache on Linux/macOS,
# sccache (MSVC-compatible) on Windows. A no-op when neither is installed, so
# local builds without a cache are unaffected. CI primes the cache across runs.
if(NOT DEFINED CMAKE_CXX_COMPILER_LAUNCHER)
    find_program(MASS_COMPILER_CACHE NAMES ccache sccache)
    if(MASS_COMPILER_CACHE)
        set(CMAKE_C_COMPILER_LAUNCHER   "${MASS_COMPILER_CACHE}")
        set(CMAKE_CXX_COMPILER_LAUNCHER "${MASS_COMPILER_CACHE}")
        message(STATUS "Compiler cache: ${MASS_COMPILER_CACHE}")
        # sccache + MSVC can't cache when debug info goes to a separate .pdb
        # (/Zi); embed it instead (/Z7) so Windows builds stay cacheable.
        # CMAKE_MSVC_DEBUG_INFORMATION_FORMAT + CMP0141 need CMake >= 3.25.
        if(MSVC AND MASS_COMPILER_CACHE MATCHES "sccache"
           AND CMAKE_VERSION VERSION_GREATER_EQUAL "3.25")
            cmake_policy(SET CMP0141 NEW)
            set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "Embedded")
        endif()
    endif()
endif()

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

# Ship the MSVC C runtime DLLs (vcruntime140/msvcp140/...) app-local, next to
# the worker + installer, so clients don't need a pre-installed VC++ Redist —
# and specifically don't need one NEW ENOUGH for the STL our toolset targets.
# We build with the dynamic CRT (/MD, via the x64-windows-static-md triplet), so
# the exes import msvcp140.dll etc.; a client with an older/absent redist fails
# to launch (e.g. unresolved __std_find_first_of_trivial_pos_1 from a newer STL).
# Windows' loader prefers a DLL sitting beside the exe over the System32 copy, so
# bundling the toolset's own redist DLLs guarantees the exact CRT we tested with.
# InstallRequiredSystemLibraries resolves the correct DLLs for the ACTIVE toolset
# (no hardcoded version/path), and the packaging glob in the Makefile picks up
# whatever .dll lands in the output dir — so no change needed there.
if(WIN32)
    set(CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS_SKIP ON)  # resolve the list; don't auto-install
    include(InstallRequiredSystemLibraries)
    # The actual copy target is created after add_subdirectory(src) below, once
    # the mass-worker-llama-cpp target exists (we copy into its output dir).
endif()

# Warnings — keep tight, treat as a real signal not noise. -Werror gated
# behind option so CI can flip it without forcing it on local dev.
#
# These flags are collected into MASS_WORKER_WARNING_FLAGS and applied
# per-target (see mass_worker_warnings() below) to OUR code only — NOT via a
# global add_compile_options, which would leak onto the vendored llama.cpp /
# ggml subtree and fail the build on upstream warnings we can't fix.
option(MASS_WORKER_WERROR "Treat warnings as errors" OFF)

if(MSVC)
    # Restrict to CXX so we don't leak MSVC-only flags onto nvcc's command
    # line. nvcc parses `/W4` as a path argument and fails the build with
    # "A single input file is required for a non-link phase".
    set(MASS_WORKER_WARNING_FLAGS
        $<$<COMPILE_LANGUAGE:CXX>:/W4>
        $<$<COMPILE_LANGUAGE:CXX>:/permissive->
        $<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor>)
    if(MASS_WORKER_WERROR)
        list(APPEND MASS_WORKER_WARNING_FLAGS $<$<COMPILE_LANGUAGE:CXX>:/WX>)
    endif()
else()
    set(MASS_WORKER_WARNING_FLAGS
        $<$<COMPILE_LANGUAGE:CXX>:-Wall>
        $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
        $<$<COMPILE_LANGUAGE:CXX>:-Wpedantic>
        $<$<COMPILE_LANGUAGE:CXX>:-Wshadow>
        $<$<COMPILE_LANGUAGE:CXX>:-Wconversion>
        $<$<COMPILE_LANGUAGE:CXX>:-Wsign-conversion>
        # -Wextra enables missing-field-initializers, which on GCC fires on
        # partial designated initializers — idiomatic here, and the language
        # value-initializes the omitted members. Clang doesn't warn on them.
        $<$<COMPILE_LANGUAGE:CXX>:-Wno-missing-field-initializers>)
    if(MASS_WORKER_WERROR)
        list(APPEND MASS_WORKER_WARNING_FLAGS $<$<COMPILE_LANGUAGE:CXX>:-Werror>)
    endif()
endif()

# Apply the project warning flags to one of our own targets. Call this for
# every target we author; never for vendored code.
function(mass_worker_warnings target)
    target_compile_options(${target} PRIVATE ${MASS_WORKER_WARNING_FLAGS})
endfunction()

# --- Dependencies (via vcpkg manifest) ---
find_package(Protobuf CONFIG REQUIRED)
find_package(gRPC CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)
find_package(CLI11 CONFIG REQUIRED)
find_package(CURL CONFIG REQUIRED)
find_package(OpenSSL REQUIRED)
# zstd compresses the self-extracting installer's appended payload (see
# mass_worker/payload.hpp). Statically linked, so the installer stays a single
# self-contained exe with no library to find before it has installed one.
find_package(zstd CONFIG REQUIRED)

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
# The worker speaks only worker.proto (the hub stream) and the llama-cpp
# payload; it never uses the public service.proto (mass.v1.Mass), which now
# imports gateway/gateway.proto and would drag that in for no reason.
mass_generate_proto(
    PROTO_DIR "${MASS_PROTO_DIR}/proto"
    PROTOS    worker/worker.proto llama-cpp/payload.proto
    OUT_DIR   "${CMAKE_BINARY_DIR}/proto"
    OUT_VAR   MASS_PROTO_SOURCES
)

add_library(mass_proto STATIC ${MASS_PROTO_SOURCES})
# SYSTEM: protoc's generated headers trip -Wconversion under GCC (size() ->
# int narrowing in map accessors); consumers must not inherit those warnings.
target_include_directories(mass_proto SYSTEM PUBLIC "${CMAKE_BINARY_DIR}/proto")
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

# --- Generated version header ---
# Bake the version, a `git describe` string, and the compiled GPU backend into
# a header for the `--version` flag. Computed at configure time: re-run cmake
# (or let the configure step re-trigger) to refresh the git string after a
# commit. A bare `cpu` build (all GPU options OFF) reports "cpu".
if(MASS_WORKER_LLAMA_CUDA)
    set(MASS_WORKER_GPU_BACKEND "cuda")
elseif(MASS_WORKER_LLAMA_METAL)
    set(MASS_WORKER_GPU_BACKEND "metal")
elseif(MASS_WORKER_LLAMA_VULKAN)
    set(MASS_WORKER_GPU_BACKEND "vulkan")
else()
    set(MASS_WORKER_GPU_BACKEND "cpu")
endif()

# Per-backend flags for the version header — a binary can have several. Rendered
# as C++ bool literals into version.hpp (kHasCuda/kHasMetal/kHasVulkan).
foreach(backend CUDA METAL VULKAN)
    if(MASS_WORKER_LLAMA_${backend})
        set(MASS_WORKER_HAS_${backend} "true")
    else()
        set(MASS_WORKER_HAS_${backend} "false")
    endif()
endforeach()

set(MASS_WORKER_GIT_DESCRIBE "")
find_package(Git QUIET)
if(Git_FOUND AND EXISTS "${CMAKE_SOURCE_DIR}/.git")
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" describe --tags --dirty --always
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
        OUTPUT_VARIABLE MASS_WORKER_GIT_DESCRIBE
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
endif()

configure_file(
    "${CMAKE_SOURCE_DIR}/include/mass_worker/version.hpp.in"
    "${CMAKE_BINARY_DIR}/generated/mass_worker/version.hpp"
    @ONLY
)

# --- Windows header hygiene for our own code ---
# windows.h defines min/max as function-like macros, which breaks any TU that
# also uses std::min/std::max — `std::min(a, b)` expands to `std::(a) < (b)…`,
# an "illegal token on right side of ::". Several of our TUs pull in Windows
# headers (directly or via <winsock2.h>) and were relying on ad hoc per-file
# `#define NOMINMAX`, which only covered a handful of them.
#
# Set at directory scope rather than with a global add_compile_definitions()
# because the vendored llama.cpp tree above defines WIN32_LEAN_AND_MEAN
# *unguarded* in ~21 TUs — a project-wide definition would collide with every
# one of them (C4005 macro redefinition). Placing it here, after
# add_subdirectory(third_party/llama.cpp), scopes it to our sources only.
# llama.cpp's own NOMINMAX defines are all #ifndef-guarded, so this stays
# compatible if the vendored tree is ever moved below.
if(WIN32)
    add_compile_definitions(NOMINMAX WIN32_LEAN_AND_MEAN)
endif()

# --- Core worker library + binary ---
# CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS (the resolved CRT redist DLLs) is set above
# and consumed inside src/CMakeLists.txt, which owns the mass-worker-llama-cpp target
# and attaches the POST_BUILD copy that bundles those DLLs next to it.
add_subdirectory(src)

# --- Tests ---
option(MASS_WORKER_BUILD_TESTS "Build unit tests" ON)
if(MASS_WORKER_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# --- Install ---
install(TARGETS mass-worker-llama-cpp
    RUNTIME DESTINATION bin
)
