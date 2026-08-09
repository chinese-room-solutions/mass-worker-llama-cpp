# Worker library — everything except main(). Tests link against this.
#
# GPU utilization has a per-OS backend (see gpu_util.hpp): the DRM fdinfo
# path is Linux-only, the IOKit path is the Objective-C++ .mm and Apple-only.
# gpu_util.cpp is cross-platform (vendor classify, fdinfo parser, NVML).
if(APPLE)
    set(MASS_WORKER_GPU_UTIL_PLATFORM_SRC gpu_util_darwin.mm)
elseif(UNIX)
    set(MASS_WORKER_GPU_UTIL_PLATFORM_SRC gpu_util_linux.cpp)
else()
    # Windows: no portable per-process engine-busy counter for Intel/AMD
    # (NVIDIA goes through NVML in gpu_util.cpp). This TU supplies the
    # make_drm_or_native_sampler definition gpu_util.cpp references, returning
    # the null (0%) sampler — without it the Windows link fails.
    set(MASS_WORKER_GPU_UTIL_PLATFORM_SRC gpu_util_windows.cpp)
endif()

# Service registration has a per-OS backend like gpu_util: the systemd /
# launchd / SCM code each lives in its own TU; service.cpp holds the shared
# pure parts (exe self-locate, flag forwarding) used by all three + the tests.
if(WIN32)
    set(MASS_WORKER_SERVICE_PLATFORM_SRC service_windows.cpp)
elseif(APPLE)
    set(MASS_WORKER_SERVICE_PLATFORM_SRC service_darwin.cpp)
else()
    set(MASS_WORKER_SERVICE_PLATFORM_SRC service_linux.cpp)
endif()

# The setup form's native folder picker (see pick_folder.hpp) is windowless and
# per-OS: a COM IFileOpenDialog on Windows, an osascript/zenity/kdialog shell-out
# elsewhere (one POSIX TU covers macOS + Linux). No GUI-toolkit link, so the lean
# installer stays self-contained.
if(WIN32)
    set(MASS_WORKER_PICK_FOLDER_SRC pick_folder_windows.cpp)
else()
    set(MASS_WORKER_PICK_FOLDER_SRC pick_folder_posix.cpp)
endif()

# --- Lean setup library -----------------------------------------------------
#
# Everything the installer (mass-worker-setup) needs: the wizard, config /
# credentials persistence, service install/uninstall/staging, and the
# self-extracting payload reader. Deliberately has NO heavy deps — no llama /
# ggml / gRPC / curl — so the installer is a small exe that doesn't depend on
# the very DLLs it installs. The worker lib links this and adds the runtime.
#
# The Windows SCM ServiceMain (service_windows_run.cpp) is NOT here: it drives a
# live Runner, so it belongs to the worker. install_service / uninstall_service
# / staging (service_windows.cpp) are payload-free and live here.
add_library(mass_worker_setup_lib STATIC
    service.cpp
    ${MASS_WORKER_SERVICE_PLATFORM_SRC}
    credentials.cpp
    config.cpp
    wizard.cpp
    form.cpp
    menu.cpp
    ${MASS_WORKER_PICK_FOLDER_SRC}
    term.cpp
    term_input.cpp
    term_screen.cpp
    proc.cpp
    payload.cpp
    # Writer half of the payload format. Its own TU so the linker keeps zstd's
    # compressor out of mass-worker-setup, which only ever decompresses; the
    # packer and the round-trip test are what pull it in.
    payload_pack.cpp
    fsutil.cpp
    version.cpp
    logging.cpp
)
target_include_directories(mass_worker_setup_lib
    PUBLIC
        "${CMAKE_SOURCE_DIR}/include"
        # configure_file writes version.hpp here (top-level CMakeLists).
        "${CMAKE_BINARY_DIR}/generated"
)
target_link_libraries(mass_worker_setup_lib
    PUBLIC
        # Header-only spdlog AND header-only fmt so the standalone installer has
        # NO spdlog/fmt DLL to ship beside it — it must not depend on libraries
        # it hasn't yet installed. vcpkg builds spdlog against external fmt, so
        # spdlog headers pull in fmt; giving them the header-only fmt target
        # inlines fmt too, leaving the installer a single self-contained exe.
        # The worker lib gets both transitively from here, so the whole project
        # uses one (header-only) spdlog/fmt and avoids an ODR split between a DLL
        # and an inlined copy.
        spdlog::spdlog_header_only
        fmt::fmt-header-only
        CLI11::CLI11
        # Static zstd for the appended payload — the one non-header-only dep the
        # installer carries, and it must stay static for the same reason: the
        # installer cannot depend on a library it hasn't installed yet.
        $<IF:$<TARGET_EXISTS:zstd::libzstd_static>,zstd::libzstd_static,zstd::libzstd>
)
# Windows service registration uses the SCM (OpenSCManager / CreateService) and
# token APIs, all in advapi32; the folder picker uses COM (ole32) and the shell's
# IFileOpenDialog (shell32).
if(WIN32)
    target_link_libraries(mass_worker_setup_lib PUBLIC advapi32 ole32 shell32)
endif()
target_compile_features(mass_worker_setup_lib PUBLIC cxx_std_23)
mass_worker_warnings(mass_worker_setup_lib)

# --- Worker library ---------------------------------------------------------
#
# The full runtime: model loading, inference, the gRPC hub stream, hardware
# enumeration — plus the Windows SCM ServiceMain that drives the Runner. Builds
# on the lean setup lib (PUBLIC, so the worker exe sees the wizard/service API).
add_library(mass_worker_lib STATIC
    runner.cpp
    assign_pool.cpp
    worker_service.cpp
    mass_client.cpp
    hardware.cpp
    gpu_util.cpp
    ${MASS_WORKER_GPU_UTIL_PLATFORM_SRC}
    $<$<BOOL:${WIN32}>:service_windows_run.cpp>
    fetch.cpp
    cache.cpp
    bench.cpp
    chat_model.cpp
    ctx_pool.cpp
    embedding_model.cpp
    llama_backend.cpp
)
target_include_directories(mass_worker_lib
    PUBLIC
        "${CMAKE_SOURCE_DIR}/include"
        "${CMAKE_BINARY_DIR}/generated"
)
# llama / ggml are linked here for ggml-backend's device enumeration. The
# llama target is added in the top-level CMakeLists via add_subdirectory of
# third_party/llama.cpp; if that's missing the build fails clearly.
target_link_libraries(mass_worker_lib
    PUBLIC
        mass_worker_setup_lib
        mass_proto
        CURL::libcurl
        OpenSSL::SSL
        OpenSSL::Crypto
        llama
        ggml
        # llama.cpp's high-level helpers (chat templates, reasoning parsing).
        # Target is named `llama-common` (not `common`) by the upstream
        # CMakeLists; see third_party/llama.cpp/common/CMakeLists.txt.
        llama-common
        # libmtmd: vision/audio projector + bitmap loaders + helper that
        # interleaves text decode with image encode. Added via direct
        # add_subdirectory in the top-level CMake (LLAMA_BUILD_TOOLS=OFF).
        mtmd
)
# dlopen lives in libdl on Linux; macOS exposes it from libSystem (no
# explicit link needed). Used in gpu_util.cpp to runtime-load NVML without
# making it a build dependency.
if(UNIX AND NOT APPLE)
    target_link_libraries(mass_worker_lib PUBLIC ${CMAKE_DL_LIBS})
endif()
# macOS GPU utilization reads IOAccelerator via IOKit (gpu_util_darwin.mm);
# the CoreFoundation types come along for the dictionary/number wrappers.
if(APPLE)
    target_link_libraries(mass_worker_lib PUBLIC
        "-framework IOKit"
        "-framework CoreFoundation"
    )
endif()
# Our code is C++20; llama.cpp is C++17 — the project-level CMakeLists
# deliberately doesn't set a global standard so we don't drag llama into
# C++20 (it has u8 literals that change type and stop compiling).
target_compile_features(mass_worker_lib PUBLIC cxx_std_23)
mass_worker_warnings(mass_worker_lib)

# llama.cpp's public headers (common.h, ggml-backend.h, jinja/*.h, …) are
# #included straight into our TUs, so our warnings-as-errors would flag
# upstream warnings we can't fix (MSVC C4305/C4505, GCC -Wshadow on the
# ggml_backend_graph_copy function/struct clash). Mark the vendored include
# dirs SYSTEM on the llama targets' own interface so EVERY consumer — the lib,
# the binaries, and the tests, which inherit these dirs transitively — gets
# them as -isystem, while full warnings stay on our own sources.
foreach(_llama_tgt llama ggml ggml-base llama-common mtmd)
    if(TARGET ${_llama_tgt})
        get_target_property(_inc ${_llama_tgt} INTERFACE_INCLUDE_DIRECTORIES)
        if(_inc)
            set_target_properties(${_llama_tgt} PROPERTIES
                INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${_inc}")
        endif()
    endif()
endforeach()

# Worker binary. With no appended payload it stages its on-disk siblings on
# --install-service (build/download-tree behavior).
add_executable(mass-worker-llama-cpp main.cpp)
target_link_libraries(mass-worker-llama-cpp
    PRIVATE
        mass_worker_lib
)
target_compile_features(mass-worker-llama-cpp PRIVATE cxx_std_23)
mass_worker_warnings(mass-worker-llama-cpp)

# Bundle the MSVC CRT redistributable DLLs (vcruntime140/msvcp140/...) next to
# the worker so the installer ships them and clients need no pre-installed VC++
# Redist — and specifically none newer than the STL our toolset targets. We build
# with the dynamic CRT (/MD), so the exe imports msvcp140.dll etc.; Windows'
# loader prefers a DLL beside the exe over System32, so these win. The list is
# resolved by InstallRequiredSystemLibraries in the top-level CMakeLists for the
# active toolset (no hardcoded path/version). The Makefile's packaging glob then
# picks up every .dll in this dir, so the CRT rides along automatically.
if(WIN32 AND CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS)
    add_custom_command(TARGET mass-worker-llama-cpp POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS}
                "$<TARGET_FILE_DIR:mass-worker-llama-cpp>"
        COMMENT "Bundling MSVC CRT redistributable DLLs next to the worker"
        VERBATIM)
endif()

# Cross-platform library resolution for the STAGED worker (Linux/macOS only).
# On Windows a DLL beside the exe is found automatically; on Unix the binary
# searches only its RPATH. stage_install() lays the bundled .so/.dylib files out
# FLAT beside the worker (copy_siblings / payload extract → dst_dir / filename),
# so the runtime search path is the binary's OWN directory — $ORIGIN on Linux,
# @loader_path on macOS. BUILD_WITH_INSTALL_RPATH bakes exactly that (and only
# that) into the binary at build time, so a copy staged outside the build tree
# is self-contained rather than failing with "libcurl.so not found". The ggml/
# llama DLLs already sit beside it in the build tree too, so this is also correct
# for running straight from build/bin. Scoped to this target so it doesn't
# perturb the vendored llama.cpp targets or the test binaries.
if(UNIX)
    if(APPLE)
        set(_mass_worker_rpath "@loader_path")
    else()
        set(_mass_worker_rpath "$ORIGIN")
    endif()
    set_target_properties(mass-worker-llama-cpp PROPERTIES
        INSTALL_RPATH "${_mass_worker_rpath}"
        BUILD_WITH_INSTALL_RPATH ON
    )
endif()

# --- Self-contained installer: mass-worker-setup ----------------------------
#
# A SEPARATE binary from the worker, with its own entry point (setup_main.cpp)
# that only configures + installs/uninstalls — it never runs the worker, so it
# links the lean setup lib and has NO llama/ggml/gRPC dependency (it must not
# depend on the DLLs it installs). On its own it is a small console exe; `make
# package` appends the worker + its runtime libraries to it (a self-extracting
# archive). At install time stage_install() detects the appended payload,
# extracts it into the install dir, and registers the staged worker
# (mass-worker-llama-cpp[.exe]) — the installer keeps its own distinct name.
add_executable(mass-worker-setup setup_main.cpp)
target_link_libraries(mass-worker-setup PRIVATE mass_worker_setup_lib)
target_compile_features(mass-worker-setup PRIVATE cxx_std_23)
mass_worker_warnings(mass-worker-setup)

# Build-time packer that turns mass-worker-setup + worker + libs into the
# self-extracting installer (see tools/pack.cpp). `make package` runs it; it is a
# thin CLI over append_payload() in the setup lib, which is also where the
# matching reader lives — one definition of the format, no drift.
add_executable(mass-worker-pack "${CMAKE_SOURCE_DIR}/tools/pack.cpp")
target_link_libraries(mass-worker-pack PRIVATE mass_worker_setup_lib)
target_compile_features(mass-worker-pack PRIVATE cxx_std_23)
mass_worker_warnings(mass-worker-pack)
