# Worker library — everything except main(). Tests link against this.
add_library(mass_worker_lib STATIC
    runner.cpp
    worker_service.cpp
    mass_client.cpp
    hardware.cpp
    fetch.cpp
    cache.cpp
    bench.cpp
    chat_model.cpp
    embedding_model.cpp
    llama_backend.cpp
    logging.cpp
)
target_include_directories(mass_worker_lib
    PUBLIC
        "${CMAKE_SOURCE_DIR}/include"
)
# llama / ggml are linked here for ggml-backend's device enumeration. The
# llama target is added in the top-level CMakeLists via add_subdirectory of
# third_party/llama.cpp; if that's missing the build fails clearly.
target_link_libraries(mass_worker_lib
    PUBLIC
        mass_proto
        spdlog::spdlog
        CLI11::CLI11
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
# explicit link needed). Used in hardware.cpp to runtime-load NVML without
# making it a build dependency.
if(UNIX AND NOT APPLE)
    target_link_libraries(mass_worker_lib PUBLIC ${CMAKE_DL_LIBS})
endif()
# Our code is C++20; llama.cpp is C++17 — the project-level CMakeLists
# deliberately doesn't set a global standard so we don't drag llama into
# C++20 (it has u8 literals that change type and stop compiling).
target_compile_features(mass_worker_lib PUBLIC cxx_std_23)

# Worker binary.
add_executable(mass-worker-llama main.cpp)
target_link_libraries(mass-worker-llama
    PRIVATE
        mass_worker_lib
)
target_compile_features(mass-worker-llama PRIVATE cxx_std_23)
