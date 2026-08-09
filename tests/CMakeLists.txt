find_package(GTest CONFIG REQUIRED)

# Smoke test — proves the lib links and gtest harness works.
add_executable(mass_worker_smoke_test
    smoke_test.cpp
)
target_link_libraries(mass_worker_smoke_test
    PRIVATE
        mass_worker_lib
        GTest::gtest
        GTest::gtest_main
)
target_compile_features(mass_worker_smoke_test PRIVATE cxx_std_23)
mass_worker_warnings(mass_worker_smoke_test)
add_test(NAME smoke COMMAND mass_worker_smoke_test)

# Runner tests: spin up an in-process fake WorkerHub server on a free port,
# run the real Runner against it, and assert on observed messages.
add_executable(mass_worker_runner_test
    runner_test.cpp
)
target_link_libraries(mass_worker_runner_test
    PRIVATE
        mass_worker_lib
        GTest::gtest
        GTest::gtest_main
)
target_compile_features(mass_worker_runner_test PRIVATE cxx_std_23)
mass_worker_warnings(mass_worker_runner_test)
add_test(NAME runner COMMAND mass_worker_runner_test)

# AssignPool tests — the runner's grow-on-demand dispatch pool. Blocking
# fake tasks + rendezvous gates prove growth under backlog, the hard cap,
# idle reuse, and drain-on-shutdown deterministically (every block carries
# a deadline; no wall-clock assertions).
add_executable(mass_worker_assign_pool_test
    assign_pool_test.cpp
)
target_link_libraries(mass_worker_assign_pool_test
    PRIVATE
        mass_worker_lib
        GTest::gtest
        GTest::gtest_main
)
target_compile_features(mass_worker_assign_pool_test PRIVATE cxx_std_23)
mass_worker_warnings(mass_worker_assign_pool_test)
add_test(NAME assign_pool COMMAND mass_worker_assign_pool_test)

# MassClient tests — the pure seams: the URL→dial-target parse (scheme/TLS
# selection, path/userinfo stripping) and the enrollment-failure diagnosis
# (which gRPC status earns a join-token hypothesis, which is fatal vs. worth
# retrying). No network, no channel is dialed.
add_executable(mass_worker_mass_client_test
    mass_client_test.cpp
)
target_link_libraries(mass_worker_mass_client_test
    PRIVATE
        mass_worker_lib
        GTest::gtest
        GTest::gtest_main
)
target_compile_features(mass_worker_mass_client_test PRIVATE cxx_std_23)
mass_worker_warnings(mass_worker_mass_client_test)
add_test(NAME mass_client COMMAND mass_worker_mass_client_test)

# Hardware tests — verify CPU/GPU enumeration via ggml-backend.
add_executable(mass_worker_hardware_test
    hardware_test.cpp
)
target_link_libraries(mass_worker_hardware_test
    PRIVATE
        mass_worker_lib
        GTest::gtest
        GTest::gtest_main
)
target_compile_features(mass_worker_hardware_test PRIVATE cxx_std_23)
mass_worker_warnings(mass_worker_hardware_test)
add_test(NAME hardware COMMAND mass_worker_hardware_test)

# GPU-util tests — pure parsing/conversion/classification (no /proc, no
# syscalls), so they run deterministically in CI on any host.
add_executable(mass_worker_gpu_util_test
    gpu_util_test.cpp
)
target_link_libraries(mass_worker_gpu_util_test
    PRIVATE
        mass_worker_lib
        GTest::gtest
        GTest::gtest_main
)
target_compile_features(mass_worker_gpu_util_test PRIVATE cxx_std_23)
mass_worker_warnings(mass_worker_gpu_util_test)
add_test(NAME gpu_util COMMAND mass_worker_gpu_util_test)

# Bench tests — the error seam only (unknown device → typed error, never a
# zero measurement). The real micro-benches are too heavy for CI.
add_executable(mass_worker_bench_test
    bench_test.cpp
)
target_link_libraries(mass_worker_bench_test
    PRIVATE
        mass_worker_lib
        GTest::gtest
        GTest::gtest_main
)
target_compile_features(mass_worker_bench_test PRIVATE cxx_std_23)
mass_worker_warnings(mass_worker_bench_test)
add_test(NAME bench COMMAND mass_worker_bench_test)

# Chat-model tests — the pure sampler-chain decision table (plan_sampler).
# No model load, no GPU; deterministic in CI.
add_executable(mass_worker_chat_model_test
    chat_model_test.cpp
)
target_link_libraries(mass_worker_chat_model_test
    PRIVATE
        mass_worker_lib
        GTest::gtest
        GTest::gtest_main
)
target_compile_features(mass_worker_chat_model_test PRIVATE cxx_std_23)
mass_worker_warnings(mass_worker_chat_model_test)
add_test(NAME chat_model COMMAND mass_worker_chat_model_test)

# Ctx-pool tests — the pure headroom decision table (CtxPoolHeadroom)
# behind the grow loop chat + embed pools share. No model load, no GPU;
# deterministic in CI.
add_executable(mass_worker_ctx_pool_test
    ctx_pool_test.cpp
)
target_link_libraries(mass_worker_ctx_pool_test
    PRIVATE
        mass_worker_lib
        GTest::gtest
        GTest::gtest_main
)
target_compile_features(mass_worker_ctx_pool_test PRIVATE cxx_std_23)
mass_worker_warnings(mass_worker_ctx_pool_test)
add_test(NAME ctx_pool COMMAND mass_worker_ctx_pool_test)

# Batch-runner tests — the concurrency seam batch jobs fan out through.
# Fake work functions + rendezvous gates prove observed parallelism, index
# alignment, failure propagation, and cancellation deterministically (no
# wall-clock assertions). Pure std, so it links the lean setup lib.
add_executable(mass_worker_batch_runner_test
    batch_runner_test.cpp
)
target_link_libraries(mass_worker_batch_runner_test
    PRIVATE
        mass_worker_setup_lib
        GTest::gtest
        GTest::gtest_main
)
target_compile_features(mass_worker_batch_runner_test PRIVATE cxx_std_23)
mass_worker_warnings(mass_worker_batch_runner_test)
add_test(NAME batch_runner COMMAND mass_worker_batch_runner_test)

# Worker-service tests — the pure device-placement decision table
# (enabled_placement_ids) and the register-frame contract. No model load,
# no network; deterministic in CI.
add_executable(mass_worker_worker_service_test
    worker_service_test.cpp
)
target_link_libraries(mass_worker_worker_service_test
    PRIVATE
        mass_worker_lib
        GTest::gtest
        GTest::gtest_main
)
target_compile_features(mass_worker_worker_service_test PRIVATE cxx_std_23)
mass_worker_warnings(mass_worker_worker_service_test)
add_test(NAME worker_service COMMAND mass_worker_worker_service_test)

# Fetch + cache tests — pure logic + filesystem; no network in CI.
add_executable(mass_worker_fetch_test
    fetch_test.cpp
)
target_link_libraries(mass_worker_fetch_test
    PRIVATE
        mass_worker_lib
        GTest::gtest
        GTest::gtest_main
)
target_compile_features(mass_worker_fetch_test PRIVATE cxx_std_23)
mass_worker_warnings(mass_worker_fetch_test)
add_test(NAME fetch COMMAND mass_worker_fetch_test)

# Service tests — pure flag-forwarding + the host OS's unit/plist/binPath
# rendering. No privileged side effects (no actual install), so they run in CI.
add_executable(mass_worker_service_test
    service_test.cpp
)
target_link_libraries(mass_worker_service_test
    PRIVATE
        mass_worker_lib
        GTest::gtest
        GTest::gtest_main
)
target_compile_features(mass_worker_service_test PRIVATE cxx_std_23)
mass_worker_warnings(mass_worker_service_test)
add_test(NAME service COMMAND mass_worker_service_test)

# Config tests — config.conf round-trip, partial/missing/corrupt handling.
# Pure filesystem, no network.
add_executable(mass_worker_config_test
    config_test.cpp
)
target_link_libraries(mass_worker_config_test
    PRIVATE
        mass_worker_lib
        GTest::gtest
        GTest::gtest_main
)
target_compile_features(mass_worker_config_test PRIVATE cxx_std_23)
mass_worker_warnings(mass_worker_config_test)
add_test(NAME config COMMAND mass_worker_config_test)

# Credentials tests — connection-credentials file round-trip + missing/partial
# handling. Pure filesystem, no network.
add_executable(mass_worker_credentials_test
    credentials_test.cpp
)
target_link_libraries(mass_worker_credentials_test
    PRIVATE
        mass_worker_lib
        GTest::gtest
        GTest::gtest_main
)
target_compile_features(mass_worker_credentials_test PRIVATE cxx_std_23)
mass_worker_warnings(mass_worker_credentials_test)
add_test(NAME credentials COMMAND mass_worker_credentials_test)

# Terminal-input parser tests — the pure byte→Key seam (escape sequences, the
# lone-ESC disambiguation). No tty, so deterministic in CI. Links the lean setup
# lib directly: the parser has no llama/ggml deps.
add_executable(mass_worker_term_input_test
    term_input_test.cpp
)
target_link_libraries(mass_worker_term_input_test
    PRIVATE
        mass_worker_setup_lib
        GTest::gtest
        GTest::gtest_main
)
target_compile_features(mass_worker_term_input_test PRIVATE cxx_std_23)
mass_worker_warnings(mass_worker_term_input_test)
add_test(NAME term_input COMMAND mass_worker_term_input_test)

# Terminal helpers — word-wrap (word/hard-break/overlong-word/boundary). Pure.
add_executable(mass_worker_term_test
    term_test.cpp
)
target_link_libraries(mass_worker_term_test
    PRIVATE
        mass_worker_setup_lib
        GTest::gtest
        GTest::gtest_main
)
target_compile_features(mass_worker_term_test PRIVATE cxx_std_23)
mass_worker_warnings(mass_worker_term_test)
add_test(NAME term COMMAND mass_worker_term_test)

# Terminal-session tests — the guarantee that the alternate screen, the hidden
# cursor and the OSC 11 background are undone on EVERY exit path. The signal
# cases fork a child with stdout on a pipe, so the assertions are on the bytes a
# signal handler actually wrote (POSIX only; Windows has no fork).
add_executable(mass_worker_term_screen_test
    term_screen_test.cpp
)
target_link_libraries(mass_worker_term_screen_test
    PRIVATE
        mass_worker_setup_lib
        GTest::gtest
        GTest::gtest_main
)
target_compile_features(mass_worker_term_screen_test PRIVATE cxx_std_23)
mass_worker_warnings(mass_worker_term_screen_test)
add_test(NAME term_screen COMMAND mass_worker_term_screen_test)

# Setup-form model tests — choice cycling, int/field validation, first-invalid
# scan. Pure model logic (no rendering, no tty).
add_executable(mass_worker_form_test
    form_test.cpp
)
target_link_libraries(mass_worker_form_test
    PRIVATE
        mass_worker_setup_lib
        GTest::gtest
        GTest::gtest_main
)
target_compile_features(mass_worker_form_test PRIVATE cxx_std_23)
mass_worker_warnings(mass_worker_form_test)
add_test(NAME form COMMAND mass_worker_form_test)

# The reusable three-column menu renderer — geometry (gutter-centered) + row
# styling. Pure, no tty.
add_executable(mass_worker_menu_test
    menu_test.cpp
)
target_link_libraries(mass_worker_menu_test
    PRIVATE
        mass_worker_setup_lib
        GTest::gtest
        GTest::gtest_main
)
target_compile_features(mass_worker_menu_test PRIVATE cxx_std_23)
mass_worker_warnings(mass_worker_menu_test)
add_test(NAME menu COMMAND mass_worker_menu_test)

# Wizard internals — the elevation_reason explanation (why root is needed) over
# the path_is_writable probe. Pure logic + a filesystem writability check; no tty,
# no actual install/elevation.
add_executable(mass_worker_wizard_test
    wizard_test.cpp
)
target_link_libraries(mass_worker_wizard_test
    PRIVATE
        mass_worker_setup_lib
        GTest::gtest
        GTest::gtest_main
)
target_compile_features(mass_worker_wizard_test PRIVATE cxx_std_23)
mass_worker_warnings(mass_worker_wizard_test)
add_test(NAME wizard COMMAND mass_worker_wizard_test)

# Go<->C++ TUI parity goldens — the form/menu/modal renders diffed byte-for-byte
# against mass-sdk's tui/testdata fixtures (copied into testdata/). Pure, no tty
# (styling pinned off via term::CapsOverride).
add_executable(mass_worker_golden_test
    golden_test.cpp
)
target_link_libraries(mass_worker_golden_test
    PRIVATE
        mass_worker_setup_lib
        GTest::gtest
        GTest::gtest_main
)
target_compile_definitions(mass_worker_golden_test
    PRIVATE MASS_WORKER_TESTDATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/testdata")
target_compile_features(mass_worker_golden_test PRIVATE cxx_std_23)
mass_worker_warnings(mass_worker_golden_test)
add_test(NAME golden COMMAND mass_worker_golden_test)

# Self-extracting payload round trip — the packer and reader driven against each
# other over the real record format: byte-for-byte content, the execute bit,
# empty files, same-file deduplication, and the failure modes that must never
# produce a half-installed worker (truncation, bit rot, a foreign format
# generation). Links the lean setup lib, which holds both halves.
add_executable(mass_worker_payload_test
    payload_test.cpp
)
target_link_libraries(mass_worker_payload_test
    PRIVATE
        mass_worker_setup_lib
        GTest::gtest
        GTest::gtest_main
)
target_compile_features(mass_worker_payload_test PRIVATE cxx_std_23)
mass_worker_warnings(mass_worker_payload_test)
add_test(NAME payload COMMAND mass_worker_payload_test)
