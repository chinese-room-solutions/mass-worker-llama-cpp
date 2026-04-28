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
add_test(NAME runner COMMAND mass_worker_runner_test)

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
add_test(NAME hardware COMMAND mass_worker_hardware_test)

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
add_test(NAME fetch COMMAND mass_worker_fetch_test)
