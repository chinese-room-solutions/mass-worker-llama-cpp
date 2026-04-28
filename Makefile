# Convenience wrappers around CMake. The actual build system is CMake;
# these targets exist so `make build` / `make test` / `make lint` work
# the same way they do in sibling MASS repos.

.PHONY: configure build test lint clean format

BUILD_DIR ?= build
BUILD_TYPE ?= Release
JOBS ?= $(shell nproc 2>/dev/null || echo 4)

configure:
	cmake -B $(BUILD_DIR) -S . -DCMAKE_BUILD_TYPE=$(BUILD_TYPE)

build: configure
	cmake --build $(BUILD_DIR) --config $(BUILD_TYPE) -j $(JOBS)

test: build
	ctest --test-dir $(BUILD_DIR) -C $(BUILD_TYPE) --output-on-failure -j $(JOBS)

lint:
	@command -v clang-tidy >/dev/null 2>&1 || { echo "clang-tidy not on PATH"; exit 1; }
	@find src include tests -name '*.cpp' -o -name '*.hpp' | \
		xargs clang-tidy -p $(BUILD_DIR)

format:
	@command -v clang-format >/dev/null 2>&1 || { echo "clang-format not on PATH"; exit 1; }
	@find src include tests -name '*.cpp' -o -name '*.hpp' | \
		xargs clang-format -i

clean:
	rm -rf $(BUILD_DIR)
