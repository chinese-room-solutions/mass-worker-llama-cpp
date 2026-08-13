# Convenience wrappers around CMake. The actual build system is CMake;
# these targets exist so `make build` / `make test` / `make lint` work
# the same way they do in sibling MASS repos.
#
# Backend selection: `make build` auto-detects the right GPU backend for
# the host — Metal on macOS (the only path on Apple Silicon), Vulkan
# (cross-vendor GPU) everywhere else. For NVIDIA-only maximum throughput
# use `make build-cuda`, which configures into a separate build-cuda/
# directory so the host-default tree stays intact. The per-backend targets
# (build-vulkan / build-metal / build-cuda) can also be invoked directly.

.PHONY: configure build build-vulkan build-metal build-cuda test lint clean format package

BUILD_DIR ?= build
BUILD_TYPE ?= Release
JOBS ?= $(shell nproc 2>/dev/null || echo 4)
DIST_DIR ?= dist

# Resolve the vcpkg toolchain file. The CMakeLists picks it up from
# $VCPKG_ROOT, but pass it explicitly when that's set so a clean build dir
# (which has no cached toolchain path) configures without relying on the env
# leaking through. Empty when VCPKG_ROOT is unset — then CMakeLists' own
# fallback applies.
ifdef VCPKG_ROOT
VCPKG_TOOLCHAIN := -DCMAKE_TOOLCHAIN_FILE=$(VCPKG_ROOT)/scripts/buildsystems/vcpkg.cmake
endif

# On Windows, static-link the vcpkg deps into our binaries (one worker exe +
# only the llama/ggml DLLs, not ~20 support DLLs) via the repo's Release-only
# static triplet. Passed on the cmake command line because vcpkg's manifest
# install won't honor an overlay triplet set from inside CMakeLists. Other OSes
# use their default (dynamic) triplet. Override by setting VCPKG_FLAGS= empty.
ifeq ($(OS),Windows_NT)
# Pin the MSVC toolset (-T version=...) so CMake's VS generator uses the SAME
# compiler vcpkg uses to build the dependencies. With multiple VS installs
# (e.g. Community + BuildTools) that update independently, vcpkg picks the newest
# toolset while CMake may default to an older one — the two then produce
# ABI-incompatible objects and the link fails with unresolved STL intrinsics
# (e.g. __std_find_first_of_trivial_pos_1, added in a newer STL). This bit us
# after a background VS update bumped BuildTools to 14.44 while Community stayed
# at 14.43.
#
# We pin an explicit known-good version (TOOLSET_PIN) for reproducibility, but
# fall back to the newest installed toolset if that exact version isn't present,
# so the repo still builds on a machine that has a different VS. Override either
# by passing TOOLSET_VERSION=<x.y.z> on the make command line.
TOOLSET_PIN := 14.44.35207
VSWHERE     := C:/Program Files (x86)/Microsoft Visual Studio/Installer/vswhere.exe
# All installed MSVC toolset versions, newest last.
MSVC_TOOLSETS = $(shell "$(VSWHERE)" -all -products '*' -property installationPath 2>/dev/null \
	| tr -d '\r' | tr '\\\\' '/' 2>/dev/null | sed 's|^C:|/c|' \
	| while IFS= read -r p; do ls -d "$$p"/VC/Tools/MSVC/*/ 2>/dev/null; done \
	| sed -E 's|.*/MSVC/([0-9.]+)/*$$|\1|' | sort -V)
# Use the pinned version if it's installed; otherwise the newest available.
TOOLSET_VERSION ?= $(if $(filter $(TOOLSET_PIN),$(MSVC_TOOLSETS)),$(TOOLSET_PIN),$(lastword $(MSVC_TOOLSETS)))
TOOLSET_FLAG := $(if $(TOOLSET_VERSION),-T version=$(TOOLSET_VERSION),)
VCPKG_FLAGS ?= $(VCPKG_TOOLCHAIN) $(TOOLSET_FLAG) \
	-DVCPKG_TARGET_TRIPLET=x64-windows-static-md-rel \
	-DVCPKG_OVERLAY_TRIPLETS=$(CURDIR)/triplets
else
VCPKG_FLAGS ?= $(VCPKG_TOOLCHAIN)
endif

# Skip the configure step when the build dir is already configured. The
# release workflow configures build/ itself (Ninja + a release triplet, no MSVC
# toolset pin) and then calls `make package`, whose `build` prerequisite would
# otherwise re-run `cmake -B build` with this Makefile's local-dev flags — the
# MSVC-generator-only `-T version=` (invalid under Ninja) and the
# static-md-rel overlay triplet — clobbering the already-configured tree into a
# reconfigure failure or a redundant vcpkg install. Reconfigure explicitly with
# `make RECONFIGURE=1 build` (a fresh dir always configures, cache or not).
#
# $1 = build dir. Emits the cmake configure command, or a skip notice.
configured = test -f "$(1)/CMakeCache.txt" && [ -z "$(RECONFIGURE)" ]
skip_msg = echo "$(1)/CMakeCache.txt exists; skipping configure (RECONFIGURE=1 to force)"

# Default `make build` follows the host OS: Metal on macOS, Vulkan elsewhere.
ifeq ($(shell uname -s),Darwin)
build: build-metal
else
build: build-vulkan
endif

build-vulkan:
	@$(call configured,build) && { $(call skip_msg,build); } || \
		cmake -B build -S . -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) $(VCPKG_FLAGS) \
			-DMASS_WORKER_LLAMA_VULKAN=ON -DMASS_WORKER_LLAMA_CUDA=OFF
	cmake --build build --config $(BUILD_TYPE) -j $(JOBS)

build-metal:
	@$(call configured,build) && { $(call skip_msg,build); } || \
		cmake -B build -S . -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) $(VCPKG_FLAGS) \
			-DMASS_WORKER_LLAMA_VULKAN=OFF -DMASS_WORKER_LLAMA_METAL=ON
	cmake --build build --config $(BUILD_TYPE) -j $(JOBS)

build-cuda:
	@$(call configured,build-cuda) && { $(call skip_msg,build-cuda); } || \
		cmake -B build-cuda -S . -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) $(VCPKG_FLAGS) \
			-DMASS_WORKER_LLAMA_VULKAN=OFF -DMASS_WORKER_LLAMA_CUDA=ON
	cmake --build build-cuda --config $(BUILD_TYPE) -j $(JOBS)

# Generic configure for the legacy single-build-dir flow. New code paths
# should prefer `build-vulkan` / `build-cuda`.
configure:
	@$(call configured,$(BUILD_DIR)) && { $(call skip_msg,$(BUILD_DIR)); } || \
		cmake -B $(BUILD_DIR) -S . -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) $(VCPKG_FLAGS)

test: build
	ctest --test-dir $(BUILD_DIR) -C $(BUILD_TYPE) --output-on-failure -j $(JOBS)

# Lint/format only what the host can compile: the other platforms' files
# have no compile commands here, so clang-tidy findings on them are parse
# noise, not analysis.
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
LINT_EXCLUDES := -not -name '*_windows*' -not -name '*_linux*'
else ifneq (,$(findstring NT,$(UNAME_S)))
LINT_EXCLUDES := -not -name '*_darwin*' -not -name '*_linux*' -not -name '*_posix*' -not -name '*.mm'
else
LINT_EXCLUDES := -not -name '*_windows*' -not -name '*_darwin*' -not -name '*.mm'
endif

# clang-tidy is single-threaded per process (upstream parallelises via the
# run-clang-tidy wrapper), so fan out with xargs like the build does — each
# file re-parses the vendored llama.cpp headers, and one process takes tens
# of minutes where $(JOBS) take a couple.
lint:
	@command -v clang-tidy >/dev/null 2>&1 || { echo "clang-tidy not on PATH"; exit 1; }
	@find src include tests \( -name '*.cpp' -o -name '*.hpp' \) $(LINT_EXCLUDES) | \
		xargs -P $(JOBS) -n 4 clang-tidy -p $(BUILD_DIR)

format:
	@command -v clang-format >/dev/null 2>&1 || { echo "clang-format not on PATH"; exit 1; }
	@find src include tests \( -name '*.cpp' -o -name '*.hpp' \) $(LINT_EXCLUDES) | \
		xargs -P $(JOBS) -n 4 clang-format -i

# Build the self-contained installer: mass-worker-setup with the worker binary
# and its runtime libraries appended (a self-extracting archive). Output lands
# in dist/. Mirrors the sibling repos' `make package` convention; here the
# bundling is done by the in-repo packer (tools/pack.cpp) rather than a .mass
# zip, because the worker ships native deps and the installer is one exe.
#
# The binaries live in $(BUILD_DIR)/bin (single-config) or
# $(BUILD_DIR)/bin/$(BUILD_TYPE) (the MSVC multi-config generator); BIN_DIR
# resolves whichever exists.
package: build
	cmake --build $(BUILD_DIR) --config $(BUILD_TYPE) -j $(JOBS) \
		--target mass-worker-setup mass-worker-pack
	@mkdir -p $(DIST_DIR)
	@set -e; \
	BIN_DIR="$(BUILD_DIR)/bin/$(BUILD_TYPE)"; \
	[ -d "$$BIN_DIR" ] || BIN_DIR="$(BUILD_DIR)/bin"; \
	echo "Packaging from $$BIN_DIR"; \
	: "Derive the executable suffix from the binary that was actually built"; \
	: "rather than the build-HOST OS, so a cross-compile (e.g. a Windows .exe"; \
	: "built on Linux) packages the right files."; \
	if [ -f "$$BIN_DIR/mass-worker-setup.exe" ]; then EXE_SUF=".exe"; else EXE_SUF=""; fi; \
	SETUP_OUT="mass-worker-setup$$EXE_SUF"; \
	for b in mass-worker-pack mass-worker-setup mass-worker-llama-cpp; do \
		[ -x "$$BIN_DIR/$$b$$EXE_SUF" ] || { echo "error: missing build output $$BIN_DIR/$$b$$EXE_SUF" >&2; exit 1; }; \
	done; \
	LIBS=$$(ls "$$BIN_DIR"/*.dll "$$BIN_DIR"/*.so "$$BIN_DIR"/*.so.* "$$BIN_DIR"/*.dylib 2>/dev/null \
		| grep -viE '/(gtest|gtest_main)[^/]*$$' || true); \
	"$$BIN_DIR/mass-worker-pack$$EXE_SUF" \
		--host "$$BIN_DIR/mass-worker-setup$$EXE_SUF" \
		--out "$(DIST_DIR)/$$SETUP_OUT" \
		"$$BIN_DIR/mass-worker-llama-cpp$$EXE_SUF" $$LIBS; \
	: "The packer writes a plain file; on Unix it must be executable so the"; \
	: "installer runs from dist/ (directly or via the double-click launcher)."; \
	[ "$$EXE_SUF" = ".exe" ] || chmod +x "$(DIST_DIR)/$$SETUP_OUT"; \
	echo "Installer: $(DIST_DIR)/$$SETUP_OUT"; \
	: "Wrap the console installer into the host OS's single double-clickable"; \
	: "container so a non-technical user can launch the terminal wizard with one"; \
	: "click and the dist folder stays clutter-free: AppImage (Linux) / .app"; \
	: "(macOS). The container opens a terminal for the wizard internally. Windows"; \
	: "needs none — the OS gives a console-subsystem exe its own console. Skipped"; \
	: "for a cross-built .exe (the container is a host-OS artifact)."; \
	if [ "$$EXE_SUF" != ".exe" ]; then \
	    BUNDLE=$$(sh "$(CURDIR)/tools/make-bundle.sh" \
	        --name "MASS Worker Setup" \
	        --id mass-worker-setup \
	        --bin "$(DIST_DIR)/$$SETUP_OUT" \
	        --outdir "$(DIST_DIR)"); \
	    if [ -n "$$BUNDLE" ] && [ -e "$$BUNDLE" ]; then \
	        : "The raw installer stays beside the container: the release uploads"; \
	        : "it as the join bootstrap's non-container executable, and it is"; \
	        : "what install.sh fetches."; \
	        echo "Bundle: $$BUNDLE"; \
	    else \
	        echo "warning: bundling produced no artifact; kept $(DIST_DIR)/$$SETUP_OUT" >&2; \
	    fi; \
	fi

clean:
	rm -rf build build-cuda dist
