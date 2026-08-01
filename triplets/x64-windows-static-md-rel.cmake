# Custom vcpkg triplet: x64-windows-static-md, Release-only.
#
# Same as the stock x64-windows-static-md (static deps, dynamic CRT) but builds
# ONLY the Release variant of each dependency — vcpkg builds both Debug and
# Release by default, which doubles the (already long) cold dependency build for
# a Debug half we never ship. Our own code still builds Debug or Release freely
# via CMAKE_BUILD_TYPE; this only affects the vcpkg deps.
#
# Selected via VCPKG_OVERLAY_TRIPLETS=triplets + VCPKG_TARGET_TRIPLET set in the
# top-level CMakeLists. To debug a dependency itself, drop VCPKG_BUILD_TYPE (or
# use the stock x64-windows-static-md triplet) so its Debug variant is built.
set(VCPKG_TARGET_ARCHITECTURE x64)
set(VCPKG_CRT_LINKAGE dynamic)
set(VCPKG_LIBRARY_LINKAGE static)
set(VCPKG_BUILD_TYPE release)
