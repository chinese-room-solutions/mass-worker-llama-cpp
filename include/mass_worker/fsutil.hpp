#pragma once

#include <string>
#include <string_view>

namespace mass_worker::fsutil {

// Write `content` to `path` as an owner-only (0600) file, atomically and without
// a permission race. Used for the worker's small state files (credentials,
// config.conf, install.record) — some hold a bearer token, and even the others
// shouldn't be world-readable on a shared host.
//
// The write is done as: create a sibling temp file, restrict it to 0600 BEFORE
// writing any content (so the bytes are never momentarily world-readable under a
// permissive umask), write + flush, then atomically rename it into place (a
// same-directory rename is atomic, so a reader sees either the old file or the
// complete new one, never a half-write). On Windows the chmod is a best-effort
// no-op (ACLs differ); the atomic-rename behaviour still holds.
//
// Returns false (and leaves any prior file intact) on any I/O failure.
[[nodiscard]] bool write_private_file(std::string_view path, std::string_view content);

}  // namespace mass_worker::fsutil
