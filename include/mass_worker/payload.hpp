#pragma once

#include <cstdint>
#include <expected>
#include <functional>
#include <string>
#include <vector>

namespace mass_worker {

// Self-extracting installer support.
//
// `make package` appends a payload (the worker binary + its runtime libraries)
// to a copy of mass-worker-setup, followed by a fixed-size trailer. The result
// is one file that carries everything it needs. At install time the setup exe
// reads its OWN file, finds the trailer, and writes the bundled files out.
//
// Payload layout, appended after the host exe's normal end-of-file:
//   [ record 0 ][ record 1 ] ... [ record N-1 ][ trailer ]
// record (little-endian):
//   uint32 name_len | name bytes (utf-8 leaf, no NUL)
//   uint8  kind
//   kind == kRecordFile:
//     uint64 raw_size  | uint64 comp_size | comp_size bytes (one zstd frame)
//   kind == kRecordAlias:
//     uint32 target_len | target bytes — the name of an EARLIER kRecordFile
//     whose contents this entry duplicates byte for byte.
// trailer (kTrailerSize bytes, little-endian, at the very end of the file):
//   uint64 payload_size  (bytes from the first record up to but excluding trailer)
//   uint64 file_count    (records, aliases included)
//   char   magic[8]      (kPayloadMagic)
//
// Two things keep the installer small. Compression: the shipped payload is
// ~113 MB of ELF/PE binaries, and zstd takes that to ~21 MB (5.3x) — it beat
// deflate 1.7x on the real payload, which is worth one vcpkg line for an
// artifact every worker host downloads. Per-file frames rather than one solid
// stream: solid saved 0.3% here, not enough to give up seekable records,
// per-record streaming, and per-file error attribution. Alias records: the
// libraries arrive as symlink chains (libggml.so -> .so.0 -> .so.0.10.0), so a
// naive pack stored every library three times; dropping the copies cut 167 MB,
// more than compression's 92 MB.
//
// The packer (append_payload) and the reader (extract_payload) share the
// constants below so they cannot drift.

// magic[8] = a 7-byte family prefix + a format generation. The prefix says
// "this file carries a MASS payload" at all; the generation says which layout.
// Split this way so a packer/reader mismatch is a loud versioned error instead
// of "no payload found" (silent no-op) or, worse, garbage records.
inline constexpr char kPayloadMagic[8] = {'M', 'W', 'P', 'L', 'O', 'A', 'D', '2'};
inline constexpr std::size_t kPayloadMagicPrefixLen = 7;
inline constexpr std::size_t kTrailerSize = 8 + 8 + 8;  // size + count + magic

inline constexpr std::uint8_t kRecordFile = 0;
inline constexpr std::uint8_t kRecordAlias = 1;

// zstd window, pinned on BOTH sides rather than left to the compression level's
// default: it is the installer's peak decompression footprint (2^23 = 8 MiB, on
// top of two 128 KiB streaming buffers). The reader also refuses frames that ask
// for more, so a corrupt or hostile frame header cannot make it allocate freely.
inline constexpr int kPayloadWindowLog = 23;

struct PayloadError {
    std::string message;
};

// Build a self-extracting installer at `out`: a byte-for-byte copy of `host`
// followed by `files` as a compressed payload and the trailer.
//
// Entries extract as flat leaves by basename, so duplicate basenames are
// rejected. Inputs that resolve to the same file on disk (the symlink chains
// llama.cpp/ggml install) are stored once, the rest as aliases; record order
// always follows input order.
[[nodiscard]] std::expected<void, PayloadError> append_payload(
    const std::string& host, const std::string& out, const std::vector<std::string>& files);

// True if `archive` has a payload appended (the trailer magic is present at its
// tail). False for the plain worker / a bare setup exe.
[[nodiscard]] bool has_payload(const std::string& archive);

// True if THIS running executable has a payload appended.
[[nodiscard]] bool has_appended_payload();

// Optional progress sink for extraction: called once per record with (done,
// total) after each file is written, so a caller can render a progress bar.
// done runs 1..total; total is the payload's record count.
using ProgressFn = std::function<void(std::size_t done, std::size_t total)>;

// Extract `archive`'s appended payload into dst_dir, creating it if needed
// (files overwrite). Returns the number of files written. It is an error to
// call this when has_payload() is false. If `progress` is set it is invoked
// after each extracted file (see ProgressFn).
[[nodiscard]] std::expected<std::uint64_t, PayloadError> extract_payload(
    const std::string& archive, const std::string& dst_dir, const ProgressFn& progress = {});

// extract_payload() on this running executable — what the installer uses.
[[nodiscard]] std::expected<std::uint64_t, PayloadError> extract_appended_payload(
    const std::string& dst_dir, const ProgressFn& progress = {});

}  // namespace mass_worker
