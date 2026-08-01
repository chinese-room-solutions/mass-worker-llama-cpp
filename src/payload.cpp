#include "mass_worker/payload.hpp"

#include <spdlog/spdlog.h>
#include <zstd.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <optional>
#include <utility>
#include <vector>

#include "mass_worker/service.hpp"  // current_executable_path

namespace mass_worker {

namespace {

namespace fs = std::filesystem;

// Read a fixed number of bytes at an absolute offset. Returns false on short
// read / seek failure so callers can treat a truncated file as "no payload".
bool read_at(std::ifstream& f, std::uint64_t offset, void* out, std::size_t n) {
    f.clear();
    f.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
    if (!f) return false;
    f.read(static_cast<char*>(out), static_cast<std::streamsize>(n));
    return std::cmp_equal(f.gcount(), n);
}

std::uint64_t load_u64_le(const unsigned char* p) {
    std::uint64_t v = 0;
    for (int i = 0; i < 8; ++i) v |= static_cast<std::uint64_t>(p[i]) << (8 * i);
    return v;
}

std::uint32_t load_u32_le(const unsigned char* p) {
    std::uint32_t v = 0;
    for (int i = 0; i < 4; ++i) v |= static_cast<std::uint32_t>(p[i]) << (8 * i);
    return v;
}

struct Trailer {
    std::uint64_t payload_size;
    std::uint64_t file_count;
    std::uint64_t file_size;
    char format;  // the magic's generation byte; validated at extract time
};

// Parse the trailer at the tail of `archive`. nullopt when there is no payload
// at all (magic prefix absent / file too small / payload doesn't fit). A payload
// in a format this build can't read still parses — extract_payload reports that
// as a versioned error rather than silently installing nothing.
std::optional<Trailer> read_trailer(const std::string& archive, std::ifstream& f) {
    std::error_code ec;
    const auto size = static_cast<std::uint64_t>(fs::file_size(archive, ec));
    if (ec || size < kTrailerSize) return std::nullopt;

    std::array<unsigned char, kTrailerSize> buf{};
    if (!read_at(f, size - kTrailerSize, buf.data(), buf.size())) return std::nullopt;

    if (std::memcmp(buf.data() + 16, kPayloadMagic, kPayloadMagicPrefixLen) != 0) {
        return std::nullopt;  // no magic → a plain exe, not a package
    }
    const std::uint64_t payload_size = load_u64_le(buf.data());
    const std::uint64_t file_count = load_u64_le(buf.data() + 8);
    // Sanity: the payload + trailer must fit inside the file.
    if (payload_size + kTrailerSize > size) return std::nullopt;
    return Trailer{payload_size, file_count, size,
                   static_cast<char>(buf[16 + kPayloadMagicPrefixLen])};
}

// A payload entry is a flat leaf beside the worker, never a traversal. Reject
// path separators AND ':' — on Windows the latter is the Alternate Data Stream
// separator (e.g. "worker.dll:hidden" writes a hidden stream rather than a
// file), which would slip past a separator-only check.
bool safe_entry_name(const std::string& name) {
    return !name.empty() && name.find_first_of("/\\:") == std::string::npos;
}

// Streaming zstd decompressor: one reusable context plus its two I/O buffers, so
// extracting N records allocates the 8 MiB window once rather than per record.
class Inflater {
public:
    Inflater() : in_(ZSTD_DStreamInSize()), out_(ZSTD_DStreamOutSize()) {}
    ~Inflater() { ZSTD_freeDCtx(ctx_); }
    Inflater(const Inflater&) = delete;
    Inflater& operator=(const Inflater&) = delete;

    [[nodiscard]] bool valid() const { return ctx_ != nullptr; }

    // Decompress the single zstd frame at src's [offset, offset+comp_size) into
    // dst, which must receive exactly raw_size bytes. `name` only labels errors.
    [[nodiscard]] std::expected<void, PayloadError> run(std::ifstream& src, std::uint64_t offset,
                                                        std::uint64_t comp_size,
                                                        std::uint64_t raw_size, std::ofstream& dst,
                                                        const std::string& name) {
        const auto fail = [&name](const std::string& why) {
            return std::unexpected(PayloadError{"corrupt payload entry " + name + ": " + why});
        };

        ZSTD_DCtx_reset(ctx_, ZSTD_reset_session_only);
        // Refuse frames that want more window than the packer is allowed to ask
        // for: keeps the installer's footprint bounded even on a garbage header.
        if (ZSTD_isError(ZSTD_DCtx_setParameter(ctx_, ZSTD_d_windowLogMax, kPayloadWindowLog))) {
            return fail("could not bound the decompressor window");
        }

        src.clear();
        src.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
        if (!src) return fail("could not seek to its data");

        std::uint64_t left = comp_size;
        std::uint64_t produced = 0;
        ZSTD_inBuffer zin{in_.data(), 0, 0};
        std::size_t need = 1;  // nonzero after the last call ⇒ frame incomplete
        while (need != 0) {
            if (zin.pos == zin.size) {
                if (left == 0) break;  // ran out of record before the frame ended
                const auto want =
                    static_cast<std::streamsize>(std::min<std::uint64_t>(left, in_.size()));
                src.read(in_.data(), want);
                if (src.gcount() != want) return fail("short read");
                left -= static_cast<std::uint64_t>(want);
                zin = ZSTD_inBuffer{in_.data(), static_cast<std::size_t>(want), 0};
            }
            ZSTD_outBuffer zout{out_.data(), out_.size(), 0};
            need = ZSTD_decompressStream(ctx_, &zout, &zin);
            if (ZSTD_isError(need)) return fail(ZSTD_getErrorName(need));
            produced += zout.pos;
            if (produced > raw_size) return fail("more data than its header declares");
            dst.write(out_.data(), static_cast<std::streamsize>(zout.pos));
            if (!dst) return std::unexpected(PayloadError{"write failed extracting " + name});
        }
        if (need != 0) return fail("truncated compressed data");
        if (left != 0 || zin.pos != zin.size) return fail("trailing bytes after its frame");
        if (produced != raw_size) {
            return fail("decompressed to " + std::to_string(produced) + " bytes, header declares " +
                        std::to_string(raw_size));
        }
        return {};
    }

private:
    ZSTD_DCtx* ctx_ = ZSTD_createDCtx();
    std::vector<char> in_;
    std::vector<char> out_;
};

// The worker executable and its shared libraries both need the execute bit (a
// non-executable binary fails to launch with systemd's 203/EXEC; .so files are
// conventionally 0755 too), and ofstream/copy_file create 0644. ec ignored: on
// Windows perms are ACL-governed and this is a best-effort no-op.
void make_executable(const fs::path& p) {
    std::error_code ec;
    fs::permissions(p,
                    fs::perms::owner_all | fs::perms::group_read | fs::perms::group_exec |
                        fs::perms::others_read | fs::perms::others_exec,
                    fs::perm_options::replace, ec);
}

}  // namespace

bool has_payload(const std::string& archive) {
    std::ifstream f(archive, std::ios::binary);
    if (!f) return false;
    return read_trailer(archive, f).has_value();
}

bool has_appended_payload() {
    const std::string self = current_executable_path();
    return !self.empty() && has_payload(self);
}

std::expected<std::uint64_t, PayloadError> extract_payload(const std::string& archive,
                                                           const std::string& dst_dir,
                                                           const ProgressFn& progress) {
    std::ifstream f(archive, std::ios::binary);
    if (!f) {
        return std::unexpected(PayloadError{"could not open " + archive + " to read its payload"});
    }
    const auto trailer = read_trailer(archive, f);
    if (!trailer) {
        return std::unexpected(PayloadError{"no appended payload found in " + archive});
    }
    if (trailer->format != kPayloadMagic[kPayloadMagicPrefixLen]) {
        return std::unexpected(PayloadError{std::string("payload format '") + trailer->format +
                                            "' cannot be read by this build (it "
                                            "reads '" +
                                            kPayloadMagic[kPayloadMagicPrefixLen] +
                                            "'); the packer and the installer "
                                            "come from different builds — rebuild and repackage"});
    }

    Inflater inflater;
    if (!inflater.valid()) {
        return std::unexpected(PayloadError{"could not create the payload decompressor"});
    }

    std::error_code ec;
    fs::create_directories(dst_dir, ec);
    if (ec) {
        return std::unexpected(
            PayloadError{"could not create install directory " + dst_dir + ": " + ec.message()});
    }

    // The payload begins payload_size+trailer bytes before EOF.
    std::uint64_t off = trailer->file_size - kTrailerSize - trailer->payload_size;
    const std::uint64_t end = trailer->file_size - kTrailerSize;

    // Read n bytes of record header at `off`, advancing it. Any overrun of the
    // payload region is a truncated/corrupt package, never a valid short record.
    const auto take = [&](void* out, std::uint64_t n) {
        if (off + n > end || !read_at(f, off, out, static_cast<std::size_t>(n))) return false;
        off += n;
        return true;
    };
    const auto take_name = [&](std::string& name) {
        unsigned char len[4];
        if (!take(len, 4)) return false;
        name.assign(load_u32_le(len), '\0');
        return take(name.data(), name.size());
    };

    std::uint64_t written = 0;
    for (std::uint64_t i = 0; i < trailer->file_count; ++i) {
        std::string name;
        if (!take_name(name)) {
            return std::unexpected(PayloadError{"truncated payload (entry name)"});
        }
        if (!safe_entry_name(name)) {
            return std::unexpected(PayloadError{"unsafe payload entry name: " + name});
        }
        unsigned char kind = 0;
        if (!take(&kind, 1)) {
            return std::unexpected(PayloadError{"truncated payload (entry kind for " + name + ")"});
        }
        const fs::path dst = fs::path(dst_dir) / name;

        if (kind == kRecordAlias) {
            // A second name for a file already written (the symlink chains the
            // llama.cpp/ggml install produces). Materialised as a real copy, not
            // a link: the install layout stays what it has always been, and
            // Windows needs a privilege for symlinks.
            std::string target;
            if (!take_name(target)) {
                return std::unexpected(
                    PayloadError{"truncated payload (alias target for " + name + ")"});
            }
            if (!safe_entry_name(target)) {
                return std::unexpected(PayloadError{"unsafe payload alias target: " + target});
            }
            fs::copy_file(fs::path(dst_dir) / target, dst, fs::copy_options::overwrite_existing,
                          ec);
            if (ec) {
                std::string why = "could not copy " + target;
                why += " to " + name;
                why += ": " + ec.message();
                return std::unexpected(PayloadError{why});
            }
        } else if (kind == kRecordFile) {
            unsigned char sizes[16];
            if (!take(sizes, 16)) {
                return std::unexpected(PayloadError{"truncated payload (sizes for " + name + ")"});
            }
            const std::uint64_t raw_size = load_u64_le(sizes);
            const std::uint64_t comp_size = load_u64_le(sizes + 8);
            if (off + comp_size > end) {
                return std::unexpected(PayloadError{"truncated payload (data for " + name + ")"});
            }
            std::ofstream out(dst, std::ios::binary | std::ios::trunc);
            if (!out) {
                return std::unexpected(
                    PayloadError{"could not open " + dst.string() + " for writing"});
            }
            if (auto r = inflater.run(f, off, comp_size, raw_size, out, name); !r) {
                return std::unexpected(r.error());
            }
            out.close();  // flush + surface a failed final write before we chmod
            if (!out) {
                return std::unexpected(PayloadError{"write failed extracting " + name});
            }
            off += comp_size;
        } else {
            return std::unexpected(PayloadError{"unknown payload entry kind " +
                                                std::to_string(kind) + " for " + name});
        }

        make_executable(dst);
        ++written;
        if (progress)
            progress(static_cast<std::size_t>(written),
                     static_cast<std::size_t>(trailer->file_count));
    }

    spdlog::info("extracted {} payload files into {}", written, dst_dir);
    return written;
}

std::expected<std::uint64_t, PayloadError> extract_appended_payload(const std::string& dst_dir,
                                                                    const ProgressFn& progress) {
    const std::string self = current_executable_path();
    if (self.empty()) {
        return std::unexpected(PayloadError{"could not locate the installer to read its payload"});
    }
    return extract_payload(self, dst_dir, progress);
}

}  // namespace mass_worker
