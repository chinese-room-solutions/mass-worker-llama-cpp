// Writer half of the self-extracting installer format. Kept beside the reader
// (payload.cpp) and sharing its constants so the two can't drift; tools/pack.cpp
// is a thin CLI over this, and the round-trip test drives both halves directly.
//
// This TU is the only user of zstd's compressor. It lives in its own object file
// so the linker leaves it out of mass-worker-setup, which decompresses but never
// compresses.

#include <zstd.h>

#include <algorithm>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <fstream>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "mass_worker/payload.hpp"

namespace mass_worker {

namespace {

namespace fs = std::filesystem;

// Level 19 on the shipped payload: 5.3x, against ~4.5x at both 15 and 9. The
// window is pinned at kPayloadWindowLog for every level, so the gain is level
// 19's match search, not a wider window — ~30 s of one-off packaging time per
// release for ~4 MB off every download.
constexpr int kCompressionLevel = 19;

void put_u32_le(std::ostream& o, std::uint32_t v) {
    for (int i = 0; i < 4; ++i) o.put(static_cast<char>((v >> (8 * i)) & 0xFF));
}

void put_u64_le(std::ostream& o, std::uint64_t v) {
    for (int i = 0; i < 8; ++i) o.put(static_cast<char>((v >> (8 * i)) & 0xFF));
}

void put_name(std::ostream& o, const std::string& name) {
    put_u32_le(o, static_cast<std::uint32_t>(name.size()));
    o.write(name.data(), static_cast<std::streamsize>(name.size()));
}

// One payload entry: either its own compressed record, or a second name for an
// earlier entry that resolves to the same file on disk.
struct Entry {
    static constexpr std::size_t kNoAlias = static_cast<std::size_t>(-1);

    fs::path path;
    std::string name;
    std::size_t alias_of = kNoAlias;
};

// Streaming zstd compressor: one reusable context plus its two I/O buffers, so
// the packer never holds a whole library in memory.
class Deflater {
public:
    Deflater() : in_(ZSTD_CStreamInSize()), out_(ZSTD_CStreamOutSize()) {}
    ~Deflater() { ZSTD_freeCCtx(ctx_); }
    Deflater(const Deflater&) = delete;
    Deflater& operator=(const Deflater&) = delete;

    [[nodiscard]] bool valid() const { return ctx_ != nullptr; }

    // Compress raw_size bytes from src as one zstd frame appended to dst,
    // returning the frame's byte length.
    [[nodiscard]] std::expected<std::uint64_t, PayloadError> run(std::ifstream& src,
                                                                 std::uint64_t raw_size,
                                                                 std::ostream& dst,
                                                                 const std::string& name) {
        const auto fail = [&name](const std::string& why) {
            return std::unexpected(PayloadError{"could not compress " + name + ": " + why});
        };

        ZSTD_CCtx_reset(ctx_, ZSTD_reset_session_and_parameters);
        const auto set = [this](ZSTD_cParameter p, int v) {
            return !ZSTD_isError(ZSTD_CCtx_setParameter(ctx_, p, v));
        };
        // Frame checksum: a bit flip in a shipped installer then fails the
        // decode loudly instead of writing a plausible-looking binary.
        if (!set(ZSTD_c_compressionLevel, kCompressionLevel) ||
            !set(ZSTD_c_windowLog, kPayloadWindowLog) || !set(ZSTD_c_checksumFlag, 1)) {
            return fail("the compressor rejected its settings");
        }
        if (ZSTD_isError(ZSTD_CCtx_setPledgedSrcSize(ctx_, raw_size))) {
            return fail("the compressor rejected the source size");
        }

        std::uint64_t left = raw_size;
        std::uint64_t produced = 0;
        for (;;) {
            std::size_t got = 0;
            if (left > 0) {
                const auto want =
                    static_cast<std::streamsize>(std::min<std::uint64_t>(left, in_.size()));
                src.read(in_.data(), want);
                got = static_cast<std::size_t>(src.gcount());
                if (got == 0) return fail("it shrank while being packed");
                left -= got;
            }
            const ZSTD_EndDirective mode = left == 0 ? ZSTD_e_end : ZSTD_e_continue;
            ZSTD_inBuffer zin{in_.data(), got, 0};
            for (;;) {
                ZSTD_outBuffer zout{out_.data(), out_.size(), 0};
                const std::size_t rem = ZSTD_compressStream2(ctx_, &zout, &zin, mode);
                if (ZSTD_isError(rem)) return fail(ZSTD_getErrorName(rem));
                dst.write(out_.data(), static_cast<std::streamsize>(zout.pos));
                if (!dst) return fail("write failed");
                produced += zout.pos;
                if (mode == ZSTD_e_end ? rem == 0 : zin.pos == zin.size) break;
            }
            if (mode == ZSTD_e_end) return produced;
        }
    }

private:
    ZSTD_CCtx* ctx_ = ZSTD_createCCtx();
    std::vector<char> in_;
    std::vector<char> out_;
};

// Resolve the input list into entries, rejecting packaging mistakes and marking
// every input that resolves to a file an earlier input already covers.
std::expected<std::vector<Entry>, PayloadError> plan_entries(
    const std::vector<std::string>& files) {
    std::vector<Entry> entries;
    entries.reserve(files.size());
    for (const auto& f : files) {
        Entry e{.path = fs::path(f), .name = fs::path(f).filename().string()};
        if (e.name.empty()) {
            return std::unexpected(PayloadError{"payload entry '" + f + "' has no filename"});
        }
        // Entries extract as flat leaves by basename, so two inputs with the
        // same filename would silently overwrite each other at install time.
        // Reject that here — it's a packaging mistake (e.g. globbing two dirs),
        // and the half-extracted install it produces is far harder to diagnose.
        for (const auto& prior : entries) {
            if (prior.name == e.name) {
                return std::unexpected(PayloadError{"duplicate payload entry name '" + e.name +
                                                    "' — entries must have unique filenames"});
            }
        }
        std::error_code ec;
        if (!fs::exists(e.path, ec) || ec) {
            return std::unexpected(PayloadError{"payload entry " + f + " does not exist"});
        }
        // The ggml/llama libraries arrive as symlink chains (libggml.so ->
        // .so.0 -> .so.0.10.0) and the packaging glob picks up every name, so a
        // naive pack stored each library three times — 59% of the old payload.
        // Identity, not content hashing: it addresses the actual cause exactly,
        // needs no pass over 100+ MB, and can't collide.
        for (std::size_t i = 0; i < entries.size(); ++i) {
            if (entries[i].alias_of == Entry::kNoAlias &&
                fs::equivalent(entries[i].path, e.path, ec) && !ec) {
                e.alias_of = i;
                break;
            }
        }
        entries.push_back(std::move(e));
    }
    return entries;
}

}  // namespace

std::expected<void, PayloadError> append_payload(const std::string& host, const std::string& out,
                                                 const std::vector<std::string>& files) {
    if (files.empty()) {
        return std::unexpected(PayloadError{"refusing to write a payload with no files"});
    }
    auto entries = plan_entries(files);
    if (!entries) return std::unexpected(entries.error());

    Deflater deflater;
    if (!deflater.valid()) {
        return std::unexpected(PayloadError{"could not create the payload compressor"});
    }

    std::error_code ec;
    fs::create_directories(fs::path(out).parent_path(), ec);

    // Start the installer as a byte-for-byte copy of the host exe.
    std::ofstream o(out, std::ios::binary | std::ios::trunc);
    if (!o) return std::unexpected(PayloadError{"cannot open output " + out});
    {
        std::ifstream in(host, std::ios::binary);
        if (!in) return std::unexpected(PayloadError{"cannot read host exe " + host});
        o << in.rdbuf();
        if (!o) return std::unexpected(PayloadError{"cannot copy host exe " + host});
    }

    // The payload begins here; track its byte length for the trailer.
    const std::streampos payload_begin = o.tellp();

    for (const auto& e : *entries) {
        put_name(o, e.name);
        if (e.alias_of != Entry::kNoAlias) {
            o.put(static_cast<char>(kRecordAlias));
            put_name(o, (*entries)[e.alias_of].name);
            continue;
        }
        const auto raw_size = static_cast<std::uint64_t>(fs::file_size(e.path, ec));
        if (ec) {
            return std::unexpected(
                PayloadError{"cannot stat " + e.path.string() + ": " + ec.message()});
        }
        std::ifstream in(e.path, std::ios::binary);
        if (!in) {
            return std::unexpected(PayloadError{"cannot read " + e.path.string()});
        }
        o.put(static_cast<char>(kRecordFile));
        put_u64_le(o, raw_size);
        // The frame length isn't known until it's written, so reserve its slot
        // and patch it once the data is down.
        const std::streampos comp_size_at = o.tellp();
        put_u64_le(o, std::uint64_t{0});
        auto comp_size = deflater.run(in, raw_size, o, e.name);
        if (!comp_size) return std::unexpected(comp_size.error());
        o.seekp(comp_size_at);
        put_u64_le(o, *comp_size);
        o.seekp(0, std::ios::end);
        if (!o) return std::unexpected(PayloadError{"write failed packing " + e.name});
    }

    // Trailer: payload_size, file_count, magic — read from the file's tail.
    put_u64_le(o, static_cast<std::uint64_t>(o.tellp() - payload_begin));
    put_u64_le(o, static_cast<std::uint64_t>(entries->size()));
    o.write(kPayloadMagic, sizeof(kPayloadMagic));
    o.close();
    if (!o) return std::unexpected(PayloadError{"write failed finishing " + out});
    return {};
}

}  // namespace mass_worker
