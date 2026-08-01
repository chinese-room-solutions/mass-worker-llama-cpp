#include "mass_worker/payload.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "mass_worker/service.hpp"  // current_executable_path

namespace {

namespace fs = std::filesystem;

using mass_worker::append_payload;
using mass_worker::extract_payload;
using mass_worker::has_payload;

std::string read_all(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    return {std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
}

void write_all(const fs::path& p, std::string_view bytes) {
    std::ofstream f(p, std::ios::binary | std::ios::trunc);
    f.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

// The payload the installer really carries is a handful of native binaries, so
// the fixture uses this test executable's own bytes: real ELF/PE content with
// realistic compressibility, no fixture committed to the repo.
class PayloadTest : public ::testing::Test {
protected:
    void SetUp() override {
        const auto* info = ::testing::UnitTest::GetInstance()->current_test_info();
        root_ = fs::temp_directory_path() / (std::string("mass_payload_test_") + info->name());
        fs::remove_all(root_);
        fs::create_directories(root_ / "src");
        dst_ = root_ / "out";

        host_ = root_ / "src" / "host-exe";
        write_all(host_, "host executable bytes, copied through verbatim");

        binary_ = root_ / "src" / "worker-bin";
        fs::copy_file(mass_worker::current_executable_path(), binary_);

        empty_ = root_ / "src" / "empty.dat";
        write_all(empty_, "");

        small_ = root_ / "src" / "small.dat";
        write_all(small_, std::string("\x00\x01\xFF binary-ish\n", 15));
    }
    void TearDown() override {
        std::error_code ec;
        fs::remove_all(root_, ec);
    }

    // Pack `files` onto the fixture host, returning the installer's path.
    fs::path pack(const std::vector<fs::path>& files, const std::string& out_name = "installer") {
        std::vector<std::string> args;
        args.reserve(files.size());
        for (const auto& f : files) args.push_back(f.string());
        const fs::path out = root_ / out_name;
        auto r = append_payload(host_.string(), out.string(), args);
        EXPECT_TRUE(r.has_value()) << (r ? "" : r.error().message);
        return out;
    }

    // Absolute offset of the first record's compressed data, and the payload
    // region's start — enough to corrupt a chosen byte deliberately.
    struct Layout {
        std::uint64_t payload_size;
        std::uint64_t first_data_begin;
    };
    static Layout layout_of(const fs::path& archive) {
        const std::string bytes = read_all(archive);
        const auto load_le = [&bytes](std::size_t at, std::size_t width) {
            std::uint64_t v = 0;
            for (std::size_t i = 0; i < width; ++i)
                v |= static_cast<std::uint64_t>(static_cast<unsigned char>(bytes[at + i]))
                     << (8 * i);
            return v;
        };
        const std::uint64_t payload_size = load_le(bytes.size() - mass_worker::kTrailerSize, 8);
        const auto begin =
            static_cast<std::size_t>(bytes.size() - mass_worker::kTrailerSize - payload_size);
        const std::uint64_t name_len = load_le(begin, 4);
        // name_len | name | kind | raw_size | comp_size | data
        return Layout{payload_size, begin + 4 + name_len + 1 + 8 + 8};
    }

    fs::path root_;
    fs::path dst_;
    fs::path host_;
    fs::path binary_;
    fs::path empty_;
    fs::path small_;
};

TEST_F(PayloadTest, RoundTripsBytesPermissionsAndEmptyFiles) {
    const fs::path installer = pack({binary_, empty_, small_});
    EXPECT_TRUE(has_payload(installer.string()));

    // The host's own bytes must survive at the front — the installer IS the host.
    const std::string packed = read_all(installer);
    const std::string host = read_all(host_);
    EXPECT_EQ(packed.substr(0, host.size()), host);

    std::vector<std::size_t> steps;
    auto n = extract_payload(installer.string(), dst_.string(),
                             [&steps](std::size_t done, std::size_t total) {
                                 EXPECT_EQ(total, 3u);
                                 steps.push_back(done);
                             });
    ASSERT_TRUE(n.has_value()) << (n ? "" : n.error().message);
    EXPECT_EQ(*n, 3u);
    EXPECT_EQ(steps, (std::vector<std::size_t>{1, 2, 3}));

    for (const auto& src : {binary_, empty_, small_}) {
        const fs::path got = dst_ / src.filename();
        ASSERT_TRUE(fs::exists(got)) << got;
        EXPECT_EQ(read_all(got), read_all(src)) << got;
        EXPECT_EQ(fs::file_size(got), fs::file_size(src)) << got;
        // The worker and its .so files are exec'd / dlopen'd from where they land.
        EXPECT_TRUE((fs::status(got).permissions() & fs::perms::owner_exec) != fs::perms::none)
            << got;
    }
    EXPECT_EQ(fs::file_size(dst_ / empty_.filename()), 0u);
}

TEST_F(PayloadTest, CompressesThePayload) {
    const fs::path installer = pack({binary_});
    // Loose bound on purpose — this asserts compression happened at all, not a
    // ratio that would break the test on a different toolchain's binary.
    EXPECT_LT(fs::file_size(installer), fs::file_size(host_) + (fs::file_size(binary_) / 2));
}

TEST_F(PayloadTest, StoresEntriesThatAreTheSameFileOnlyOnce) {
    // The ggml/llama libraries are shipped as name chains pointing at one file
    // (libggml.so -> .so.0 -> .so.0.10.0) and the packaging glob passes every
    // name, so storing each name's bytes tripled the payload. A hard link stands
    // in for the symlink chain: same file identity, and creatable without
    // privileges on Windows too.
    const fs::path link = root_ / "src" / "worker-bin.0";
    std::error_code ec;
    fs::create_hard_link(binary_, link, ec);
    if (ec) GTEST_SKIP() << "cannot hard-link on this filesystem: " << ec.message();

    const auto one = fs::file_size(pack({binary_}, "one"));
    const auto two = fs::file_size(pack({binary_, link}, "two"));
    // The second name costs a record header, not a second copy of the binary.
    EXPECT_LT(two, one + 128);

    auto n = extract_payload(pack({binary_, link}, "two").string(), dst_.string());
    ASSERT_TRUE(n.has_value()) << (n ? "" : n.error().message);
    EXPECT_EQ(*n, 2u);
    // Both names must land as ordinary independent files, as they always have.
    EXPECT_EQ(read_all(dst_ / "worker-bin"), read_all(binary_));
    EXPECT_EQ(read_all(dst_ / "worker-bin.0"), read_all(binary_));
    EXPECT_FALSE(fs::is_symlink(dst_ / "worker-bin.0"));
    EXPECT_TRUE((fs::status(dst_ / "worker-bin.0").permissions() & fs::perms::owner_exec) !=
                fs::perms::none);
}

TEST_F(PayloadTest, RejectsDuplicateBasenames) {
    fs::create_directories(root_ / "src2");
    const fs::path clash = root_ / "src2" / small_.filename();
    write_all(clash, "different bytes, same leaf name");

    auto r = append_payload(host_.string(), (root_ / "clash").string(),
                            {small_.string(), clash.string()});
    ASSERT_FALSE(r.has_value());
    EXPECT_NE(r.error().message.find("duplicate payload entry name"), std::string::npos)
        << r.error().message;
}

TEST_F(PayloadTest, PlainFileHasNoPayload) {
    EXPECT_FALSE(has_payload(host_.string()));
    EXPECT_FALSE(has_payload((root_ / "does-not-exist").string()));
    auto r = extract_payload(host_.string(), dst_.string());
    EXPECT_FALSE(r.has_value());
}

TEST_F(PayloadTest, TruncatedArchiveIsRejected) {
    const fs::path installer = pack({binary_, small_});
    std::string bytes = read_all(installer);
    write_all(installer, std::string_view(bytes).substr(0, bytes.size() / 2));

    // Truncation takes the trailer with it, so the file reads as payload-free —
    // the installer refuses to provision rather than writing a partial worker.
    EXPECT_FALSE(has_payload(installer.string()));
    auto r = extract_payload(installer.string(), dst_.string());
    ASSERT_FALSE(r.has_value());
    EXPECT_FALSE(fs::exists(dst_ / binary_.filename()));
}

TEST_F(PayloadTest, ClippedPayloadRegionIsRejected) {
    // A consistent trailer over a payload whose tail is missing — what a partial
    // write leaves behind. The record walk must run off the end of the region and
    // say so, not extract a half binary and call the install done.
    const fs::path installer = pack({binary_, small_});
    std::string bytes = read_all(installer);
    const std::size_t trailer_at = bytes.size() - mass_worker::kTrailerSize;
    constexpr std::uint64_t kCut = 1024;
    const std::uint64_t shrunk = layout_of(installer).payload_size - kCut;
    std::string payload_size(8, '\0');
    for (std::size_t i = 0; i < 8; ++i)
        payload_size[i] = static_cast<char>((shrunk >> (8 * i)) & 0xFF);
    write_all(installer, bytes.substr(0, trailer_at - kCut) + payload_size +
                             bytes.substr(trailer_at + 8, mass_worker::kTrailerSize - 8));

    auto r = extract_payload(installer.string(), dst_.string());
    ASSERT_FALSE(r.has_value());
    EXPECT_NE(r.error().message.find("truncated payload"), std::string::npos) << r.error().message;
}

TEST_F(PayloadTest, CorruptCompressedDataIsRejected) {
    const fs::path installer = pack({binary_, small_});
    const auto at = layout_of(installer).first_data_begin + 4096;
    std::string bytes = read_all(installer);
    ASSERT_LT(at, bytes.size());
    bytes[at] = static_cast<char>(bytes[at] ^ 0x40);
    write_all(installer, bytes);

    auto r = extract_payload(installer.string(), dst_.string());
    ASSERT_FALSE(r.has_value());
    EXPECT_NE(r.error().message.find("corrupt payload entry"), std::string::npos)
        << r.error().message;
    // Whatever was written before the failure must not pass for the real thing.
    EXPECT_NE(read_all(dst_ / binary_.filename()), read_all(binary_));
}

TEST_F(PayloadTest, ForeignFormatGenerationIsRejectedLoudly) {
    // A packer and a reader from different builds: the family magic still
    // matches, so this is a payload — just not one we know how to read. It has
    // to fail with that in the message, not look like a plain exe.
    const fs::path installer = pack({small_});
    std::string bytes = read_all(installer);
    bytes[bytes.size() - 1] = 'Z';  // magic's generation byte
    write_all(installer, bytes);

    EXPECT_TRUE(has_payload(installer.string()));
    auto r = extract_payload(installer.string(), dst_.string());
    ASSERT_FALSE(r.has_value());
    EXPECT_NE(r.error().message.find("payload format"), std::string::npos) << r.error().message;
    EXPECT_NE(r.error().message.find("different builds"), std::string::npos) << r.error().message;
}

}  // namespace
