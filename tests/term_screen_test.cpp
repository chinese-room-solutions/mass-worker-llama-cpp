#include "mass_worker/term_screen.hpp"

#include <gtest/gtest.h>

#include <functional>
#include <string>

#include "mass_worker/term.hpp"

#ifndef _WIN32
#include <sys/wait.h>
#include <unistd.h>

#include <array>
#include <csignal>
#endif

namespace mass_worker::term {
namespace {

// The restore is one fixed byte string, and every undo must be in it: a terminal
// left on the alternate screen, tinted, or without a cursor is a worse outcome
// than whatever output the screen was hiding.
TEST(TermScreen, RestoreSequenceCarriesEveryUndo) {
    const std::string seq(restore_sequence());
    EXPECT_NE(seq.find("\033[0m"), std::string::npos) << "SGR reset";
    EXPECT_NE(seq.find("\033[?25h"), std::string::npos) << "cursor shown";
    EXPECT_NE(seq.find("\033[?1049l"), std::string::npos) << "alternate screen left";
    EXPECT_NE(seq.find("\033]111"), std::string::npos) << "OSC 11 background reset";
    // The SGR reset must precede the buffer switch, or the page background bleeds
    // onto the restored main screen.
    EXPECT_LT(seq.find("\033[0m"), seq.find("\033[?1049l"));
}

// Styling off (a pipe, NO_COLOR, a terminal that refused VT) → the session is
// inert: no escapes, and nothing to restore.
TEST(TermScreen, IsInertWhenStylingIsOff) {
    const CapsOverride plain(false, false);
    ASSERT_EQ(screen_depth(), 0);
    {
        const Screen screen;
        EXPECT_EQ(screen_depth(), 0);
    }
    EXPECT_EQ(screen_depth(), 0);
}

#ifndef _WIN32

// Run `body` in a forked child with stdout on a pipe, and return what it wrote
// plus the child's wait status. Forking is what lets us assert on the actual
// BYTES — including the ones a signal handler writes, which no in-process test
// could observe (the handler re-raises and the process dies).
struct ChildRun {
    std::string output;
    int status{0};
};
ChildRun run_child(const std::function<void()>& body) {
    std::array<int, 2> fds{};
    if (::pipe(fds.data()) != 0) return {};
    const pid_t pid = ::fork();
    if (pid == 0) {
        ::close(fds[0]);
        ::dup2(fds[1], STDOUT_FILENO);
        ::close(fds[1]);
        body();
        ::_exit(0);
    }
    ::close(fds[1]);
    ChildRun run;
    std::array<char, 256> buf{};
    for (;;) {
        const ssize_t n = ::read(fds[0], buf.data(), buf.size());
        if (n <= 0) break;
        run.output.append(buf.data(), static_cast<std::size_t>(n));
    }
    ::close(fds[0]);
    ::waitpid(pid, &run.status, 0);
    return run;
}

// Nested Screens (a modal inside the wizard's session) enter and leave exactly
// once: the operator's screen must be switched away from — and back to — one
// time, however many views drew on it.
TEST(TermScreen, NestingEntersAndLeavesOnce) {
    const ChildRun run = run_child([] {
        const CapsOverride styled(true, true);
        const Screen outer;
        {
            const Screen inner;
            if (screen_depth() != 2) ::_exit(2);
        }
        if (screen_depth() != 1) ::_exit(3);
    });
    ASSERT_TRUE(WIFEXITED(run.status)) << "child died unexpectedly";
    ASSERT_EQ(WEXITSTATUS(run.status), 0);

    const std::string& out = run.output;
    EXPECT_EQ(out.find("\033[?1049h"), out.rfind("\033[?1049h")) << "entered more than once";
    EXPECT_EQ(out.find("\033[?1049l"), out.rfind("\033[?1049l")) << "left more than once";
    EXPECT_LT(out.find("\033[?1049h"), out.find("\033[?1049l"));
    // The whole restore, as the very last thing written.
    EXPECT_TRUE(out.ends_with(restore_sequence())) << "restore is not the final write";
}

// Signal death mid-render is the case RAII cannot cover: the process never
// unwinds. The handler must still restore the terminal, and must still die of
// what killed it so the caller's wait status is honest.
class TermScreenSignal : public testing::TestWithParam<int> {};

TEST_P(TermScreenSignal, RestoresTheTerminalAndReRaises) {
    const int sig = GetParam();
    const ChildRun run = run_child([sig] {
        const CapsOverride styled(true, true);
        const Screen screen;
        ::raise(sig);
        ::_exit(4);  // unreachable: the handler re-raises with the default action
    });
    ASSERT_TRUE(WIFSIGNALED(run.status)) << "the signal was swallowed, not re-raised";
    EXPECT_EQ(WTERMSIG(run.status), sig);
    EXPECT_TRUE(run.output.ends_with(restore_sequence()))
        << "the terminal was left on the alternate screen";
}

// SIGQUIT is handled the same way but left out here: its default action dumps
// core, which would have every test run tripping the host's core handler.
INSTANTIATE_TEST_SUITE_P(FatalSignals, TermScreenSignal, testing::Values(SIGTERM, SIGHUP, SIGINT));

// After the last Screen leaves, the handlers are gone: a later signal must kill
// the process the way it always would, without writing escapes at a terminal
// that is no longer ours.
TEST(TermScreen, HandlersAreRemovedWithTheLastScreen) {
    const ChildRun run = run_child([] {
        const CapsOverride styled(true, true);
        {
            const Screen screen;
        }
        ::raise(SIGTERM);
        ::_exit(5);
    });
    ASSERT_TRUE(WIFSIGNALED(run.status));
    EXPECT_EQ(WTERMSIG(run.status), SIGTERM);
    // Exactly one restore — the RAII one, not a second from the handler.
    const std::string& out = run.output;
    EXPECT_EQ(out.find(restore_sequence()), out.rfind(restore_sequence()));
}

#endif  // !_WIN32

}  // namespace
}  // namespace mass_worker::term
