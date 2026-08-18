#include "gtest/gtest.h"

#include <cstdlib>
#include <string>

#include "rtp_llm/cpp/engine_base/stream/SpeculativeSequenceLimit.h"

namespace rtp_llm {

namespace {

class StreamAsyncEnvScope {
public:
    StreamAsyncEnvScope() {
        const char* old = std::getenv("RTP_LLM_STREAM_ASYNC");
        had_old_        = old != nullptr;
        if (had_old_) {
            old_value_ = old;
        }
    }
    ~StreamAsyncEnvScope() {
        if (had_old_) {
            setenv("RTP_LLM_STREAM_ASYNC", old_value_.c_str(), 1);
        } else {
            unsetenv("RTP_LLM_STREAM_ASYNC");
        }
    }

private:
    bool        had_old_ = false;
    std::string old_value_;
};

}  // namespace

TEST(SpeculativeSequenceLimitTest, ReserveTruthTableMatchesSyncAndAsyncPipelines) {
    EXPECT_EQ(speculativeReservedTokenCount(/*propose_step=*/0, /*stream_async=*/false), 0);
    EXPECT_EQ(speculativeReservedTokenCount(/*propose_step=*/3, /*stream_async=*/false), 3);
    EXPECT_EQ(speculativeReservedTokenCount(/*propose_step=*/0, /*stream_async=*/true), 1);
    EXPECT_EQ(speculativeReservedTokenCount(/*propose_step=*/3, /*stream_async=*/true), 7);
}

TEST(SpeculativeSequenceLimitTest, StreamAsyncFlagAcceptsOnlyExactlyOne) {
    StreamAsyncEnvScope scope;

    unsetenv("RTP_LLM_STREAM_ASYNC");
    EXPECT_FALSE(speculativeStreamAsyncEnabled());

    setenv("RTP_LLM_STREAM_ASYNC", "1", 1);
    EXPECT_TRUE(speculativeStreamAsyncEnabled());

    // A permissive bool parser would report these as enabled, reserving 2*step+1 tokens for
    // an async pipeline that the other readers leave off.
    for (const char* value : {"0", "true", "True", "TRUE", "yes", "on", "2", ""}) {
        setenv("RTP_LLM_STREAM_ASYNC", value, 1);
        EXPECT_FALSE(speculativeStreamAsyncEnabled()) << "value=" << value;
    }
}

TEST(SpeculativeSequenceLimitTest, CommitsOnePrefixForEveryRemainingBoundary) {
    EXPECT_EQ(committedSpeculativeTokenCount(/*current_seq_len=*/7, /*effective_max_tokens=*/7, 3), 0);
    EXPECT_EQ(committedSpeculativeTokenCount(/*current_seq_len=*/6, /*effective_max_tokens=*/7, 3), 1);
    EXPECT_EQ(committedSpeculativeTokenCount(/*current_seq_len=*/5, /*effective_max_tokens=*/7, 3), 2);
    EXPECT_EQ(committedSpeculativeTokenCount(/*current_seq_len=*/4, /*effective_max_tokens=*/7, 3), 3);
    // A long-context boundary: only 2 of the 4 emitted tokens fit under the limit.
    EXPECT_EQ(committedSpeculativeTokenCount(
                  /*current_seq_len=*/56315, /*effective_max_tokens=*/56317, /*emitted_tokens=*/4),
              2);
}

}  // namespace rtp_llm
