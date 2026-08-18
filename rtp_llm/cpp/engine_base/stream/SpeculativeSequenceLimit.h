#pragma once

#include <cstddef>
#include <cstdlib>
#include <string>

namespace rtp_llm {

// Only the exact string "1" enables the async pipeline. Every other reader of this flag
// (MtpExecutor, MtpBatchStreamProcessor, DecodeRpcServer, cuda_graph_runner) and the Python
// admission check parse it that way, so accepting "true"/"yes" here would reserve tokens for
// a pipeline that stays off and shrink the limit below what admission allowed.
inline bool speculativeStreamAsyncEnabled() {
    const char* value = std::getenv("RTP_LLM_STREAM_ASYNC");
    return value != nullptr && std::string(value) == "1";
}

constexpr size_t speculativeReservedTokenCount(size_t propose_step, bool stream_async) {
    return stream_async ? propose_step * 2 + 1 : propose_step;
}

constexpr int committedSpeculativeTokenCount(int current_seq_len, size_t effective_max_tokens, int emitted_tokens) {
    if (current_seq_len < 0 || emitted_tokens <= 0
        || static_cast<size_t>(current_seq_len) >= effective_max_tokens) {
        return 0;
    }
    const size_t remaining = effective_max_tokens - static_cast<size_t>(current_seq_len);
    return remaining < static_cast<size_t>(emitted_tokens) ? static_cast<int>(remaining) : emitted_tokens;
}

}  // namespace rtp_llm
