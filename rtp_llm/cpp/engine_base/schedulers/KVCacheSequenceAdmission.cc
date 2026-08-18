#include "rtp_llm/cpp/engine_base/schedulers/KVCacheSequenceAdmission.h"

#include <string>

#include "rtp_llm/cpp/utils/ErrorCode.h"

namespace rtp_llm {

KVCacheSequenceLimits sequenceLimitsForAdmission(const GenerateStreamPtr&             stream,
                                                 const std::shared_ptr<KVCacheManager>& cache_manager) {
    const size_t physical_capacity = cache_manager ? cache_manager->maxAvailableTokensNum() : 0;
    const size_t reserve_step      = stream->reserveStep();
    size_t       logical_limit     = stream->maxTokenNum();
    if (reserve_step > 0 && stream->getSPOutputBuffer() == nullptr) {
        // maxTokenNum() subtracts the executor window only once the speculative output buffer is
        // attached, and that happens after admission. Without this, a prompt that leaves no room
        // for the window is admitted here, and the moment the buffer appears the limit drops below
        // the prompt length: the stream then finishes having produced nothing, which is exactly the
        // outcome the admission check exists to prevent. Subtracting reserved_token_count reproduces
        // the earlier rule input_length + reserve_step <= max_seq_len.
        const size_t reserved = kvCacheReservedTokenCount(reserve_step);
        logical_limit         = logical_limit > reserved ? logical_limit - reserved : 0;
    }
    return calculateKVCacheSequenceLimits(physical_capacity, reserve_step, logical_limit);
}

bool admitStreamToKVCache(const GenerateStreamPtr& stream, const std::shared_ptr<KVCacheManager>& cache_manager) {
    const int input_length_value = stream->inputLength();
    auto&     max_new_tokens     = stream->generateConfig()->max_new_tokens;
    const int min_new_tokens     = stream->generateConfig()->min_new_tokens;
    if (input_length_value < 0 || max_new_tokens <= 0 || min_new_tokens < 0) {
        stream->reportError(ErrorCode::INVALID_PARAMS,
                            "input length and generation token counts must be non-negative, and max_new_tokens "
                            "must be positive");
        return false;
    }
    if (min_new_tokens > max_new_tokens) {
        stream->reportError(ErrorCode::INVALID_PARAMS,
                            "min_new_tokens " + std::to_string(min_new_tokens) + " exceeds max_new_tokens "
                                + std::to_string(max_new_tokens));
        return false;
    }

    const auto limits       = sequenceLimitsForAdmission(stream, cache_manager);
    const auto input_length = static_cast<size_t>(input_length_value);
    if (input_length >= limits.effective_limit) {
        stream->reportError(
            ErrorCode::EXCEEDS_KV_CACHE_MAX_LEN,
            "input len " + std::to_string(input_length)
                + " leaves no generation room under effective sequence limit "
                + std::to_string(limits.effective_limit) + " (kv cache capacity "
                + std::to_string(limits.physical_capacity) + ", reserve_step "
                + std::to_string(limits.reserve_step) + ", logical limit " + std::to_string(limits.logical_limit)
                + ")");
        return false;
    }

    const size_t available_new_tokens = limits.effective_limit - input_length;
    if (static_cast<size_t>(min_new_tokens) > available_new_tokens) {
        stream->reportError(
            ErrorCode::EXCEEDS_KV_CACHE_MAX_LEN,
            "input len " + std::to_string(input_length) + " with min_new_tokens "
                + std::to_string(min_new_tokens) + " exceeds effective sequence limit "
                + std::to_string(limits.effective_limit) + " (kv cache capacity "
                + std::to_string(limits.physical_capacity) + ", reserve_step "
                + std::to_string(limits.reserve_step) + ", logical limit " + std::to_string(limits.logical_limit)
                + ")");
        return false;
    }

    if (static_cast<size_t>(max_new_tokens) > available_new_tokens) {
        max_new_tokens = static_cast<int>(available_new_tokens);
    }
    return true;
}

}  // namespace rtp_llm
