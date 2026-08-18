#pragma once

#include <algorithm>
#include <cstddef>
#include <memory>

#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"

namespace rtp_llm {

struct KVCacheSequenceLimits {
    // All capacities and limits are token counts. reserve_step is the executor
    // window width: one target token plus its speculative proposal slots.
    size_t physical_capacity     = 0;
    size_t reserve_step          = 0;
    size_t reserved_token_count  = 0;
    size_t physical_limit        = 0;
    size_t logical_limit         = 0;
    size_t effective_limit       = 0;
};

constexpr size_t kvCacheReservedTokenCount(size_t reserve_step) {
    return reserve_step > 0 ? reserve_step - 1 : 0;
}

constexpr KVCacheSequenceLimits calculateKVCacheSequenceLimits(size_t physical_capacity,
                                                               size_t reserve_step,
                                                               size_t logical_limit) {
    const size_t reserved_token_count = kvCacheReservedTokenCount(reserve_step);
    const size_t physical_limit        =
        physical_capacity > reserved_token_count ? physical_capacity - reserved_token_count : 0;
    // The window is subtracted from the physical capacity here and must not be subtracted from it
    // twice. The logical side is the caller's business: maxTokenNum() accounts for the window only
    // after the speculative output buffer is attached, so sequenceLimitsForAdmission() subtracts it
    // from logical_limit while that buffer is still absent.
    return {physical_capacity,
            reserve_step,
            reserved_token_count,
            physical_limit,
            logical_limit,
            std::min(logical_limit, physical_limit)};
}

KVCacheSequenceLimits sequenceLimitsForAdmission(const GenerateStreamPtr&             stream,
                                                 const std::shared_ptr<KVCacheManager>& cache_manager);

bool admitStreamToKVCache(const GenerateStreamPtr& stream, const std::shared_ptr<KVCacheManager>& cache_manager);

}  // namespace rtp_llm
