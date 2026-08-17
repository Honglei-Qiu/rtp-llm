#pragma once

#include "rtp_llm/cpp/api_server/Exception.h"
#include "rtp_llm/cpp/api_server/RequestCapabilityPredicate.h"
#include "rtp_llm/cpp/engine_base/EngineBase.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateConfig.h"

namespace rtp_llm {

inline void validateRequestCapabilities(const GenerateConfig& generate_config, EngineBase& engine) {
    if (speculationRejectsRequest(generate_config, engine)) {
        throw HttpApiServerException(HttpApiServerException::ERROR_INPUT_FORMAT_ERROR,
                                     "speculative decoding does not support probability or logit output");
    }
}

}  // namespace rtp_llm
