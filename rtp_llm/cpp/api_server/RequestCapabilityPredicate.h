#pragma once

#include "rtp_llm/cpp/engine_base/EngineBase.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateConfig.h"

namespace rtp_llm {

// Pure predicates, split out of RequestCapabilityValidator.h so the model-RPC entry points can
// reuse them without depending on the HTTP exception type. Every field here asks the engine for
// per-token probability or logit output, which a speculative executor cannot produce — the
// probs/logits tensors are default-constructed on the MTP dispatch path, so under speculation
// these are either silently wrong or silently dropped. return_target_logprob is excluded on
// purpose: it defaults to true, so including it would reject every speculative request.
inline bool requestsProbabilityOutput(const GenerateConfig& generate_config) {
    return generate_config.return_all_probs != ReturnAllProbsMode::NONE
           || generate_config.return_softmax_probs || generate_config.return_logits
           || generate_config.return_prompt_logits || generate_config.return_cum_log_probs
           || generate_config.calculate_loss != 0;
}

// Broader than isMTPEagle: the executor selection also admits eagle3/vanilla/deterministic, all
// of which hit the same undefined-probability path.
inline bool speculationRejectsRequest(const GenerateConfig& generate_config, EngineBase& engine) {
    return requestsProbabilityOutput(generate_config) && engine.hasSpeculativeExecutor();
}

}  // namespace rtp_llm
