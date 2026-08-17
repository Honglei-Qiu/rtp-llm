#include <gtest/gtest.h>

#include "rtp_llm/cpp/api_server/RequestCapabilityPredicate.h"
#include "rtp_llm/cpp/api_server/test/mock/MockEngineBase.h"

using namespace ::testing;

namespace rtp_llm {

// requestsProbabilityOutput is the field set that a speculative executor cannot produce. Each
// field is asserted independently so that dropping any one of them from the predicate (e.g. a
// revert to keying on return_all_probs alone) turns the corresponding case red.
TEST(RequestCapabilityPredicateTest, DefaultConfigRequestsNoProbabilityOutput) {
    GenerateConfig config;
    // return_target_logprob defaults to true and must NOT count as a probability request, or every
    // request (speculative or not) would be rejected.
    EXPECT_TRUE(config.return_target_logprob);
    EXPECT_FALSE(requestsProbabilityOutput(config));
}

TEST(RequestCapabilityPredicateTest, ReturnTargetLogprobAloneIsNotProbabilityOutput) {
    GenerateConfig config;
    config.return_target_logprob = true;
    // Nothing else set.
    EXPECT_FALSE(requestsProbabilityOutput(config));
}

TEST(RequestCapabilityPredicateTest, EachProbabilityFieldIsDetectedIndependently) {
    {
        GenerateConfig config;
        config.return_all_probs = ReturnAllProbsMode::DEFAULT;
        EXPECT_TRUE(requestsProbabilityOutput(config));
    }
    {
        GenerateConfig config;
        config.return_all_probs = ReturnAllProbsMode::ORIGINAL;
        EXPECT_TRUE(requestsProbabilityOutput(config));
    }
    {
        GenerateConfig config;
        config.return_softmax_probs = true;
        EXPECT_TRUE(requestsProbabilityOutput(config));
    }
    {
        GenerateConfig config;
        config.return_logits = true;
        EXPECT_TRUE(requestsProbabilityOutput(config));
    }
    {
        GenerateConfig config;
        config.return_prompt_logits = true;
        EXPECT_TRUE(requestsProbabilityOutput(config));
    }
    {
        GenerateConfig config;
        config.return_cum_log_probs = true;
        EXPECT_TRUE(requestsProbabilityOutput(config));
    }
    {
        GenerateConfig config;
        config.calculate_loss = 1;
        EXPECT_TRUE(requestsProbabilityOutput(config));
    }
}

TEST(RequestCapabilityPredicateTest, ReturnAllProbsNoneIsNotProbabilityOutput) {
    GenerateConfig config;
    config.return_all_probs = ReturnAllProbsMode::NONE;
    EXPECT_FALSE(requestsProbabilityOutput(config));
}

// speculationRejectsRequest is the AND of "wants probability output" and "engine runs a
// speculative executor". Both operands must be true; neither alone rejects.
TEST(RequestCapabilityPredicateTest, SpeculationRejectsOnlyWhenProbabilityAndSpeculative) {
    MockEngineBase engine;

    GenerateConfig probability_request;
    probability_request.return_logits = true;
    GenerateConfig plain_request;  // no probability output requested

    EXPECT_CALL(engine, hasSpeculativeExecutor()).WillRepeatedly(Return(true));
    EXPECT_TRUE(speculationRejectsRequest(probability_request, engine));
    EXPECT_FALSE(speculationRejectsRequest(plain_request, engine));
}

TEST(RequestCapabilityPredicateTest, SpeculationDoesNotRejectWhenNoSpeculativeExecutor) {
    MockEngineBase engine;

    GenerateConfig probability_request;
    probability_request.return_logits = true;

    EXPECT_CALL(engine, hasSpeculativeExecutor()).WillRepeatedly(Return(false));
    EXPECT_FALSE(speculationRejectsRequest(probability_request, engine));
}

}  // namespace rtp_llm
