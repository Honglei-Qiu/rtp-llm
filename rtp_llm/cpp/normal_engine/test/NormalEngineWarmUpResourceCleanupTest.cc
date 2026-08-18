#include "rtp_llm/cpp/normal_engine/test/MockEngine.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"

#include <gtest/gtest.h>

namespace rtp_llm {
namespace {

class FailingWarmUpEngine final: public NormalEngine {
public:
    using NormalEngine::NormalEngine;

    absl::StatusOr<GenerateStreamPtr> preRun(const std::shared_ptr<GenerateInput>&, preRunMode) override {
        return absl::InternalError("injected warmup failure");
    }
};

class NormalEngineWarmUpResourceCleanupTest: public DeviceTestBase {
protected:
    void TearDown() override {
        NormalExecutor::test_model_factory = nullptr;
        DeviceTestBase::TearDown();
    }

    EngineInitParams makeParams() {
        CustomConfig  config;
        ModelConfig   model_config;
        RuntimeConfig runtime_config;
        KVCacheConfig kv_cache_config;
        auto params = createEngineInitParams(config, model_config, runtime_config, kv_cache_config);
        params.runtime_config.warm_up                                      = false;
        params.runtime_config.max_generate_batch_size                      = 1;
        params.runtime_config.fifo_scheduler_config.max_context_batch_size = 1;
        NormalExecutor::test_model_factory = [vocab_size = model_config.vocab_size](const GptModelInitParams&) {
            return std::make_unique<MockModel>(vocab_size);
        };
        return params;
    }

    std::unique_ptr<FailingWarmUpEngine> makeStoppedEngine(const EngineInitParams& params) {
        auto engine = std::make_unique<FailingWarmUpEngine>(params, nullptr);
        EXPECT_TRUE(engine->stop().ok());
        engine->executor_.reset();
        return engine;
    }

    void expectTraceCanBeReacquired() {
        EXPECT_FALSE(isTraceMemoryEnabled());
        const auto token = startMemoryTrace();
        EXPECT_TRUE(isTraceMemoryEnabled());
        stopMemoryTrace(token);
        EXPECT_FALSE(isTraceMemoryEnabled());
    }
};

TEST_F(NormalEngineWarmUpResourceCleanupTest, PrefillFailureReleasesExecutorAndTrace) {
    auto params = makeParams();
    auto engine = makeStoppedEngine(params);

    EXPECT_ANY_THROW(engine->prefillWarmUp(params));

    EXPECT_EQ(engine->executor_, nullptr);
    expectTraceCanBeReacquired();
}

TEST_F(NormalEngineWarmUpResourceCleanupTest, DecodeFailureReleasesExecutorCacheAndTrace) {
    auto params = makeParams();
    auto engine = makeStoppedEngine(params);

    EXPECT_ANY_THROW(engine->decodeWarmUp(params));

    EXPECT_EQ(engine->executor_, nullptr);
    expectTraceCanBeReacquired();
}

}  // namespace
}  // namespace rtp_llm
