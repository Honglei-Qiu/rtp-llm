#include "rtp_llm/models_py/bindings/core/ExecOps.h"

#include <exception>
#include <gtest/gtest.h>
#include <thread>

#if USING_CUDA
#include <c10/cuda/CUDACachingAllocator.h>
#endif

TEST(ExecOpsMemoryTraceTest, ReportsPeakAndResets) {
#if USING_CUDA
    EXPECT_NO_THROW(rtp_llm::getGpuExecStatus());

    auto baseline_allocation = torch::empty({1}, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
    rtp_llm::runtimeSyncAndCheck();
    c10::cuda::CUDACachingAllocator::emptyCache();

    const auto first_trace = rtp_llm::startMemoryTrace();
    EXPECT_TRUE(rtp_llm::isTraceMemoryEnabled());
    auto allocation = torch::empty(
        {32 * 1024 * 1024}, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
    rtp_llm::runtimeSyncAndCheck();

    EXPECT_GE(rtp_llm::getGpuExecStatus().device_memory_status.max_consumed_bytes, allocation.nbytes());

    EXPECT_THROW(rtp_llm::startMemoryTrace(), std::exception);
    EXPECT_GE(rtp_llm::getGpuExecStatus().device_memory_status.max_consumed_bytes, allocation.nbytes());

    EXPECT_THROW(rtp_llm::stopMemoryTrace(first_trace + 1), std::exception);
    EXPECT_TRUE(rtp_llm::isTraceMemoryEnabled());
    rtp_llm::stopMemoryTrace(first_trace);
    EXPECT_FALSE(rtp_llm::isTraceMemoryEnabled());
    EXPECT_EQ(rtp_llm::getGpuExecStatus().device_memory_status.max_consumed_bytes, 0u);

    allocation = torch::Tensor();
    rtp_llm::runtimeSyncAndCheck();

    const auto second_trace = rtp_llm::startMemoryTrace();
    allocation = torch::empty(
        {32 * 1024 * 1024}, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
    rtp_llm::runtimeSyncAndCheck();
    EXPECT_GE(rtp_llm::getGpuExecStatus().device_memory_status.max_consumed_bytes, allocation.nbytes());
    EXPECT_THROW(rtp_llm::stopMemoryTrace(first_trace), std::exception);
    EXPECT_TRUE(rtp_llm::isTraceMemoryEnabled());
    rtp_llm::stopMemoryTrace(second_trace);
    EXPECT_FALSE(rtp_llm::isTraceMemoryEnabled());

    allocation = torch::Tensor();
    rtp_llm::runtimeSyncAndCheck();
    c10::cuda::CUDACachingAllocator::emptyCache();

    const auto third_trace = rtp_llm::startMemoryTrace();
    EXPECT_EQ(rtp_llm::getGpuExecStatus().device_memory_status.max_consumed_bytes, 0u);
    rtp_llm::stopMemoryTrace(third_trace);
    EXPECT_FALSE(rtp_llm::isTraceMemoryEnabled());

    const auto explicit_trace = rtp_llm::startMemoryTrace();
    rtp_llm::setTraceMemory(false);
    EXPECT_TRUE(rtp_llm::isTraceMemoryEnabled());
    EXPECT_THROW(rtp_llm::setTraceMemory(true), std::exception);
    EXPECT_TRUE(rtp_llm::isTraceMemoryEnabled());
    rtp_llm::stopMemoryTrace(explicit_trace);

    std::thread legacy_owner([]() { rtp_llm::setTraceMemory(true); });
    legacy_owner.join();
    rtp_llm::setTraceMemory(true);
    EXPECT_TRUE(rtp_llm::isTraceMemoryEnabled());
    std::thread legacy_releaser([]() { rtp_llm::setTraceMemory(false); });
    legacy_releaser.join();
    rtp_llm::setTraceMemory(false);
    EXPECT_FALSE(rtp_llm::isTraceMemoryEnabled());

    baseline_allocation = torch::Tensor();
#else
    GTEST_SKIP() << "memory peak tracing is only used by the CUDA warmup path";
#endif
}
