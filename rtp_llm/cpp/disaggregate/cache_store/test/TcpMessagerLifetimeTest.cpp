#include "gtest/gtest.h"
#include "rtp_llm/cpp/disaggregate/cache_store/RequestBlockBufferStore.h"
#include "rtp_llm/cpp/disaggregate/cache_store/TcpMessager.h"

namespace rtp_llm {

TEST(TcpMessagerLifetimeTest, testDestructionKeepsRegisteredServiceAliveUntilServerShutdownCompletes) {
    std::shared_ptr<MemoryUtil> memory_util;
    auto                        buffer_store = std::make_shared<RequestBlockBufferStore>(memory_util);
    auto                        messager     = std::make_shared<TcpMessager>(memory_util, buffer_store, nullptr);
    auto                        service      = std::make_shared<TcpCacheStoreServiceImpl>(memory_util,
                                                                   buffer_store,
                                                                   nullptr,
                                                                   messager->timer_manager_,
                                                                   messager->locked_block_buffer_manager_,
                                                                   nullptr,
                                                                   -1);

    messager->service_ = service;
    std::weak_ptr<TcpCacheStoreServiceImpl> weak_service = service;
    service.reset();

    bool service_alive_after_server_shutdown = false;
    messager->tcp_server_ = std::shared_ptr<TcpServer>(new TcpServer(), [&](TcpServer* server) {
        delete server;
        service_alive_after_server_shutdown = !weak_service.expired();
    });

    messager.reset();

    EXPECT_TRUE(service_alive_after_server_shutdown);
    EXPECT_TRUE(weak_service.expired());
}

}  // namespace rtp_llm
