#include "rtp_llm/cpp/testing/TestBase.h"
#include <array>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <tuple>

#define private public
#include "rtp_llm/cpp/engine_base/stream/GenerateTypes.h"
#include "rtp_llm/cpp/model_rpc/LocalRpcServer.h"
#include "rtp_llm/cpp/model_rpc/QueryConverter.h"
#include "rtp_llm/cpp/model_rpc/RpcErrorCode.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.grpc.pb.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.pb.h"
#include "rtp_llm/cpp/models/logits_processor/LogitsProcessorFactory.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"

using namespace std;
namespace rtp_llm {

class QueryConverterTest: public DeviceTestBase {};

namespace {

// All four shape/payload guards throw std::invalid_argument, so a type-only expectation
// lets them substitute for each other: deleting the negative-dimension guard still throws,
// from the int64 guard instead. Matching the message is what makes each guard's test die
// when that specific guard is removed.
::testing::AssertionResult throwsInvalidArgumentContaining(const std::function<void()>& call,
                                                           const std::string&           needle) {
    try {
        call();
    } catch (const std::invalid_argument& error) {
        const std::string message = error.what();
        if (message.find(needle) != std::string::npos) {
            return ::testing::AssertionSuccess();
        }
        return ::testing::AssertionFailure() << "message \"" << message << "\" lacks \"" << needle << "\"";
    } catch (const std::exception& error) {
        return ::testing::AssertionFailure() << "threw a different type: " << error.what();
    }
    return ::testing::AssertionFailure() << "did not throw";
}

}  // namespace

TEST_F(QueryConverterTest, testTransInput) {
    GenerateInputPB input;
    input.mutable_request_info()->set_frontend_ip("10.0.0.1");
    input.mutable_request_info()->set_dash_ip("10.0.0.2");
    input.mutable_request_info()->set_trace_id("trace-123");
    input.mutable_request_info()->set_request_id("source-request-123");
    input.mutable_request_info()->set_source_role("frontend");
    input.add_token_ids(0);
    input.add_token_ids(1);

    auto generate_config_pb = input.mutable_generate_config();
    generate_config_pb->set_min_new_tokens(4);
    generate_config_pb->set_max_new_tokens(5);
    generate_config_pb->set_num_beams(1);
    generate_config_pb->set_num_return_sequences(1);
    generate_config_pb->set_top_k(6);
    generate_config_pb->set_top_p(0.6);
    generate_config_pb->set_temperature(0.1);
    generate_config_pb->set_repetition_penalty(0.2);
    generate_config_pb->mutable_top_p_decay()->set_value(0.7);
    generate_config_pb->mutable_top_p_min()->set_value(0.3);
    generate_config_pb->mutable_top_p_reset_ids()->set_value(7);
    generate_config_pb->mutable_json_schema()->set_value("{\"type\":\"object\"}");
    generate_config_pb->mutable_regex()->set_value("[a-z]+");
    generate_config_pb->mutable_ebnf()->set_value("root ::= \"a\"");
    generate_config_pb->mutable_structural_tag()->set_value("{\"format\":{\"type\":\"json_schema\"}}");
    generate_config_pb->mutable_task_id()->set_value("8");
    generate_config_pb->set_calculate_loss(1);
    generate_config_pb->set_return_hidden_states(true);
    generate_config_pb->set_thinking_mode(GenerateConfigPB::THINKING_MODE_ADAPTIVE);
    for (int i = 0; i < 2; ++i) {
        auto* stop_words = generate_config_pb->mutable_stop_words_list()->add_rows();
        for (int j = 0; j < 3; ++j) {
            stop_words->add_values(i * 3 + j);
        }
    }
    auto  generate_input = QueryConverter::transQuery(&input);
    auto& input_ids      = generate_input->input_ids;
    ASSERT_EQ(input_ids.numel(), 2);
    ASSERT_EQ(input_ids.data_ptr<int32_t>()[0], 0);
    ASSERT_EQ(generate_input->request_info.frontend_ip, "10.0.0.1");
    ASSERT_EQ(generate_input->request_info.dash_ip, "10.0.0.2");
    ASSERT_EQ(generate_input->request_info.trace_id, "trace-123");
    ASSERT_EQ(generate_input->request_info.request_id, "source-request-123");
    ASSERT_EQ(generate_input->request_info.source_role, "frontend");
    auto generate_config = generate_input->generate_config;
    ASSERT_EQ(generate_config->min_new_tokens, 4);
    ASSERT_EQ(generate_config->max_new_tokens, 5);
    ASSERT_EQ(generate_config->num_beams, 1);
    ASSERT_EQ(generate_config->num_return_sequences, 1);
    ASSERT_EQ(generate_config->top_k, 6);
    ASSERT_FLOAT_EQ(generate_config->top_p, 0.6);
    ASSERT_FLOAT_EQ(generate_config->temperature, 0.1);
    ASSERT_FLOAT_EQ(generate_config->repetition_penalty, 0.2);
    ASSERT_FLOAT_EQ(generate_config->top_p_decay.value(), 0.7);
    ASSERT_FLOAT_EQ(generate_config->top_p_min.value(), 0.3);
    ASSERT_EQ(generate_config->top_p_reset_ids.value(), 7);
    ASSERT_EQ(generate_config->json_schema.value(), "{\"type\":\"object\"}");
    ASSERT_EQ(generate_config->regex.value(), "[a-z]+");
    ASSERT_EQ(generate_config->ebnf.value(), "root ::= \"a\"");
    ASSERT_EQ(generate_config->structural_tag.value(), "{\"format\":{\"type\":\"json_schema\"}}");
    ASSERT_EQ(generate_config->task_id.value(), "8");
    ASSERT_EQ(generate_config->calculate_loss, 1);
    ASSERT_TRUE(generate_config->return_hidden_states);
    ASSERT_FALSE(generate_config->return_logits);
    ASSERT_EQ(generate_config->thinking_mode, ThinkingMode::ADAPTIVE);
    ASSERT_EQ(generate_config->stop_words_list.size(), 2);
    vector<int> stop_words_1{0, 1, 2};
    vector<int> stop_words_2{3, 4, 5};
    ASSERT_EQ(generate_config->stop_words_list[0], stop_words_1);
    ASSERT_EQ(generate_config->stop_words_list[1], stop_words_2);
}

TEST_F(QueryConverterTest, TransGenerateConfigResolvesThinkingState) {
    using Case = std::tuple<GenerateConfigPB::ThinkingModePB, bool, ThinkingMode, bool>;
    const std::array<Case, 5> cases{{
        {GenerateConfigPB::THINKING_MODE_UNSPECIFIED, false, ThinkingMode::UNSPECIFIED, false},
        {GenerateConfigPB::THINKING_MODE_UNSPECIFIED, true, ThinkingMode::UNSPECIFIED, true},
        {GenerateConfigPB::THINKING_MODE_DISABLED, true, ThinkingMode::DISABLED, false},
        {GenerateConfigPB::THINKING_MODE_ADAPTIVE, true, ThinkingMode::ADAPTIVE, false},
        {GenerateConfigPB::THINKING_MODE_ENABLED, false, ThinkingMode::ENABLED, true},
    }};

    for (const auto& [proto_mode, legacy_in_think_mode, expected_mode, expected_in_think_mode] : cases) {
        GenerateConfigPB config_pb;
        config_pb.set_thinking_mode(proto_mode);
        config_pb.set_in_think_mode(legacy_in_think_mode);

        const auto config = QueryConverter::transGenerateConfig(&config_pb);

        EXPECT_EQ(config->thinking_mode, expected_mode);
        EXPECT_EQ(config->in_think_mode, expected_in_think_mode);
    }
}

TEST(ThinkingModeTest, NormalizeThinkingModeRejectsOutOfRangeValues) {
    EXPECT_EQ(normalizeThinkingMode(-1), ThinkingMode::UNSPECIFIED);
    EXPECT_EQ(normalizeThinkingMode(4), ThinkingMode::UNSPECIFIED);
}

TEST(ThinkingModeTest, NormalizeThinkingModePreservesValidValues) {
    EXPECT_EQ(normalizeThinkingMode(0), ThinkingMode::UNSPECIFIED);
    EXPECT_EQ(normalizeThinkingMode(1), ThinkingMode::DISABLED);
    EXPECT_EQ(normalizeThinkingMode(2), ThinkingMode::ADAPTIVE);
    EXPECT_EQ(normalizeThinkingMode(3), ThinkingMode::ENABLED);
}

TEST_F(QueryConverterTest, RoleAddrReadsLegacyTypedAndDualWritePayloads) {
    GenerateConfigPB config;

    auto* legacy = config.add_role_addrs();
    legacy->set_role(RoleAddrPB::PREFILL);
    legacy->set_ip("legacy-prefill");

    auto* string_only = config.add_role_addrs();
    string_only->set_role_str("DECODE");
    string_only->set_ip("string-decode");

    auto* dual = config.add_role_addrs();
    dual->set_role(RoleAddrPB::VIT);
    dual->set_role_str("VIT");
    dual->set_ip("dual-vit");

    const auto role_addrs = QueryConverter::getRoleAddrs(&config);
    ASSERT_EQ(role_addrs.size(), 3);
    EXPECT_EQ(role_addrs[0].role, RoleType::PREFILL);
    EXPECT_EQ(role_addrs[1].role, RoleType::DECODE);
    EXPECT_EQ(role_addrs[2].role, RoleType::VIT);
}

TEST_F(QueryConverterTest, RoleAddrPreservesPdfusionDefaultAndRejectsConflicts) {
    GenerateConfigPB legacy_pdfusion;
    legacy_pdfusion.add_role_addrs()->set_role(RoleAddrPB::PDFUSION);
    EXPECT_EQ(QueryConverter::getRoleAddrs(&legacy_pdfusion)[0].role, RoleType::PDFUSION);

    GenerateConfigPB conflict;
    auto*            conflicting = conflict.add_role_addrs();
    conflicting->set_role(RoleAddrPB::PREFILL);
    conflicting->set_role_str("DECODE");
    EXPECT_THROW(QueryConverter::getRoleAddrs(&conflict), std::runtime_error);

    GenerateConfigPB omitted_legacy_default;
    omitted_legacy_default.add_role_addrs();
    EXPECT_EQ(QueryConverter::getRoleAddrs(&omitted_legacy_default)[0].role, RoleType::PDFUSION);
}

TEST_F(QueryConverterTest, testTransOutput) {
    auto output_token_ids = torch::empty({1, 3}, torch::kInt32);
    auto data             = output_token_ids.data_ptr<int>();
    for (int i = 0; i < 3; ++i) {
        data[i] = i;
    }
    GenerateOutputs outputs;
    GenerateOutput  res;
    res.output_ids            = output_token_ids;
    res.finished              = true;
    res.aux_info.cost_time_us = 1000;
    res.aux_info.iter_count   = 9;
    res.aux_info.input_len    = 8;
    res.aux_info.output_len   = 7;
    auto hidden_states_tensor = torch::empty({3, 2}, torch::kFloat32);
    auto hidden_states_data   = hidden_states_tensor.data_ptr<float>();
    for (int i = 0; i < 6; ++i) {
        hidden_states_data[i] = i;
    }
    res.hidden_states.emplace(hidden_states_tensor);
    outputs.generate_outputs.push_back(res);

    GenerateOutputsPB outputs_pb;
    QueryConverter::transResponse(&outputs_pb, &outputs, true, "", 10000);

    auto& output_pb   = outputs_pb.flatten_output();
    auto  aux_info_pb = output_pb.aux_info(0);
    EXPECT_EQ(aux_info_pb.cost_time_us(), 1000);
    EXPECT_EQ(aux_info_pb.iter_count(), 9);
    EXPECT_EQ(aux_info_pb.input_len(), 8);
    EXPECT_EQ(aux_info_pb.output_len(), 7);
    auto output_ids_pb = output_pb.output_ids();
    ASSERT_EQ(output_ids_pb.data_type(), TensorPB_DataType::TensorPB_DataType_INT32);
    ASSERT_EQ(output_ids_pb.shape_size(), 3);
    ASSERT_EQ(output_ids_pb.shape(0), 1);
    ASSERT_EQ(output_ids_pb.shape(1), 1);
    ASSERT_EQ(output_ids_pb.shape(2), 3);
    auto            output_ids_string = output_ids_pb.int32_data();
    vector<int32_t> output_ids_vector;
    output_ids_vector.resize(output_ids_string.size() / sizeof(int32_t));
    std::memcpy(output_ids_vector.data(), output_ids_string.data(), output_ids_string.size());
    for (int i = 0; i < 3; ++i) {
        ASSERT_EQ(output_ids_vector[i], i);
    }
    ASSERT_TRUE(output_pb.has_hidden_states());
    auto hidden_states_pb = output_pb.hidden_states();
    ASSERT_EQ(hidden_states_pb.data_type(), TensorPB_DataType::TensorPB_DataType_FP32);
    ASSERT_EQ(hidden_states_pb.shape_size(), 3);
    ASSERT_EQ(hidden_states_pb.shape(0), 1);
    ASSERT_EQ(hidden_states_pb.shape(1), 3);
    ASSERT_EQ(hidden_states_pb.shape(2), 2);
    auto          hidden_states_string = hidden_states_pb.fp32_data();
    vector<float> hidden_states_vector;
    hidden_states_vector.resize(hidden_states_string.size() / sizeof(float));
    std::memcpy(hidden_states_vector.data(), hidden_states_string.data(), hidden_states_string.size());
    for (int i = 0; i < 6; ++i) {
        ASSERT_FLOAT_EQ(hidden_states_vector[i], i);
    }
}

TEST_F(QueryConverterTest, TransTensorPB_FP32) {

    torch::Tensor tensor = torch::rand({2, 3}, torch::kFloat32);
    TensorPB      tensor_pb;
    QueryConverter::transTensorPB(&tensor_pb, tensor);
    EXPECT_EQ(tensor_pb.data_type(), TensorPB::FP32);
    ASSERT_EQ(tensor_pb.shape_size(), 2);
    EXPECT_EQ(tensor_pb.shape(0), 2);
    EXPECT_EQ(tensor_pb.shape(1), 3);

    // 验证数据一致性
    const std::string& proto_data        = tensor_pb.fp32_data();
    const float*       proto_ptr         = reinterpret_cast<const float*>(proto_data.data());
    torch::Tensor      contiguous_tensor = tensor.contiguous();
    const float*       tensor_ptr        = contiguous_tensor.data_ptr<float>();

    ASSERT_EQ(proto_data.size(), contiguous_tensor.numel() * sizeof(float));
    for (int i = 0; i < contiguous_tensor.numel(); ++i) {
        EXPECT_FLOAT_EQ(proto_ptr[i], tensor_ptr[i]);
    }
}

TEST_F(QueryConverterTest, TransTensorPB_BF16) {
    torch::Tensor tensor = torch::rand({3}, torch::kBFloat16);
    TensorPB      tensor_pb;
    QueryConverter::transTensorPB(&tensor_pb, tensor);

    EXPECT_EQ(tensor_pb.data_type(), TensorPB::BF16);

    const std::string& proto_data    = tensor_pb.bf16_data();
    size_t             expected_size = tensor.numel() * sizeof(c10::BFloat16);
    ASSERT_EQ(proto_data.size(), expected_size);

    const char* tensor_data = static_cast<const char*>(tensor.contiguous().data_ptr());
    EXPECT_EQ(std::memcmp(proto_data.data(), tensor_data, expected_size), 0);
}

TEST_F(QueryConverterTest, TransTensorPB_ScalarShape) {
    torch::Tensor tensor = torch::tensor(42, torch::kInt32);
    TensorPB      tensor_pb;
    QueryConverter::transTensorPB(&tensor_pb, tensor);
    EXPECT_EQ(tensor_pb.shape_size(), 0);
}

TEST_F(QueryConverterTest, TransTensorPB_NonContiguous) {
    torch::Tensor tensor = torch::rand({3, 4}, torch::kFloat32).transpose(0, 1);
    TensorPB      tensor_pb;
    QueryConverter::transTensorPB(&tensor_pb, tensor);

    torch::Tensor      contiguous_tensor = tensor.contiguous();
    const std::string& proto_data        = tensor_pb.fp32_data();
    const float*       proto_ptr         = reinterpret_cast<const float*>(proto_data.data());
    const float*       tensor_ptr        = contiguous_tensor.data_ptr<float>();

    for (int i = 0; i < contiguous_tensor.numel(); ++i) {
        EXPECT_FLOAT_EQ(proto_ptr[i], tensor_ptr[i]);
    }
}

TEST_F(QueryConverterTest, TransTensorPB_UnsupportedType) {
    torch::Tensor tensor = torch::ones({1}, torch::kInt64);
    TensorPB      tensor_pb;
    tensor_pb.add_shape(7);
    tensor_pb.set_fp32_data(std::string(sizeof(float), '\0'));

    EXPECT_THROW(QueryConverter::transTensorPB(&tensor_pb, tensor), std::runtime_error);
    ASSERT_EQ(tensor_pb.shape_size(), 1);
    EXPECT_EQ(tensor_pb.shape(0), 7);
    EXPECT_EQ(tensor_pb.fp32_data().size(), sizeof(float));
}

TEST_F(QueryConverterTest, TransTensorRejectsOversizedPayload) {
    TensorPB tensor_pb;
    tensor_pb.set_data_type(TensorPB::FP32);
    tensor_pb.add_shape(1);
    tensor_pb.set_fp32_data(std::string(2 * sizeof(float), '\0'));

    EXPECT_THROW(QueryConverter::transTensor(tensor_pb), std::invalid_argument);
}

TEST_F(QueryConverterTest, TransTensorRejectsTruncatedAndUnexpectedPayloads) {
    TensorPB tensor_pb;
    tensor_pb.set_data_type(TensorPB::FP32);
    tensor_pb.add_shape(2);
    tensor_pb.set_fp32_data(std::string(sizeof(float), '\0'));
    EXPECT_THROW(QueryConverter::transTensor(tensor_pb), std::invalid_argument);

    tensor_pb.set_fp32_data(std::string(2 * sizeof(float), '\0'));
    tensor_pb.set_int32_data(std::string(2 * sizeof(int32_t), '\0'));
    EXPECT_THROW(QueryConverter::transTensor(tensor_pb), std::invalid_argument);
}

TEST_F(QueryConverterTest, TransTensorRejectsInvalidShape) {
    TensorPB negative_pb;
    negative_pb.set_data_type(TensorPB::FP16);
    negative_pb.add_shape(-1);
    EXPECT_TRUE(throwsInvalidArgumentContaining([&] { QueryConverter::transTensor(negative_pb); },
                                                "negative dimension"));

    // (2^63-1) * 2 does not overflow size_t, so this reaches the int64 element limit rather
    // than the size_t overflow guard above it.
    TensorPB over_int64_pb;
    over_int64_pb.set_data_type(TensorPB::FP16);
    over_int64_pb.add_shape(std::numeric_limits<int64_t>::max());
    over_int64_pb.add_shape(2);
    EXPECT_TRUE(throwsInvalidArgumentContaining([&] { QueryConverter::transTensor(over_int64_pb); },
                                                "exceeds the tensor limit"));

    // 4 * 2^62 wraps size_t, so the running product must be rejected before it is computed.
    TensorPB element_overflow_pb;
    element_overflow_pb.set_data_type(TensorPB::FP16);
    element_overflow_pb.add_shape(4);
    element_overflow_pb.add_shape(int64_t{1} << 62);
    EXPECT_TRUE(throwsInvalidArgumentContaining([&] { QueryConverter::transTensor(element_overflow_pb); },
                                                "element count overflows size_t"));

    // 2^62 elements is under the int64 limit, but 2^62 * 4 bytes wraps size_t, so only the
    // byte-count guard catches this one.
    TensorPB byte_overflow_pb;
    byte_overflow_pb.set_data_type(TensorPB::FP32);
    byte_overflow_pb.add_shape(int64_t{1} << 62);
    EXPECT_TRUE(throwsInvalidArgumentContaining([&] { QueryConverter::transTensor(byte_overflow_pb); },
                                                "byte count overflows size_t"));
}

TEST_F(QueryConverterTest, TransTensorRejectsImplausibleRank) {
    // All-zero dimensions is the payload that slipped past every other guard: the element
    // count stays 0, so the payload-size comparison passes, yet the shape vector is still
    // built from the declared rank before any check ran.
    TensorPB at_limit_pb;
    at_limit_pb.set_data_type(TensorPB::FP32);
    for (int dimension = 0; dimension < 64; ++dimension) {
        at_limit_pb.add_shape(0);
    }
    const auto at_limit = QueryConverter::transTensor(at_limit_pb);
    EXPECT_EQ(at_limit.dim(), 64);
    EXPECT_EQ(at_limit.numel(), 0);

    TensorPB over_limit_pb;
    over_limit_pb.set_data_type(TensorPB::FP32);
    for (int dimension = 0; dimension < 65; ++dimension) {
        over_limit_pb.add_shape(0);
    }
    EXPECT_TRUE(throwsInvalidArgumentContaining([&] { QueryConverter::transTensor(over_limit_pb); },
                                                "rank exceeds 64"));
}

TEST_F(QueryConverterTest, TransTensorRoundTripsFp16Payload) {
    const auto tensor = torch::tensor({1.5, -2.25, 3.0}, torch::kFloat16);
    TensorPB   tensor_pb;
    QueryConverter::transTensorPB(&tensor_pb, tensor);

    EXPECT_EQ(tensor_pb.data_type(), TensorPB::FP16);
    EXPECT_EQ(tensor_pb.fp16_data().size(), 3 * sizeof(c10::Half));
    EXPECT_TRUE(tensor_pb.fp32_data().empty());

    const auto restored = QueryConverter::transTensor(tensor_pb);
    EXPECT_EQ(restored.scalar_type(), torch::kFloat16);
    EXPECT_EQ(restored.sizes(), torch::IntArrayRef({3}));
    EXPECT_TRUE(torch::equal(restored, tensor));

    // A payload sized for FP32 must not pass as three FP16 elements.
    tensor_pb.set_fp16_data(std::string(3 * sizeof(float), '\0'));
    EXPECT_TRUE(throwsInvalidArgumentContaining([&] { QueryConverter::transTensor(tensor_pb); },
                                                "does not match its shape and data type"));
}

TEST_F(QueryConverterTest, TransTensorAcceptsUnsetMessageAsEmptyTensor) {
    // rtp_llm/utils/grpc_util.py trans_from_tensor sends a bare TensorPB for a None or
    // empty tensor, and URL-based multimodal inputs always take that path.
    const auto empty = QueryConverter::transTensor(TensorPB());
    EXPECT_EQ(empty.numel(), 0);
    EXPECT_EQ(empty.sizes(), torch::IntArrayRef({0}));
    EXPECT_EQ(empty.scalar_type(), torch::kFloat32);
}

TEST_F(QueryConverterTest, TransTensorAcceptsScalarAndZeroSizedShape) {
    TensorPB scalar_pb;
    scalar_pb.set_data_type(TensorPB::INT32);
    const int32_t value = 42;
    scalar_pb.set_int32_data(&value, sizeof(value));

    const auto scalar = QueryConverter::transTensor(scalar_pb);
    EXPECT_EQ(scalar.dim(), 0);
    EXPECT_EQ(scalar.item<int32_t>(), value);

    TensorPB empty_pb;
    empty_pb.set_data_type(TensorPB::BF16);
    empty_pb.add_shape(2);
    empty_pb.add_shape(0);
    empty_pb.add_shape(3);

    const auto empty = QueryConverter::transTensor(empty_pb);
    EXPECT_EQ(empty.sizes(), torch::IntArrayRef({2, 0, 3}));
    EXPECT_EQ(empty.numel(), 0);
}

TEST_F(QueryConverterTest, TransTensorPBClearsReusedMessage) {
    TensorPB tensor_pb;
    tensor_pb.add_shape(9);
    tensor_pb.set_fp32_data(std::string(sizeof(float), '\0'));

    const auto tensor = torch::tensor({7, 8}, torch::kInt32);
    QueryConverter::transTensorPB(&tensor_pb, tensor);

    ASSERT_EQ(tensor_pb.shape_size(), 1);
    EXPECT_EQ(tensor_pb.shape(0), 2);
    EXPECT_TRUE(tensor_pb.fp32_data().empty());
    EXPECT_EQ(tensor_pb.int32_data().size(), 2 * sizeof(int32_t));
    EXPECT_TRUE(torch::equal(QueryConverter::transTensor(tensor_pb), tensor));
}

TEST_F(QueryConverterTest, TransTensorPBRoundTripsZeroSizedTensor) {
    const auto tensor = torch::empty({2, 0, 3}, torch::kBFloat16);
    TensorPB  tensor_pb;

    QueryConverter::transTensorPB(&tensor_pb, tensor);

    ASSERT_EQ(tensor_pb.shape_size(), 3);
    EXPECT_EQ(tensor_pb.shape(0), 2);
    EXPECT_EQ(tensor_pb.shape(1), 0);
    EXPECT_EQ(tensor_pb.shape(2), 3);
    EXPECT_TRUE(tensor_pb.bf16_data().empty());
    const auto restored = QueryConverter::transTensor(tensor_pb);
    EXPECT_EQ(restored.sizes(), tensor.sizes());
    EXPECT_EQ(restored.scalar_type(), tensor.scalar_type());
}

TEST_F(QueryConverterTest, TransTensorPBFailureLeavesReusedMessageUnchanged) {
    TensorPB tensor_pb;
    tensor_pb.set_data_type(TensorPB::INT32);
    tensor_pb.add_shape(1);
    const int32_t value = 17;
    tensor_pb.set_int32_data(&value, sizeof(value));
    const auto original = tensor_pb.SerializeAsString();

    // A sparse tensor passes the dtype switch and the sizes() loop, then contiguous() rejects
    // it — so the failure lands after the local message has been populated, which is exactly
    // the window the Swap is meant to make invisible. A Meta tensor cannot be used here:
    // data_ptr() checks has_storage() but not storage_initialized(), so it returns nullptr
    // without throwing and set_fp32_data(nullptr, 8) segfaults.
    const auto uncopyable = torch::zeros({2, 2}, torch::kFloat32).to_sparse();
    EXPECT_THROW(QueryConverter::transTensorPB(&tensor_pb, uncopyable), std::exception);
    EXPECT_EQ(tensor_pb.SerializeAsString(), original);
}

// Typed grammar fields wire as google.protobuf.StringValue → Optional<string>.
TEST_F(QueryConverterTest, GrammarTypedFieldsAreAccepted) {
    GenerateInputPB input;
    input.add_token_ids(0);
    input.mutable_generate_config()->mutable_json_schema()->set_value(R"({"type":"object"})");
    auto cfg = QueryConverter::transQuery(&input)->generate_config;
    ASSERT_TRUE(cfg->json_schema.has_value());
    EXPECT_EQ(cfg->json_schema.value(), R"({"type":"object"})");
}

TEST_F(QueryConverterTest, MultipleGrammarConstraintsAreRejectedByFactory) {
    using SetGrammarField = void (*)(GenerateConfigPB*);
    struct GrammarFieldCase {
        const char*     name;
        SetGrammarField set;
    };
    const std::array<GrammarFieldCase, 4> fields{{
        {"json_schema",
         [](GenerateConfigPB* config) { config->mutable_json_schema()->set_value(R"({"type":"object"})"); }},
        {"regex", [](GenerateConfigPB* config) { config->mutable_regex()->set_value("[a-z]+"); }},
        {"ebnf", [](GenerateConfigPB* config) { config->mutable_ebnf()->set_value("root ::= \"a\""); }},
        {"structural_tag",
         [](GenerateConfigPB* config) { config->mutable_structural_tag()->set_value(R"({"type":"structural_tag"})"); }},
    }};

    for (size_t first = 0; first < fields.size(); ++first) {
        for (size_t second = first + 1; second < fields.size(); ++second) {
            SCOPED_TRACE(std::string(fields[first].name) + "+" + fields[second].name);
            GenerateInputPB input;
            input.add_token_ids(0);
            auto* config = input.mutable_generate_config();
            fields[first].set(config);
            fields[second].set(config);

            auto generate_input = QueryConverter::transQuery(&input);
            auto result         = LogitsProcessorFactory::createLogitsProcessors(
                std::move(generate_input), /*init_batch_size=*/1, /*max_batch_size=*/1, /*eos_token_id=*/0);

            ASSERT_FALSE(result.ok());
            EXPECT_EQ(result.status().code(), ErrorCode::INVALID_PARAMS);
            EXPECT_NE(result.status().ToString().find(fields[first].name), std::string::npos);
            EXPECT_NE(result.status().ToString().find(fields[second].name), std::string::npos);
        }
    }
}

TEST_F(QueryConverterTest, GrammarWithMultipleSequencesIsRejectedByFactory) {
    struct MultiSequenceCase {
        const char* name;
        void (*configure)(GenerateConfigPB*);
    };
    const std::array<MultiSequenceCase, 3> cases{{
        {"num_beams", [](GenerateConfigPB* config) { config->set_num_beams(2); }},
        {"variable_num_beams", [](GenerateConfigPB* config) { config->add_variable_num_beams(2); }},
        {"num_return_sequences", [](GenerateConfigPB* config) { config->set_num_return_sequences(2); }},
    }};

    for (const auto& test_case : cases) {
        SCOPED_TRACE(test_case.name);
        GenerateInputPB input;
        input.add_token_ids(0);
        auto* config = input.mutable_generate_config();
        config->mutable_regex()->set_value("[a-z]+");
        test_case.configure(config);

        auto generate_input = QueryConverter::transQuery(&input);
        auto result         = LogitsProcessorFactory::createLogitsProcessors(
            std::move(generate_input), /*init_batch_size=*/1, /*max_batch_size=*/2, /*eos_token_id=*/0);

        ASSERT_FALSE(result.ok());
        EXPECT_EQ(result.status().code(), ErrorCode::INVALID_PARAMS);
        EXPECT_NE(result.status().ToString().find("does not support beam search or num_return_sequences > 1"),
                  std::string::npos);
    }
}

TEST_F(QueryConverterTest, TimeoutErrorCodeMapsToGrpcDeadline) {
    EXPECT_EQ(transErrorCodeToGrpc(ErrorCode::GENERATE_TIMEOUT), grpc::StatusCode::DEADLINE_EXCEEDED);
    EXPECT_EQ(transErrorCodeToGrpc(ErrorCode::DEADLINE_EXCEEDED), grpc::StatusCode::DEADLINE_EXCEEDED);
    EXPECT_EQ(transErrorCodeToGrpc(ErrorCode::WAIT_TO_RUN_TIMEOUT), grpc::StatusCode::DEADLINE_EXCEEDED);
    EXPECT_EQ(transErrorCodeToGrpc(ErrorCode::KEEP_ALIVE_TIMEOUT), grpc::StatusCode::DEADLINE_EXCEEDED);
}

}  // namespace rtp_llm
