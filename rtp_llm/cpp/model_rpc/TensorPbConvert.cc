#include "rtp_llm/cpp/model_rpc/TensorPbConvert.h"

#include <cstring>
#include <limits>
#include <stdexcept>

namespace rtp_llm {

namespace {

struct TensorPayload {
    c10::ScalarType    dtype;
    const std::string* bytes;
    size_t             element_size;
};

TensorPayload selectPayload(const TensorPB& tensor_pb) {
    const auto reject_unexpected_payload = [](const std::string& payload, const char* field_name) {
        if (!payload.empty()) {
            throw std::invalid_argument(std::string("TensorPB contains unexpected payload field: ") + field_name);
        }
    };

    switch (tensor_pb.data_type()) {
        case TensorPB::FP32:
            reject_unexpected_payload(tensor_pb.int32_data(), "int32_data");
            reject_unexpected_payload(tensor_pb.fp16_data(), "fp16_data");
            reject_unexpected_payload(tensor_pb.bf16_data(), "bf16_data");
            return {torch::kFloat32, &tensor_pb.fp32_data(), sizeof(float)};
        case TensorPB::INT32:
            reject_unexpected_payload(tensor_pb.fp32_data(), "fp32_data");
            reject_unexpected_payload(tensor_pb.fp16_data(), "fp16_data");
            reject_unexpected_payload(tensor_pb.bf16_data(), "bf16_data");
            return {torch::kInt32, &tensor_pb.int32_data(), sizeof(int32_t)};
        case TensorPB::FP16:
            reject_unexpected_payload(tensor_pb.fp32_data(), "fp32_data");
            reject_unexpected_payload(tensor_pb.int32_data(), "int32_data");
            reject_unexpected_payload(tensor_pb.bf16_data(), "bf16_data");
            return {torch::kFloat16, &tensor_pb.fp16_data(), sizeof(c10::Half)};
        case TensorPB::BF16:
            reject_unexpected_payload(tensor_pb.fp32_data(), "fp32_data");
            reject_unexpected_payload(tensor_pb.int32_data(), "int32_data");
            reject_unexpected_payload(tensor_pb.fp16_data(), "fp16_data");
            return {torch::kBFloat16, &tensor_pb.bf16_data(), sizeof(c10::BFloat16)};
        default:
            throw std::invalid_argument("TensorPB has an unsupported data type");
    }
}

size_t checkedPayloadSize(const TensorPB& tensor_pb, size_t element_size) {
    size_t element_count = 1;
    for (const int64_t dimension : tensor_pb.shape()) {
        if (dimension < 0) {
            throw std::invalid_argument("TensorPB shape contains a negative dimension");
        }
        const size_t unsigned_dimension = static_cast<size_t>(dimension);
        if (unsigned_dimension != 0 && element_count > std::numeric_limits<size_t>::max() / unsigned_dimension) {
            throw std::invalid_argument("TensorPB shape element count overflows size_t");
        }
        element_count *= unsigned_dimension;
        if (element_count > static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
            throw std::invalid_argument("TensorPB shape element count exceeds the tensor limit");
        }
    }
    if (element_count > std::numeric_limits<size_t>::max() / element_size) {
        throw std::invalid_argument("TensorPB payload byte count overflows size_t");
    }
    return element_count * element_size;
}

}  // namespace

torch::Tensor TensorPbConvert::pbToTorch(const TensorPB& tensor_pb) {
    // Bounded before the vector is built from it: nothing caps the number of shape entries,
    // and a message of nothing but zero dimensions satisfies every later check (element count
    // stays zero, so the payload comparison passes) while still forcing an allocation
    // proportional to the declared rank. The P2P path configures no gRPC receive limit at all,
    // so that is a large multiple of the message size. ATen enforces no rank limit of its own
    // (torch::empty accepts rank 128), which is why it has to be done here; 64 is a policy cap
    // set far above the handful of dimensions any activation on this wire actually has.
    constexpr int kMaxRank = 64;
    if (tensor_pb.shape_size() > kMaxRank) {
        throw std::invalid_argument("TensorPB rank exceeds " + std::to_string(kMaxRank));
    }
    std::vector<int64_t> shape(tensor_pb.shape().begin(), tensor_pb.shape().end());
    const auto           payload       = selectPayload(tensor_pb);
    const size_t         expected_size = checkedPayloadSize(tensor_pb, payload.element_size);
    // No shape and no payload is how both sides encode "no tensor": trans_from_tensor in
    // rtp_llm/utils/grpc_util.py emits a bare TensorPB for a None or empty tensor and
    // trans_tensor decodes it as a 1-D empty tensor. A rank-0 scalar carries element_size
    // bytes, so it is still handled by the size check below.
    if (tensor_pb.shape().empty() && payload.bytes->empty()) {
        return torch::empty({0}, torch::TensorOptions().dtype(payload.dtype));
    }
    if (payload.bytes->size() != expected_size) {
        throw std::invalid_argument("TensorPB payload size does not match its shape and data type");
    }

    auto tensor = torch::empty(shape, torch::TensorOptions().dtype(payload.dtype));
    if (expected_size != 0) {
        std::memcpy(tensor.data_ptr(), payload.bytes->data(), expected_size);
    }
    return tensor;
}

void TensorPbConvert::torchToPb(TensorPB* tensor_pb, const torch::Tensor& tensor) {
    if (tensor_pb == nullptr) {
        throw std::invalid_argument("TensorPB output must not be null");
    }
    TensorPB::DataType data_type;
    switch (tensor.dtype().toScalarType()) {
        case torch::kFloat32:
            data_type = TensorPB::FP32;
            break;
        case torch::kInt32:
            data_type = TensorPB::INT32;
            break;
        case torch::kFloat16:
            data_type = TensorPB::FP16;
            break;
        case torch::kBFloat16:
            data_type = TensorPB::BF16;
            break;
        default:
            throw std::runtime_error("Unsupported tensor data type.");
    }
    TensorPB converted;
    converted.set_data_type(data_type);
    auto shape = tensor.sizes();
    for (auto dim : shape) {
        converted.add_shape(dim);
    }
    torch::Tensor contiguous_tensor = tensor.contiguous();
    switch (tensor.dtype().toScalarType()) {
        case torch::kFloat32: {
            size_t      num_bytes = contiguous_tensor.numel() * sizeof(float);
            const char* data_ptr  = static_cast<const char*>(contiguous_tensor.data_ptr());
            converted.set_fp32_data(data_ptr, num_bytes);
            break;
        }
        case torch::kInt32: {
            size_t      num_bytes = contiguous_tensor.numel() * sizeof(int32_t);
            const char* data_ptr  = static_cast<const char*>(contiguous_tensor.data_ptr());
            converted.set_int32_data(data_ptr, num_bytes);
            break;
        }
        case torch::kFloat16: {
            size_t      num_bytes = contiguous_tensor.numel() * sizeof(c10::Half);
            const char* data_ptr  = static_cast<const char*>(contiguous_tensor.data_ptr());
            converted.set_fp16_data(data_ptr, num_bytes);
            break;
        }
        case torch::kBFloat16: {
            size_t      num_bytes = contiguous_tensor.numel() * sizeof(c10::BFloat16);
            const char* data_ptr  = static_cast<const char*>(contiguous_tensor.data_ptr());
            converted.set_bf16_data(data_ptr, num_bytes);
            break;
        }
        default:
            throw std::runtime_error("Unsupported tensor data type.");
    }
    tensor_pb->Swap(&converted);
}

}  // namespace rtp_llm
