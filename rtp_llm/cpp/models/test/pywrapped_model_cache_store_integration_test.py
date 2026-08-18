import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.cpp.models.test.libth_pywrapped_model_cache_store_integration_test import (
    PyModelInputs,
    PyModelOutputs,
    run_scenario,
)
from rtp_llm.models_py.modules.factory.attention import attn_factory
from rtp_llm.models_py.modules.hybrid.indexer import Indexer


class _SparseMlaImpl:
    @staticmethod
    def support(attn_configs, attn_inputs) -> bool:
        return True

    @staticmethod
    def is_sparse() -> bool:
        return True

    def __init__(self, *args, **kwargs) -> None:
        pass

    def support_cuda_graph(self) -> bool:
        return True


class _RequestLocalMlaImpl(_SparseMlaImpl):
    @classmethod
    def support_parallelism_config(cls, parallelism_config) -> bool:
        return False


class _ContextParallelMlaImpl(_SparseMlaImpl):
    @classmethod
    def support_parallelism_config(cls, parallelism_config) -> bool:
        return True


class CacheStoreForwardModel:
    """Test model that replaces attention math but keeps the real cache-store call."""

    def __init__(self) -> None:
        self.kv_cache = None
        self.forward_calls = 0
        self.micro_batch_calls = 0
        self.seen_input_lengths: list[list[int]] = []

    def initialize(self, resources) -> bool:
        self.kv_cache = resources.kv_cache
        return True

    def prepare_fmha_impl(self, inputs: PyModelInputs, is_cuda_graph: bool = False):
        return None

    def _forward_one(self, inputs: PyModelInputs) -> PyModelOutputs:
        attention_inputs = inputs.attention_inputs
        request_inputs = (
            list(attention_inputs.values())
            if isinstance(attention_inputs, dict)
            else [attention_inputs]
        )
        first_inputs = request_inputs[0]
        self.seen_input_lengths.append(first_inputs.input_lengths.tolist())

        assert self.kv_cache is not None
        for layer_cache in self.kv_cache.get_layer_cache_groups(0):
            tag_inputs = (
                attention_inputs[layer_cache.tag]
                if isinstance(attention_inputs, dict)
                else attention_inputs
            )
            if (
                tag_inputs.cache_store_inputs is not None
                and tag_inputs.cache_store_writer is not None
            ):
                tag_inputs.cache_store_writer.write(
                    tag_inputs.cache_store_inputs, layer_cache
                )

        hidden_states = torch.zeros(
            (inputs.input_ids.numel(), 1),
            dtype=torch.float16,
            device=inputs.input_ids.device,
        )
        return PyModelOutputs(hidden_states)

    def forward(self, inputs: PyModelInputs, fmha_impl=None) -> PyModelOutputs:
        self.forward_calls += 1
        return self._forward_one(inputs)

    def forward_micro_batch(self, inputs: list[PyModelInputs]) -> list[PyModelOutputs]:
        self.micro_batch_calls += 1
        return [self._forward_one(model_inputs) for model_inputs in inputs]


class ContextParallelRoutingModel(CacheStoreForwardModel):
    def __init__(self) -> None:
        super().__init__()
        attention_config = SimpleNamespace(
            indexer_topk=0,
            is_sparse=True,
            use_mla=True,
        )
        self.model_config = SimpleNamespace(
            getAttentionConfigs=lambda _tp_size: attention_config,
            max_seq_len=64,
            quant_config=None,
        )
        self.parallelism_config = SimpleNamespace(
            get_attn_tp_size=lambda: 1,
            prefill_cp_config=SimpleNamespace(is_enabled=lambda: True),
        )
        self.weights = SimpleNamespace(
            weights=[],
            get_global_weight_or_none=lambda _name: None,
        )
        self.routing: list[tuple[str, type, bool, bool]] = []

    def prepare_fmha_impl(self, inputs: PyModelInputs, is_cuda_graph: bool = False):
        attention_inputs = inputs.attention_inputs
        request_inputs = (
            list(attention_inputs.values())
            if isinstance(attention_inputs, dict)
            else [attention_inputs]
        )
        first_inputs = request_inputs[0]
        with patch.object(
            attn_factory,
            "PREFILL_MLA_IMPS",
            [_RequestLocalMlaImpl, _ContextParallelMlaImpl],
        ), patch.object(
            attn_factory,
            "DECODE_MLA_IMPS",
            [_RequestLocalMlaImpl, _ContextParallelMlaImpl],
        ):
            selected = attn_factory.AttnImplFactory.get_fmha_impl(
                self.model_config,
                self.parallelism_config,
                self.weights,
                first_inputs,
                is_cuda_graph=is_cuda_graph,
            )

        if first_inputs.is_target_verify:
            request_kind = "target_verify"
        elif first_inputs.is_prefill:
            request_kind = "prefill"
        else:
            request_kind = "decode"
        self.routing.append(
            (
                request_kind,
                type(selected),
                all(item.context_parallel_info is not None for item in request_inputs),
                Indexer._is_sparse_prefill_cp(first_inputs),
            )
        )
        return selected


def _blocks_by_key(result: dict) -> dict[str, dict]:
    return {
        block["key"]: block
        for record in result["records"]
        for block in record["blocks"]
    }


def _record_for_request(result: dict, request_id: int) -> dict:
    matches = [
        record
        for record in result["records"]
        if record["request_id"] == str(request_id)
    ]
    if len(matches) != 1:
        raise AssertionError(
            f"expected one record for request {request_id}, got {len(matches)}"
        )
    return matches[0]


class PyWrappedModelCacheStoreIntegrationTest(unittest.TestCase):
    def test_context_parallel_async_prepare_does_not_leak_into_next_graph_request(
        self,
    ) -> None:
        model = ContextParallelRoutingModel()
        result = run_scenario(model, "cp_graph_routing")

        self.assertEqual(model.seen_input_lengths, [[4]])
        self.assertEqual(
            model.routing,
            [("prefill", _ContextParallelMlaImpl, True, True)],
        )
        self.assertEqual(result["graph_can_run_calls"], 2)
        self.assertEqual(result["graph_prepare_calls"], 1)
        self.assertEqual(result["graph_forward_calls"], 1)

    def test_context_parallel_routing_is_request_local_across_cpp_pybind(self) -> None:
        model = ContextParallelRoutingModel()
        run_scenario(model, "cp_request_routing")

        self.assertEqual(model.seen_input_lengths, [[4], [3], [1]])
        self.assertEqual(
            model.routing,
            [
                ("prefill", _ContextParallelMlaImpl, True, True),
                ("target_verify", _RequestLocalMlaImpl, False, False),
                ("decode", _RequestLocalMlaImpl, False, False),
            ],
        )

    def test_multi_tag_uses_each_tag_local_physical_block_table(self) -> None:
        model = CacheStoreForwardModel()
        result = run_scenario(model, "multi_tag")

        self.assertEqual(model.forward_calls, 1)
        self.assertEqual(len(result["records"]), 2)
        blocks = _blocks_by_key(result)

        full_blocks = {
            key: block for key, block in blocks.items() if "_tag_full" in key
        }
        linear_blocks = {
            key: block for key, block in blocks.items() if "_tag_linear" in key
        }
        self.assertEqual(len(full_blocks), 2)
        self.assertEqual(len(linear_blocks), 4)
        self.assertEqual(
            sorted(
                block["address"] - result["base_addresses"]["full"]
                for block in full_blocks.values()
            ),
            [16, 32],
        )
        self.assertEqual(
            sorted(
                block["address"] - result["base_addresses"]["linear"]
                for block in linear_blocks.values()
            ),
            [72, 96, 120, 144],
        )
        self.assertEqual({block["length"] for block in full_blocks.values()}, {16})
        self.assertEqual({block["length"] for block in linear_blocks.values()}, {24})

    def test_micro_batch_slices_request_metadata_with_block_rows(self) -> None:
        model = CacheStoreForwardModel()
        result = run_scenario(model, "micro_batch")

        self.assertEqual(model.forward_calls, 0)
        self.assertEqual(model.micro_batch_calls, 1)
        self.assertEqual(model.seen_input_lengths, [[2, 4], [2]])
        self.assertEqual(len(result["records"]), 3)

        expected = {
            201: ([2101], [16]),
            202: ([2201, 2202], [32, 48]),
            203: ([2301], [64]),
        }
        base = result["base_addresses"]["default"]
        for request_id, (token_keys, offsets) in expected.items():
            record = _record_for_request(result, request_id)
            self.assertEqual(len(record["blocks"]), len(token_keys))
            self.assertEqual(
                sorted(block["address"] - base for block in record["blocks"]),
                offsets,
            )
            for token_key in token_keys:
                self.assertTrue(
                    any(
                        f"_token_id_str_{token_key}_" in block["key"]
                        for block in record["blocks"]
                    )
                )

    def test_context_parallel_publishes_original_lengths_not_local_chunk(self) -> None:
        model = CacheStoreForwardModel()
        result = run_scenario(model, "cp_actual_lengths")

        # CP turns the six-token request into a four-token rank-local chunk for
        # attention, while CacheStore must still publish three two-token blocks.
        self.assertEqual(model.seen_input_lengths, [[4]])
        record = _record_for_request(result, 301)
        self.assertEqual(len(record["blocks"]), 3)
        base = result["base_addresses"]["default"]
        self.assertEqual(
            sorted(block["address"] - base for block in record["blocks"]),
            [16, 32, 48],
        )
        self.assertEqual(
            sorted(
                token_key
                for token_key in (3102, 3104, 3106)
                if any(
                    f"_token_id_str_{token_key}_" in block["key"]
                    for block in record["blocks"]
                )
            ),
            [3102, 3104, 3106],
        )

    def test_mtp_writer_uses_selected_sub_config_for_real_write(self) -> None:
        model = CacheStoreForwardModel()
        result = run_scenario(model, "mtp_sub_config")

        record = _record_for_request(result, 401)
        self.assertEqual(len(record["blocks"]), 2)
        base = result["base_addresses"]["draft"]
        self.assertEqual(
            sorted(block["address"] - base for block in record["blocks"]),
            [32, 64],
        )
        self.assertEqual({block["length"] for block in record["blocks"]}, {32})
        self.assertTrue(
            all("model_id_7_" in block["key"] for block in record["blocks"])
        )
        self.assertTrue(all("_tag_draft" in block["key"] for block in record["blocks"]))


if __name__ == "__main__":
    unittest.main()
