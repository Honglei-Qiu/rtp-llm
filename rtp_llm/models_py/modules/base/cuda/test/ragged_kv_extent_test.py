"""
Unit test for the KV extent used by IndexerOp._get_topk_ragged.

ks/ke are built over the full KV extent (input_len + prefix_len per request), so
the gathered-KV buffers must be sized by that extent rather than by the query row
count. The gather kernel clamps to dst_k.size(0), so an undersized buffer leaves
the tail rows uninitialized while fp8_mqa_logits still reads up to ke.

Runs on CPU: the CUDA entry points are replaced with recorders.
"""

from unittest import TestCase, main
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.base.cuda import indexer_op as indexer_op_module
from rtp_llm.models_py.modules.base.cuda.indexer_op import IndexerOp

INDEX_HEAD_DIM = 128
BLOCK_SIZE = 128
INDEX_TOPK = 8


class _FmhaParams:
    def __init__(self, num_query_tokens: int, total_kv_tokens: int):
        self.ks = torch.zeros(num_query_tokens, dtype=torch.int32)
        self.ke = torch.full((num_query_tokens,), total_kv_tokens, dtype=torch.int32)
        self.expanded_seq_lens = torch.full(
            (num_query_tokens,), total_kv_tokens, dtype=torch.int32
        )
        self.topk_indices_offset = torch.zeros(num_query_tokens, dtype=torch.int32)
        self.total_kv_tokens = total_kv_tokens


class _AttentionInputs:
    def __init__(self, total_kv_tokens: int):
        self.kv_cache_kernel_block_id_device = torch.zeros((1, 4), dtype=torch.int32)
        self.cu_kv_seqlens_device = torch.tensor(
            [0, total_kv_tokens], dtype=torch.int32
        )


class _KVCache:
    def __init__(self):
        self.kv_scale_base = torch.zeros((4, BLOCK_SIZE, INDEX_HEAD_DIM))


class _RecordingOps:
    def __init__(self):
        self.gathered_rows = None
        self.gathered_scale_rows = None

    def cp_gather_indexer_k_quant_cache(
        self, kv_cache, dst_k, dst_scale, block_table, cu_seq_lens
    ):
        self.gathered_rows = dst_k.shape[0]
        self.gathered_scale_rows = dst_scale.shape[0]


class RaggedKvExtentTest(TestCase):
    def _run(self, num_query_tokens: int, total_kv_tokens: int) -> _RecordingOps:
        op = IndexerOp.__new__(IndexerOp)
        op.index_head_dim = INDEX_HEAD_DIM
        op.block_size = BLOCK_SIZE
        op.index_topk = INDEX_TOPK

        recording_ops = _RecordingOps()
        fmha_params = _FmhaParams(num_query_tokens, total_kv_tokens)
        q_fp8 = torch.zeros(
            (num_query_tokens, INDEX_HEAD_DIM), dtype=torch.float8_e4m3fn
        )
        weights = torch.zeros((num_query_tokens, 1, 1))

        def fake_mqa_logits(q, kv_fp8, w, ks, ke, clean_logits=False):
            k_fp8, _ = kv_fp8
            self.assertGreaterEqual(
                k_fp8.shape[0],
                int(ke.max().item()),
                "fp8_mqa_logits would read past the gathered KV rows",
            )
            return torch.zeros((q.shape[0], k_fp8.shape[0]))

        def fake_topk(score, lengths, topk_indices_offset, topk, row_starts):
            return torch.zeros((score.shape[0], topk), dtype=torch.int32)

        with patch.object(indexer_op_module, "rtp_llm_ops", recording_ops), patch.object(
            indexer_op_module, "deep_gemm"
        ) as fake_deep_gemm, patch(
            "rtp_llm.models_py.kernels.cuda.fast_topk.fast_topk_transform_ragged_fused",
            fake_topk,
        ):
            fake_deep_gemm.fp8_mqa_logits = fake_mqa_logits
            op._get_topk_ragged(
                q_fp8,
                weights,
                _KVCache(),
                fmha_params,
                _AttentionInputs(total_kv_tokens),
            )
        return recording_ops

    def test_prefix_reuse_gathers_full_kv_extent(self):
        """With a reused prefix the KV extent exceeds the query rows."""
        recording_ops = self._run(num_query_tokens=6, total_kv_tokens=70)
        self.assertEqual(recording_ops.gathered_rows, 70)
        self.assertEqual(recording_ops.gathered_scale_rows, 70)

    def test_no_prefix_reuse_matches_query_rows(self):
        recording_ops = self._run(num_query_tokens=6, total_kv_tokens=6)
        self.assertEqual(recording_ops.gathered_rows, 6)

    def test_kv_extent_below_query_rows_is_rejected(self):
        # An explicit raise, not an assert: the guard has to hold under python -O too.
        with self.assertRaises(RuntimeError):
            self._run(num_query_tokens=6, total_kv_tokens=4)


if __name__ == "__main__":
    main()
