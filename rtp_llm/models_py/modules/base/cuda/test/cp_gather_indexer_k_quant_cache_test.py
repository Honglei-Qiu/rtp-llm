from unittest import SkipTest, TestCase, main

import torch

from rtp_llm.ops.compute_ops import rtp_llm_ops


class CPGatherIndexerKQuantCacheTest(TestCase):
    head_dim = 128
    cache_block_size = 4
    quant_block_size = 128

    def setUp(self):
        if not torch.cuda.is_available():
            raise SkipTest("CUDA is required")
        self.device = torch.device("cuda:0")
        torch.cuda.set_device(self.device)

    def _scale_width(self, head_dim):
        return head_dim // self.quant_block_size * 4

    def _make_cache(self, head_dim=None):
        head_dim = head_dim or self.head_dim
        scale_width = self._scale_width(head_dim)
        cache = torch.zeros(
            (2, self.cache_block_size, head_dim + scale_width),
            dtype=torch.uint8,
            device=self.device,
        )
        flat = cache.flatten()
        for block in range(cache.size(0)):
            block_base = block * cache.stride(0)
            for token in range(self.cache_block_size):
                value = block * 10 + token + 1
                start = block_base + token * head_dim
                flat[start : start + head_dim] = value
                scale_byte_offset = (
                    block_base
                    + self.cache_block_size * head_dim
                    + token * scale_width
                )
                flat[scale_byte_offset : scale_byte_offset + 4].view(
                    torch.float32
                ).fill_(float(value) / 10.0)
        return cache

    def _gather(self, block_table, cu_seq_lens, num_tokens, head_dim=None):
        head_dim = head_dim or self.head_dim
        dst_k = torch.full(
            (num_tokens, head_dim),
            0xA5,
            dtype=torch.uint8,
            device=self.device,
        )
        dst_scale = torch.full(
            (num_tokens, self._scale_width(head_dim)),
            0xA5,
            dtype=torch.uint8,
            device=self.device,
        )
        rtp_llm_ops.cp_gather_indexer_k_quant_cache(
            self._make_cache(head_dim),
            dst_k,
            dst_scale,
            torch.tensor(block_table, dtype=torch.int32, device=self.device),
            torch.tensor(cu_seq_lens, dtype=torch.int32, device=self.device),
        )
        torch.cuda.synchronize()
        return dst_k, dst_scale.view(torch.float32)

    def test_gathers_last_partial_logical_page(self):
        dst_k, dst_scale = self._gather([[1, 0]], [0, 6], num_tokens=6)

        expected_values = torch.tensor(
            [11, 12, 13, 14, 1, 2], dtype=torch.uint8, device=self.device
        )
        torch.testing.assert_close(
            dst_k,
            expected_values[:, None].expand(-1, self.head_dim),
        )
        torch.testing.assert_close(
            dst_scale[:, 0], expected_values.to(torch.float32) / 10.0
        )

    def test_multiwarp_multibatch_gather(self):
        batch_size = 10
        sequence_length = 13
        num_tokens = batch_size * sequence_length
        block_table = [[0, 1, 0, 1] for _ in range(batch_size)]
        cu_seq_lens = [batch * sequence_length for batch in range(batch_size + 1)]

        dst_k, dst_scale = self._gather(
            block_table, cu_seq_lens, num_tokens=num_tokens
        )

        expected_values = []
        for token_idx in range(num_tokens):
            local_idx = token_idx % sequence_length
            block_id = block_table[token_idx // sequence_length][
                local_idx // self.cache_block_size
            ]
            expected_values.append(
                block_id * 10 + local_idx % self.cache_block_size + 1
            )
        expected_values = torch.tensor(
            expected_values, dtype=torch.uint8, device=self.device
        )
        torch.testing.assert_close(
            dst_k,
            expected_values[:, None].expand(-1, self.head_dim),
        )
        torch.testing.assert_close(
            dst_scale[:, 0], expected_values.to(torch.float32) / 10.0
        )

    def test_widest_launch_tier_and_multi_head_dim_grid(self):
        # BLOCK_Y_SIZE=32 (256 threads, 8 warps) is only selected at num_tokens>=512,
        # and grid.y is ceil(head_dim/128), so head_dim=512 is the smallest shape that
        # also runs more than one head_dim block. The rest of the suite caps out at 130
        # tokens and head_dim 128, i.e. the 64-thread tier at grid.y=1.
        head_dim = 512
        batch_size = 16
        sequence_length = 32
        num_tokens = batch_size * sequence_length
        blocks_per_sequence = sequence_length // self.cache_block_size
        block_table = [
            [block % 2 for block in range(blocks_per_sequence)]
            for _ in range(batch_size)
        ]
        cu_seq_lens = [batch * sequence_length for batch in range(batch_size + 1)]

        dst_k, dst_scale = self._gather(
            block_table, cu_seq_lens, num_tokens=num_tokens, head_dim=head_dim
        )

        expected_values = []
        for token_idx in range(num_tokens):
            local_idx = token_idx % sequence_length
            block_id = block_table[token_idx // sequence_length][
                local_idx // self.cache_block_size
            ]
            expected_values.append(
                block_id * 10 + local_idx % self.cache_block_size + 1
            )
        expected_values = torch.tensor(
            expected_values, dtype=torch.uint8, device=self.device
        )
        torch.testing.assert_close(
            dst_k,
            expected_values[:, None].expand(-1, head_dim),
        )
        torch.testing.assert_close(
            dst_scale[:, 0], expected_values.to(torch.float32) / 10.0
        )
        self.assertEqual(torch.count_nonzero(dst_scale[:, 1:]).item(), 0)

    def test_invalid_physical_blocks_are_zero_filled(self):
        for block_id in (-1, 2):
            with self.subTest(block_id=block_id):
                dst_k, dst_scale = self._gather(
                    [[block_id]], [0, 2], num_tokens=2
                )
                self.assertEqual(torch.count_nonzero(dst_k).item(), 0)
                self.assertEqual(torch.count_nonzero(dst_scale).item(), 0)

    def test_empty_logical_page_table_is_zero_filled(self):
        dst_k, dst_scale = self._gather([[]], [0, 2], num_tokens=2)

        self.assertEqual(torch.count_nonzero(dst_k).item(), 0)
        self.assertEqual(torch.count_nonzero(dst_scale).item(), 0)

    def test_tokens_beyond_logical_block_table_are_zero_filled(self):
        dst_k, dst_scale = self._gather([[0]], [0, 6], num_tokens=6)

        torch.testing.assert_close(
            dst_k[:4, 0],
            torch.tensor([1, 2, 3, 4], dtype=torch.uint8, device=self.device),
        )
        self.assertEqual(torch.count_nonzero(dst_k[4:]).item(), 0)
        self.assertEqual(torch.count_nonzero(dst_scale[4:]).item(), 0)

    def test_tokens_outside_sequence_ranges_are_zero_filled(self):
        dst_k, dst_scale = self._gather([[0]], [1, 3], num_tokens=3)

        self.assertEqual(torch.count_nonzero(dst_k[0]).item(), 0)
        self.assertEqual(torch.count_nonzero(dst_scale[0]).item(), 0)
        torch.testing.assert_close(
            dst_k[1:, 0],
            torch.tensor([1, 2], dtype=torch.uint8, device=self.device),
        )

    def test_empty_output_is_a_noop(self):
        dst_k, dst_scale = self._gather([[0]], [0, 0], num_tokens=0)

        self.assertEqual(dst_k.numel(), 0)
        self.assertEqual(dst_scale.numel(), 0)

    def test_rejects_unsupported_quant_block_size(self):
        dst_k = torch.empty(
            (1, self.head_dim), dtype=torch.uint8, device=self.device
        )
        dst_scale = torch.empty((1, 8), dtype=torch.uint8, device=self.device)
        with self.assertRaisesRegex(RuntimeError, "quant_block_size must be 128"):
            rtp_llm_ops.cp_gather_indexer_k_quant_cache(
                self._make_cache(),
                dst_k,
                dst_scale,
                torch.tensor([[0]], dtype=torch.int32, device=self.device),
                torch.tensor([0, 1], dtype=torch.int32, device=self.device),
            )

    def test_rejects_non_multiple_head_dimension(self):
        head_dim = 257
        dst_k = torch.empty((1, head_dim), dtype=torch.uint8, device=self.device)
        dst_scale = torch.empty((1, 8), dtype=torch.uint8, device=self.device)
        kv_cache = torch.empty((1, 4, 265), dtype=torch.uint8, device=self.device)
        with self.assertRaisesRegex(
            RuntimeError, "head_dim must be divisible by quant_block_size"
        ):
            rtp_llm_ops.cp_gather_indexer_k_quant_cache(
                kv_cache,
                dst_k,
                dst_scale,
                torch.tensor([[0]], dtype=torch.int32, device=self.device),
                torch.tensor([0, 1], dtype=torch.int32, device=self.device),
            )

    # The remaining host TORCH_CHECKs are pinned one at a time by mutating a valid
    # payload; each test violates exactly one gate so a revert of that gate leaves
    # the corresponding assertion red.
    def _valid_wrapper_inputs(self):
        head_dim = 128
        dst_k = torch.empty((1, head_dim), dtype=torch.uint8, device=self.device)
        dst_scale = torch.empty((1, 4), dtype=torch.uint8, device=self.device)
        kv_cache = torch.empty((1, 4, 132), dtype=torch.uint8, device=self.device)
        block_table = torch.tensor([[0]], dtype=torch.int32, device=self.device)
        cu_seq_lens = torch.tensor([0, 1], dtype=torch.int32, device=self.device)
        return kv_cache, dst_k, dst_scale, block_table, cu_seq_lens

    def _call_with(self, **overrides):
        kv_cache, dst_k, dst_scale, block_table, cu_seq_lens = self._valid_wrapper_inputs()
        rtp_llm_ops.cp_gather_indexer_k_quant_cache(
            overrides.get("kv_cache", kv_cache),
            overrides.get("dst_k", dst_k),
            overrides.get("dst_scale", dst_scale),
            overrides.get("block_table", block_table),
            overrides.get("cu_seq_lens", cu_seq_lens),
        )

    def test_rejects_wrong_block_table_dtype(self):
        block_table = torch.tensor([[0]], dtype=torch.int64, device=self.device)
        with self.assertRaisesRegex(RuntimeError, "block_table must use int32"):
            self._call_with(block_table=block_table)

    def test_rejects_wrong_cu_seq_lens_dtype(self):
        cu_seq_lens = torch.tensor([0, 1], dtype=torch.int64, device=self.device)
        with self.assertRaisesRegex(RuntimeError, "cu_seq_lens must use int32"):
            self._call_with(cu_seq_lens=cu_seq_lens)

    def test_rejects_non_uint8_dst_scale(self):
        dst_scale = torch.empty((1, 4), dtype=torch.int8, device=self.device)
        with self.assertRaisesRegex(RuntimeError, "dst_scale must use uint8 byte storage"):
            self._call_with(dst_scale=dst_scale)

    def test_rejects_non_byte_dst_k(self):
        dst_k = torch.empty((1, 128), dtype=torch.float16, device=self.device)
        with self.assertRaisesRegex(RuntimeError, "dst_k elements must be one byte"):
            self._call_with(dst_k=dst_k)

    def test_rejects_non_byte_kv_cache(self):
        kv_cache = torch.empty((1, 4, 132), dtype=torch.float16, device=self.device)
        with self.assertRaisesRegex(RuntimeError, "kv_cache elements must be one byte"):
            self._call_with(kv_cache=kv_cache)

    def test_rejects_wrong_kv_cache_rank(self):
        kv_cache = torch.empty((1, 4, 132, 1), dtype=torch.uint8, device=self.device)
        with self.assertRaisesRegex(RuntimeError, "kv_cache must be a 3D tensor"):
            self._call_with(kv_cache=kv_cache)

    def test_rejects_wrong_dst_k_rank(self):
        dst_k = torch.empty((128,), dtype=torch.uint8, device=self.device)
        with self.assertRaisesRegex(RuntimeError, "dst_k must be a 2D tensor"):
            self._call_with(dst_k=dst_k)

    def test_rejects_wrong_dst_scale_rank(self):
        dst_scale = torch.empty((4,), dtype=torch.uint8, device=self.device)
        with self.assertRaisesRegex(RuntimeError, "dst_scale must be a 2D tensor"):
            self._call_with(dst_scale=dst_scale)

    def test_rejects_wrong_block_table_rank(self):
        block_table = torch.tensor([0], dtype=torch.int32, device=self.device)
        with self.assertRaisesRegex(RuntimeError, "block_table must be a 2D tensor"):
            self._call_with(block_table=block_table)

    def test_rejects_wrong_cu_seq_lens_rank(self):
        cu_seq_lens = torch.tensor([[0, 1]], dtype=torch.int32, device=self.device)
        with self.assertRaisesRegex(RuntimeError, "cu_seq_lens must be a 1D tensor"):
            self._call_with(cu_seq_lens=cu_seq_lens)

    def test_rejects_non_contiguous_kv_cache(self):
        kv_cache = torch.empty((1, 4, 264), dtype=torch.uint8, device=self.device)[:, :, ::2]
        with self.assertRaisesRegex(RuntimeError, "kv_cache must be contiguous"):
            self._call_with(kv_cache=kv_cache)

    def test_rejects_non_contiguous_dst_k(self):
        dst_k = torch.empty((1, 256), dtype=torch.uint8, device=self.device)[:, ::2]
        with self.assertRaisesRegex(RuntimeError, "dst_k must be contiguous"):
            self._call_with(dst_k=dst_k)

    def test_rejects_cross_device_tensors(self):
        cpu_block_table = torch.tensor([[0]], dtype=torch.int32)
        with self.assertRaisesRegex(RuntimeError, "must be on the same device"):
            self._call_with(block_table=cpu_block_table)

    def test_rejects_dst_scale_width_mismatch(self):
        # quant_block_size is derived as head_dim * 4 / dst_scale.size(1), so a narrower
        # dst_scale normally trips the "quant_block_size must be 128" gate first. head_dim
        # 4224 is the smallest multiple of 128 whose integer division leaves slack: both
        # 131 and 132 give quant_block_size 128, so only 131 reaches the width gate.
        head_dim = 4224
        dst_k = torch.empty((1, head_dim), dtype=torch.uint8, device=self.device)
        dst_scale = torch.empty((1, 131), dtype=torch.uint8, device=self.device)
        kv_cache = torch.empty(
            (1, self.cache_block_size, head_dim + 132),
            dtype=torch.uint8,
            device=self.device,
        )
        with self.assertRaisesRegex(
            RuntimeError, "dst_scale width does not match head_dim and quant_block_size"
        ):
            self._call_with(kv_cache=kv_cache, dst_k=dst_k, dst_scale=dst_scale)


if __name__ == "__main__":
    main()
