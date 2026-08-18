import sys
from types import ModuleType
from unittest import TestCase, main, mock

from rtp_llm.models_py.kernels.cuda import deepgemm_wrapper as wrapper


class DeepGemmWrapperTest(TestCase):
    def setUp(self) -> None:
        self._impl_names = tuple(
            f"_{symbol}_impl" for symbol in wrapper._deep_gemm_impl_new_map
        )
        self._saved_impls = {
            name: getattr(wrapper, name) for name in self._impl_names
        }
        self._saved_available = wrapper._deep_gemm_available
        self._saved_full_name_flags = dict(wrapper._deep_gemm_impl_uses_full_name)
        self._reset_impls()
        wrapper._deep_gemm_available = None

    def tearDown(self) -> None:
        for name, impl in self._saved_impls.items():
            setattr(wrapper, name, impl)
        wrapper._deep_gemm_impl_uses_full_name.clear()
        wrapper._deep_gemm_impl_uses_full_name.update(self._saved_full_name_flags)
        wrapper._deep_gemm_available = self._saved_available

    def _reset_impls(self) -> None:
        for name in self._impl_names:
            setattr(wrapper, name, None)
        wrapper._deep_gemm_impl_uses_full_name.clear()

    def _resolve_from(self, module: ModuleType, symbol: str):
        with mock.patch.object(wrapper, "has_deep_gemm", return_value=True):
            with mock.patch.dict(sys.modules, {"deep_gemm": module}):
                return wrapper._resolve_deep_gemm_impl(symbol)

    def test_negative_probe_is_retried_then_success_is_cached(self) -> None:
        with mock.patch.object(
            wrapper.importlib.util,
            "find_spec",
            side_effect=(None, object()),
        ) as find_spec:
            self.assertFalse(wrapper.has_deep_gemm())
            self.assertTrue(wrapper.has_deep_gemm())
            self.assertTrue(wrapper.has_deep_gemm())

        self.assertEqual(find_spec.call_count, 2)

    def test_loaded_module_without_spec_is_available(self) -> None:
        module = ModuleType("deep_gemm")
        with mock.patch.dict(sys.modules, {"deep_gemm": module}):
            with mock.patch.object(
                wrapper.importlib.util,
                "find_spec",
                side_effect=ValueError("missing module spec"),
            ) as find_spec:
                self.assertTrue(wrapper.has_deep_gemm())

        find_spec.assert_not_called()

    def test_probe_error_does_not_poison_later_resolution(self) -> None:
        with mock.patch.object(
            wrapper.importlib.util,
            "find_spec",
            side_effect=(ValueError("missing module spec"), object()),
        ) as find_spec:
            self.assertFalse(wrapper.has_deep_gemm())
            self.assertTrue(wrapper.has_deep_gemm())

        self.assertEqual(find_spec.call_count, 2)

    def test_short_name_takes_precedence_over_full_name(self) -> None:
        module = ModuleType("deep_gemm")
        short_impl = mock.Mock()
        full_impl = mock.Mock()
        module.bf16_gemm_nt = short_impl
        module.gemm_bf16_bf16_bf16_nt = full_impl

        self.assertIs(self._resolve_from(module, "bf16_gemm_nt"), short_impl)
        self.assertFalse(wrapper._deep_gemm_impl_uses_full_name["bf16_gemm_nt"])

    def test_short_alias_bound_to_full_callable_uses_raw_abi(self) -> None:
        for symbol, full_name in wrapper._deep_gemm_impl_full_name_map.items():
            with self.subTest(symbol=symbol):
                self._reset_impls()
                module = ModuleType("deep_gemm")
                raw_impl = mock.Mock()
                setattr(module, wrapper._deep_gemm_impl_new_map[symbol], raw_impl)
                setattr(module, full_name, raw_impl)

                self.assertIs(self._resolve_from(module, symbol), raw_impl)
                self.assertTrue(wrapper._deep_gemm_impl_uses_full_name[symbol])

    def test_legacy_masked_fp8_name_is_preserved(self) -> None:
        module = ModuleType("deep_gemm")
        legacy_impl = mock.Mock()
        module.fp8_m_grouped_gemm_nt_masked = legacy_impl

        self.assertIs(
            self._resolve_from(module, "m_grouped_fp8_gemm_nt_masked"),
            legacy_impl,
        )
        self.assertFalse(
            wrapper._deep_gemm_impl_uses_full_name[
                "m_grouped_fp8_gemm_nt_masked"
            ]
        )

    def test_legacy_masked_fp8_keeps_wrapper_call_shape(self) -> None:
        module = ModuleType("deep_gemm")
        legacy_impl = mock.Mock()
        module.fp8_m_grouped_gemm_nt_masked = legacy_impl
        self._resolve_from(module, "m_grouped_fp8_gemm_nt_masked")

        with mock.patch.object(
            wrapper,
            "maybe_pack_ue8m0_scale",
            side_effect=lambda x, scale, disable: f"packed_{scale}",
        ) as maybe_pack:
            wrapper.m_grouped_fp8_gemm_nt_masked(
                ("a", "a_scale"),
                ("b", "b_scale"),
                "output",
                "masked_m",
                7,
                disable_ue8m0_cast=False,
            )

        legacy_impl.assert_called_once_with(
            ("a", "packed_a_scale"),
            ("b", "packed_b_scale"),
            "output",
            "masked_m",
            7,
            compiled_dims="nk",
            disable_ue8m0_cast=False,
        )
        self.assertEqual(
            maybe_pack.call_args_list,
            [
                mock.call("a", "a_scale", False),
                mock.call("b", "b_scale", False),
            ],
        )

    def test_legacy_alias_bound_to_raw_masked_fp8_skips_scale_pack(self) -> None:
        module = ModuleType("deep_gemm")
        raw_impl = mock.Mock()
        a_scale = object()
        b_scale = object()
        module.fp8_m_grouped_gemm_nt_masked = raw_impl
        module.m_grouped_gemm_fp8_fp8_bf16_nt_masked = raw_impl
        self._resolve_from(module, "m_grouped_fp8_gemm_nt_masked")

        with mock.patch.object(
            wrapper, "maybe_pack_ue8m0_scale"
        ) as maybe_pack:
            wrapper.m_grouped_fp8_gemm_nt_masked(
                ("a", a_scale),
                ("b", b_scale),
                "output",
                "masked_m",
                7,
                disable_ue8m0_cast=False,
            )

        maybe_pack.assert_not_called()
        raw_impl.assert_called_once_with(
            ("a", a_scale),
            ("b", b_scale),
            "output",
            "masked_m",
            7,
            None,
        )
        self.assertIs(raw_impl.call_args.args[0][1], a_scale)
        self.assertIs(raw_impl.call_args.args[1][1], b_scale)

    def test_each_full_name_fallback_resolves_independently(self) -> None:
        for symbol, full_name in wrapper._deep_gemm_impl_full_name_map.items():
            with self.subTest(symbol=symbol):
                self._reset_impls()
                module = ModuleType("deep_gemm")
                full_impl = mock.Mock()
                setattr(module, full_name, full_impl)

                self.assertIs(self._resolve_from(module, symbol), full_impl)
                self.assertTrue(wrapper._deep_gemm_impl_uses_full_name[symbol])

    def test_unavailable_probe_does_not_poison_later_resolution(self) -> None:
        module = ModuleType("deep_gemm")
        full_impl = mock.Mock()
        module.gemm_bf16_bf16_bf16_nt = full_impl

        with mock.patch.object(
            wrapper, "has_deep_gemm", side_effect=(False, True)
        ):
            with mock.patch.dict(sys.modules, {"deep_gemm": module}):
                self.assertIsNone(wrapper._resolve_deep_gemm_impl("bf16_gemm_nt"))
                self.assertIs(
                    wrapper._resolve_deep_gemm_impl("bf16_gemm_nt"), full_impl
                )

    def test_successful_resolution_is_cached(self) -> None:
        module = ModuleType("deep_gemm")
        first_impl = mock.Mock()
        second_impl = mock.Mock()
        module.gemm_bf16_bf16_bf16_nt = first_impl

        self.assertIs(self._resolve_from(module, "bf16_gemm_nt"), first_impl)
        module.gemm_bf16_bf16_bf16_nt = second_impl
        self.assertIs(self._resolve_from(module, "bf16_gemm_nt"), first_impl)

    def test_missing_symbol_reports_all_known_names(self) -> None:
        module = ModuleType("deep_gemm")
        full_impl = mock.Mock()

        with self.assertRaisesRegex(
            RuntimeError,
            "bf16_gemm_nt, gemm_bf16_bf16_bf16_nt",
        ):
            self._resolve_from(module, "bf16_gemm_nt")

        module.gemm_bf16_bf16_bf16_nt = full_impl
        self.assertIs(self._resolve_from(module, "bf16_gemm_nt"), full_impl)

    def test_non_callable_symbol_is_rejected(self) -> None:
        module = ModuleType("deep_gemm")
        module.gemm_bf16_bf16_bf16_nt = object()

        with self.assertRaisesRegex(RuntimeError, "not callable"):
            self._resolve_from(module, "bf16_gemm_nt")

    def test_invalid_symbol_is_rejected_before_import(self) -> None:
        with mock.patch.object(wrapper, "has_deep_gemm") as has_deep_gemm:
            with self.assertRaisesRegex(ValueError, "unknown"):
                wrapper._resolve_deep_gemm_impl("unknown")

        has_deep_gemm.assert_not_called()

    def test_full_name_adapters_use_raw_call_shapes(self) -> None:
        cases = (
            (
                "fp8_gemm_nt",
                "gemm_fp8_fp8_bf16_nt",
                lambda: wrapper.fp8_gemm_nt(
                    "a", "b", "output", disable_ue8m0_cast=False
                ),
                ("a", "b", "output", None),
            ),
            (
                "m_grouped_fp8_gemm_nt_contiguous",
                "m_grouped_gemm_fp8_fp8_bf16_nt_contiguous",
                lambda: wrapper.m_grouped_fp8_gemm_nt_contiguous(
                    "a", "b", "output", "indices", disable_ue8m0_cast=False
                ),
                ("a", "b", "output", "indices", None),
            ),
            (
                "m_grouped_fp8_gemm_nt_masked",
                "m_grouped_gemm_fp8_fp8_bf16_nt_masked",
                lambda: wrapper.m_grouped_fp8_gemm_nt_masked(
                    ("a", "a_scale"),
                    ("b", "b_scale"),
                    "output",
                    "masked_m",
                    7,
                    disable_ue8m0_cast=False,
                ),
                (("a", "a_scale"), ("b", "b_scale"), "output", "masked_m", 7, None),
            ),
            (
                "bf16_gemm_nt",
                "gemm_bf16_bf16_bf16_nt",
                lambda: wrapper.bf16_gemm_nt("a", "b", "output"),
                ("a", "b", "output", None),
            ),
            (
                "m_grouped_bf16_gemm_nt_contiguous",
                "m_grouped_gemm_bf16_bf16_bf16_nt_contiguous",
                lambda: wrapper.m_grouped_bf16_gemm_nt_contiguous(
                    "a", "b", "output", "indices"
                ),
                ("a", "b", "output", "indices", None),
            ),
            (
                "m_grouped_bf16_gemm_nt_masked",
                "m_grouped_gemm_bf16_bf16_bf16_nt_masked",
                lambda: wrapper.m_grouped_bf16_gemm_nt_masked(
                    "a", "b", "output", "masked_m", 7
                ),
                ("a", "b", "output", "masked_m", 7, None),
            ),
        )

        for symbol, full_name, invoke, expected in cases:
            with self.subTest(symbol=symbol):
                self._reset_impls()
                module = ModuleType("deep_gemm")
                full_impl = mock.Mock()
                setattr(module, full_name, full_impl)
                self._resolve_from(module, symbol)
                if symbol == "m_grouped_fp8_gemm_nt_masked":
                    with mock.patch.object(
                        wrapper,
                        "maybe_pack_ue8m0_scale",
                    ) as maybe_pack:
                        invoke()
                    maybe_pack.assert_not_called()
                else:
                    invoke()
                full_impl.assert_called_once_with(*expected)

    def test_short_alias_keeps_legacy_call_shape(self) -> None:
        module = ModuleType("deep_gemm")
        short_impl = mock.Mock()
        module.bf16_gemm_nt = short_impl

        with mock.patch.object(wrapper, "has_deep_gemm", return_value=True):
            with mock.patch.dict(sys.modules, {"deep_gemm": module}):
                wrapper.bf16_gemm_nt("a", "b", "output", "c", "mn")

        short_impl.assert_called_once_with("a", "b", "output", "c", "mn")

    def test_full_name_rejects_unrepresentable_options(self) -> None:
        module = ModuleType("deep_gemm")
        full_impl = mock.Mock()
        module.gemm_bf16_bf16_bf16_nt = full_impl

        with self.assertRaisesRegex(ValueError, "bias tensor"):
            with mock.patch.object(wrapper, "has_deep_gemm", return_value=True):
                with mock.patch.dict(sys.modules, {"deep_gemm": module}):
                    wrapper.bf16_gemm_nt("a", "b", "output", "bias")
        full_impl.assert_not_called()

        with self.assertRaisesRegex(ValueError, "compiled_dims='nk'"):
            with mock.patch.object(wrapper, "has_deep_gemm", return_value=True):
                with mock.patch.dict(sys.modules, {"deep_gemm": module}):
                    wrapper.bf16_gemm_nt("a", "b", "output", compiled_dims="mn")
        full_impl.assert_not_called()

    def test_grouped_bf16_short_alias_preserves_compiled_dims(self) -> None:
        module = ModuleType("deep_gemm")
        contiguous_impl = mock.Mock()
        masked_impl = mock.Mock()
        module.m_grouped_bf16_gemm_nt_contiguous = contiguous_impl
        module.m_grouped_bf16_gemm_nt_masked = masked_impl

        with mock.patch.object(wrapper, "has_deep_gemm", return_value=True):
            with mock.patch.dict(sys.modules, {"deep_gemm": module}):
                wrapper.m_grouped_bf16_gemm_nt_contiguous(
                    "a", "b", "output", "indices", compiled_dims="mn"
                )
                wrapper.m_grouped_bf16_gemm_nt_masked(
                    "a", "b", "output", "masked_m", 7, compiled_dims="mn"
                )

        contiguous_impl.assert_called_once_with(
            "a", "b", "output", "indices", "mn"
        )
        masked_impl.assert_called_once_with(
            "a", "b", "output", "masked_m", 7, "mn"
        )

    def test_grouped_bf16_raw_rejects_nondefault_compiled_dims(self) -> None:
        module = ModuleType("deep_gemm")
        contiguous_impl = mock.Mock()
        masked_impl = mock.Mock()
        module.m_grouped_gemm_bf16_bf16_bf16_nt_contiguous = contiguous_impl
        module.m_grouped_gemm_bf16_bf16_bf16_nt_masked = masked_impl

        with mock.patch.object(wrapper, "has_deep_gemm", return_value=True):
            with mock.patch.dict(sys.modules, {"deep_gemm": module}):
                with self.assertRaisesRegex(ValueError, "compiled_dims='nk'"):
                    wrapper.m_grouped_bf16_gemm_nt_contiguous(
                        "a", "b", "output", "indices", compiled_dims="mn"
                    )
                with self.assertRaisesRegex(ValueError, "compiled_dims='nk'"):
                    wrapper.m_grouped_bf16_gemm_nt_masked(
                        "a",
                        "b",
                        "output",
                        "masked_m",
                        7,
                        compiled_dims="mn",
                    )

        contiguous_impl.assert_not_called()
        masked_impl.assert_not_called()


if __name__ == "__main__":
    main()
