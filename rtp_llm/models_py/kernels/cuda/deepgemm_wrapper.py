import functools
import importlib.util
import sys
from contextlib import contextmanager
from typing import Any, Callable, Generator, NoReturn, Optional, Tuple

import torch
import triton
import triton.language as tl

from rtp_llm.utils.module_util import resolve_symbol

__all__ = [
    "fp8_gemm_nt",
    "m_grouped_fp8_gemm_nt_contiguous",
    "m_grouped_fp8_gemm_nt_masked",
    "bf16_gemm_nt",
    "m_grouped_bf16_gemm_nt_contiguous",
    "m_grouped_bf16_gemm_nt_masked",
    "fp8_fp4_gemm_nt",
    "m_grouped_fp8_fp4_gemm_nt_contiguous",
    "m_grouped_fp8_fp4_gemm_nt_masked",
    "fp8_fp4_paged_mqa_logits",
    "per_token_cast_to_fp4",
    "cast_back_from_fp4",
    "transpose_packed_fp4",
    "tf32_hc_prenorm_gemm",
    "has_deep_gemm",
    "is_deep_gemm_e8m0_used",
    "configure_deep_gemm_num_sms",
    "maybe_pack_ue8m0_scale",
]

_deep_gemm_impl_new_map = {
    "fp8_gemm_nt": "fp8_gemm_nt",
    "m_grouped_fp8_gemm_nt_contiguous": "m_grouped_fp8_gemm_nt_contiguous",
    "m_grouped_fp8_gemm_nt_masked": "m_grouped_fp8_gemm_nt_masked",
    "bf16_gemm_nt": "bf16_gemm_nt",
    "m_grouped_bf16_gemm_nt_contiguous": "m_grouped_bf16_gemm_nt_contiguous",
    "m_grouped_bf16_gemm_nt_masked": "m_grouped_bf16_gemm_nt_masked",
    "fp8_fp4_gemm_nt": "fp8_fp4_gemm_nt",
    "m_grouped_fp8_fp4_gemm_nt_contiguous": "m_grouped_fp8_fp4_gemm_nt_contiguous",
    "m_grouped_fp8_fp4_gemm_nt_masked": "m_grouped_fp8_fp4_gemm_nt_masked",
    "fp8_fp4_paged_mqa_logits": "fp8_fp4_paged_mqa_logits",
    "per_token_cast_to_fp4": "per_token_cast_to_fp4",
    "cast_back_from_fp4": "cast_back_from_fp4",
    "transpose_packed_fp4": "transpose_packed_fp4",
    "tf32_hc_prenorm_gemm": "tf32_hc_prenorm_gemm",
}

_deep_gemm_impl_old_map = {
    "fp8_gemm_nt": "fp8_gemm_nt",
    "m_grouped_fp8_gemm_nt_contiguous": "m_grouped_fp8_gemm_nt_contiguous",
    "m_grouped_fp8_gemm_nt_masked": "fp8_m_grouped_gemm_nt_masked",
    "bf16_gemm_nt": "bf16_gemm_nt",
    "m_grouped_bf16_gemm_nt_contiguous": "m_grouped_bf16_gemm_nt_contiguous",
    "m_grouped_bf16_gemm_nt_masked": "m_grouped_bf16_gemm_nt_masked",
    "fp8_fp4_gemm_nt": "fp8_fp4_gemm_nt",
    "m_grouped_fp8_fp4_gemm_nt_contiguous": "m_grouped_fp8_fp4_gemm_nt_contiguous",
    "m_grouped_fp8_fp4_gemm_nt_masked": "m_grouped_fp8_fp4_gemm_nt_masked",
    "fp8_fp4_paged_mqa_logits": "fp8_fp4_paged_mqa_logits",
    "per_token_cast_to_fp4": "per_token_cast_to_fp4",
    "cast_back_from_fp4": "cast_back_from_fp4",
    "transpose_packed_fp4": "transpose_packed_fp4",
    "tf32_hc_prenorm_gemm": "tf32_hc_prenorm_gemm",
}

_deep_gemm_impl_full_name_map = {
    "fp8_gemm_nt": "gemm_fp8_fp8_bf16_nt",
    "m_grouped_fp8_gemm_nt_contiguous": "m_grouped_gemm_fp8_fp8_bf16_nt_contiguous",
    "m_grouped_fp8_gemm_nt_masked": "m_grouped_gemm_fp8_fp8_bf16_nt_masked",
    "bf16_gemm_nt": "gemm_bf16_bf16_bf16_nt",
    "m_grouped_bf16_gemm_nt_contiguous": "m_grouped_gemm_bf16_bf16_bf16_nt_contiguous",
    "m_grouped_bf16_gemm_nt_masked": "m_grouped_gemm_bf16_bf16_bf16_nt_masked",
}


_fp8_gemm_nt_impl: Callable[..., Any] | None = None
_m_grouped_fp8_gemm_nt_contiguous_impl: Callable[..., Any] | None = None
_m_grouped_fp8_gemm_nt_masked_impl: Callable[..., Any] | None = None
_bf16_gemm_nt_impl: Callable[..., Any] | None = None
_m_grouped_bf16_gemm_nt_contiguous_impl: Callable[..., Any] | None = None
_m_grouped_bf16_gemm_nt_masked_impl: Callable[..., Any] | None = None
_fp8_fp4_gemm_nt_impl: Callable[..., Any] | None = None
_m_grouped_fp8_fp4_gemm_nt_contiguous_impl: Callable[..., Any] | None = None
_m_grouped_fp8_fp4_gemm_nt_masked_impl: Callable[..., Any] | None = None
_fp8_fp4_paged_mqa_logits_impl: Callable[..., Any] | None = None
_per_token_cast_to_fp4_impl: Callable[..., Any] | None = None
_cast_back_from_fp4_impl: Callable[..., Any] | None = None
_transpose_packed_fp4_impl: Callable[..., Any] | None = None
_tf32_hc_prenorm_gemm_impl: Callable[..., Any] | None = None


_deep_gemm_available: bool | None = None

# A package may expose the raw fully-qualified callable under a short alias.
# Keep the ABI decision beside the resolved callable instead of inspecting a
# Python/C extension signature at runtime.
_deep_gemm_impl_uses_full_name: dict[str, bool] = {}


def has_deep_gemm() -> bool:
    """Return whether ``deep_gemm`` is available, retrying negative probes."""
    global _deep_gemm_available
    if _deep_gemm_available is True:
        return True
    # An already-loaded module can legitimately have no spec (for example,
    # after a controlled import or in a spawned worker test).
    if sys.modules.get("deep_gemm") is not None:
        _deep_gemm_available = True
        return True
    try:
        available = importlib.util.find_spec("deep_gemm") is not None
    except (ImportError, ValueError):
        available = False
    if available:
        _deep_gemm_available = True
    return available


@functools.cache
def is_deep_gemm_e8m0_used() -> bool:
    return torch.cuda.get_device_capability()[0] in [10, 12]


@contextmanager
def configure_deep_gemm_num_sms(num_sms: int) -> Generator[None, None, None]:
    """Configure the number of sms for deep gemm."""
    if not has_deep_gemm():
        raise RuntimeError(
            "DeepGEMM is not available. Please install the `deep_gemm` package to enable DeepGEMM kernels."
        )
    import deep_gemm

    # get original num sms
    original_num_sms = deep_gemm.get_num_sms()
    # set num sms
    deep_gemm.set_num_sms(num_sms)
    try:
        yield
    finally:
        # restore original num sms
        deep_gemm.set_num_sms(original_num_sms)


def _missing_deep_gemm() -> NoReturn:
    """Placeholder for unavailable DeepGEMM package."""
    raise RuntimeError(
        "DeepGEMM is not available. Please install the `deep_gemm` package to enable DeepGEMM kernels."
    )


def _resolve_deep_gemm_impl(symbol: str) -> Callable[..., Any] | None:
    """Resolve and cache one DeepGEMM callable on first use."""
    if symbol not in _deep_gemm_impl_new_map:
        raise ValueError(f"Invalid DeepGEMM symbol: {symbol}")

    impl_name = f"_{symbol}_impl"
    impl = globals()[impl_name]
    if impl is not None:
        return impl
    if not has_deep_gemm():
        return None

    import deep_gemm

    new_name = _deep_gemm_impl_new_map[symbol]
    old_name = _deep_gemm_impl_old_map[symbol]
    # Only the original fp8/bf16 symbols carry a raw full-name fallback; FP4/tf32
    # symbols added later route through the new/old names only, so tolerate absence.
    full_name = _deep_gemm_impl_full_name_map.get(symbol)
    full_impl = getattr(deep_gemm, full_name, None) if full_name else None
    impl = resolve_symbol(deep_gemm, new_name, old_name)
    if impl is None:
        impl = full_impl
    if impl is not None and not callable(impl):
        names = ", ".join(dict.fromkeys(n for n in (new_name, old_name, full_name) if n))
        raise RuntimeError(f"DeepGEMM symbol is not callable; tried: {names}")
    # Some releases bind a short alias directly to the raw full-name
    # function. Identity/name checks cover both ordinary Python functions and
    # extension callables without runtime signature inspection.
    uses_full_name = impl is not None and (
        impl is full_impl
        or getattr(impl, "__name__", None) == full_name
    )
    if impl is None:
        names = ", ".join(dict.fromkeys(n for n in (new_name, old_name, full_name) if n))
        raise RuntimeError(f"DeepGEMM symbol not found; tried: {names}")

    # Publish the ABI kind before the callable. A concurrent reader only
    # returns a cached callable after its matching call convention is visible.
    _deep_gemm_impl_uses_full_name[symbol] = uses_full_name
    globals()[impl_name] = impl
    return impl


def _validate_full_name_options(
    symbol: str, c: Any = None, compiled_dims: str = "nk"
) -> None:
    """Reject wrapper options that cannot be represented by the raw ABI."""
    if c is not None:
        raise ValueError(
            f"{symbol} full-name implementation does not support a bias tensor"
        )
    if compiled_dims != "nk":
        raise ValueError(
            f"{symbol} full-name implementation only supports compiled_dims='nk', "
            f"got {compiled_dims!r}"
        )


def _call_full_name_normal(
    symbol: str,
    impl: Callable[..., Any],
    a: Any,
    b: Any,
    output: Any,
    c: Any,
    compiled_dims: str,
) -> Any:
    _validate_full_name_options(symbol, c, compiled_dims)
    # The raw API's optional fourth argument is a tuning-config object. The
    # wrapper's bias and scale-cast controls are not part of that ABI.
    return impl(a, b, output, None)


def _call_full_name_grouped_contiguous(
    symbol: str,
    impl: Callable[..., Any],
    a: Any,
    b: Any,
    output: Any,
    m_indices: Any,
    compiled_dims: str,
) -> Any:
    _validate_full_name_options(symbol, compiled_dims=compiled_dims)
    return impl(a, b, output, m_indices, None)


def _call_full_name_grouped_masked(
    symbol: str,
    impl: Callable[..., Any],
    a: Any,
    b: Any,
    output: Any,
    masked_m: Any,
    expected_m: int,
    compiled_dims: str,
) -> Any:
    _validate_full_name_options(symbol, compiled_dims=compiled_dims)
    # Optional tuning remains at the raw API default and in a separate change.
    return impl(a, b, output, masked_m, expected_m, None)


@triton.jit
def pack_ue8m0_kernel_vectorized(
    scale_ptr,
    output_ptr,
    M,
    K,
    K_packed,
    stride_scale_b,
    stride_scale_m,
    stride_scale_k,
    stride_out_b,
    stride_out_k_packed,
    stride_out_m,
    gran_mn: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K_PACKED: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_b = tl.program_id(2)

    # Compute starting offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k_packed = pid_k * BLOCK_K_PACKED + tl.arange(0, BLOCK_K_PACKED)

    # K offset for loading 4 elements per packed output
    # Shape: (BLOCK_K_PACKED, 4)
    offs_k = offs_k_packed[:, None] * 4 + tl.arange(0, 4)[None, :]

    # Scale row indices
    row_idxs = offs_m // gran_mn

    # Compute scale pointers with shape (BLOCK_M, BLOCK_K_PACKED, 4)
    scale_ptrs = (
        scale_ptr
        + pid_b * stride_scale_b
        + row_idxs[:, None, None] * stride_scale_m
        + offs_k[None, :, :] * stride_scale_k
    )

    # Masks
    mask_m = offs_m < M
    mask_k = offs_k < K
    mask = mask_m[:, None, None] & mask_k[None, :, :]

    # Load scale values
    vals = tl.load(scale_ptrs, mask=mask, other=0.0)

    # Convert to UE8M0 using bitcast and shift
    vals_i32 = vals.to(tl.int32, bitcast=True)
    # Extract exponent (8 bits) and mask to ensure only 8 bits
    exponents = (vals_i32 >> 23) & 0xFF

    # Pack 4 bytes into int32 using vectorized shifts
    # exponents shape: (BLOCK_M, BLOCK_K_PACKED, 4)
    # We want to pack along the last dimension

    # Create shift amounts: [0, 8, 16, 24]
    shifts = tl.arange(0, 4)[None, None, :] * 8

    # Shift each exponent to its position and combine
    shifted = exponents << shifts
    # Sum along the last axis to pack
    packed = tl.sum(shifted, axis=2).to(tl.int32)

    # Compute output pointers
    out_ptrs = (
        output_ptr
        + pid_b * stride_out_b
        + offs_k_packed[None, :] * stride_out_k_packed
        + offs_m[:, None] * stride_out_m
    )

    # Output mask
    mask_out = mask_m[:, None] & (offs_k_packed[None, :] < K_packed)

    tl.store(out_ptrs, packed, mask=mask_out)


@triton.jit
def pack_ue8m0_kernel_gran1(
    scale_ptr,
    output_ptr,
    M,
    K,
    K_packed,
    stride_scale_b,
    stride_scale_m,
    stride_scale_k,
    stride_out_b,
    stride_out_k_packed,
    stride_out_m,
    BLOCK_M: tl.constexpr,
    BLOCK_K_PACKED: tl.constexpr,
):
    """
    Specialized kernel for gran_mn=1 case (most common).

    When gran_mn=1, each M row maps directly to a scale row,
    allowing for simplified and faster memory access.
    """
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_b = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k_packed = pid_k * BLOCK_K_PACKED + tl.arange(0, BLOCK_K_PACKED)

    # For gran_mn=1: row_idxs = offs_m (no division needed)
    offs_k = offs_k_packed[:, None] * 4 + tl.arange(0, 4)[None, :]

    # Direct pointer computation (row = m)
    scale_ptrs = (
        scale_ptr
        + pid_b * stride_scale_b
        + offs_m[:, None, None] * stride_scale_m
        + offs_k[None, :, :] * stride_scale_k
    )

    mask_m = offs_m < M
    mask_k = offs_k < K
    mask = mask_m[:, None, None] & mask_k[None, :, :]

    vals = tl.load(scale_ptrs, mask=mask, other=0.0)

    # Fast UE8M0 conversion and packing
    exponents = (vals.to(tl.int32, bitcast=True) >> 23) & 0xFF
    shifts = tl.arange(0, 4)[None, None, :] * 8
    packed = tl.sum(exponents << shifts, axis=2).to(tl.int32)

    out_ptrs = (
        output_ptr
        + pid_b * stride_out_b
        + offs_k_packed[None, :] * stride_out_k_packed
        + offs_m[:, None] * stride_out_m
    )

    mask_out = mask_m[:, None] & (offs_k_packed[None, :] < K_packed)
    tl.store(out_ptrs, packed, mask=mask_out)


def pack_ue8m0_kernel_launcher(scale: torch.Tensor, gran_mn: int):
    import deep_gemm

    if scale.dim() == 2:
        scale = scale.unsqueeze(0)
        is_2d = True
    else:
        is_2d = False

    B, M_scale, K = scale.shape
    M = M_scale * gran_mn

    # Calculate aligned dimensions
    aligned_mn = deep_gemm.get_tma_aligned_size(M, 4)
    aligned_k = (K + 3) // 4 * 4
    K_packed = aligned_k // 4

    # Allocate output (Column Major)
    # Storage: (B, K_packed, aligned_mn)
    packed_storage = torch.zeros(
        (B, K_packed, aligned_mn), device=scale.device, dtype=torch.int32
    )
    # View as (B, aligned_mn, K_packed)
    packed = packed_storage.transpose(1, 2)

    BLOCK_M = 64
    BLOCK_K_PACKED = 32

    total_elements = BLOCK_M * BLOCK_K_PACKED
    num_warps = min(max(total_elements // 256, 4), 8)

    # Use software pipelining for better memory latency hiding
    num_stages = 2

    grid = (
        triton.cdiv(M, BLOCK_M),
        triton.cdiv(K_packed, BLOCK_K_PACKED),
        B,
    )

    if gran_mn == 1:
        pack_ue8m0_kernel_gran1[grid](
            scale,
            packed,
            M,
            K,
            K_packed,
            scale.stride(0),
            scale.stride(1),
            scale.stride(2),
            packed.stride(0),
            packed.stride(2),
            packed.stride(1),
            BLOCK_M=BLOCK_M,
            BLOCK_K_PACKED=BLOCK_K_PACKED,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    else:
        # Use vectorized kernel for general case
        pack_ue8m0_kernel_vectorized[grid](
            scale,
            packed,
            M,
            K,
            K_packed,
            scale.stride(0),
            scale.stride(1),
            scale.stride(2),
            packed.stride(0),
            packed.stride(2),
            packed.stride(1),
            gran_mn=gran_mn,
            BLOCK_M=BLOCK_M,
            BLOCK_K_PACKED=BLOCK_K_PACKED,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    res = packed[:, :M, :]
    if is_2d:
        return res.squeeze(0)
    return res


def fp8_gemm_nt(
    a: Tuple[torch.Tensor, torch.Tensor],
    b: Tuple[torch.Tensor, torch.Tensor],
    output: torch.Tensor,
    c: Optional[torch.Tensor] = None,
    compiled_dims: str = "nk",
    disable_ue8m0_cast: Optional[bool] = None,
) -> None:
    """Execute FP8 GEMM (A * B^T).

    Args:
        a (Tuple[torch.Tensor, torch.Tensor]): FP8 data and scales for the first matrix.
        b (Tuple[torch.Tensor, torch.Tensor]): FP8 data and scales for the second matrix.
        output (torch.Tensor): Output tensor.
        c (Optional[torch.Tensor], optional): Optional bias tensor. Defaults to None.
        compiled_dims (str, optional): Compiled dimensions. Defaults to "nk".
        disable_ue8m0_cast (bool, optional): Whether to disable E8M0 type cast for E8M0 scale.
            Defaults to None, which will be set to False if E8M0 scale is used, otherwise True.

    Returns:
        None
    """
    impl = _resolve_deep_gemm_impl("fp8_gemm_nt")
    if impl is None:
        return _missing_deep_gemm()
    if _deep_gemm_impl_uses_full_name.get("fp8_gemm_nt", False):
        return _call_full_name_normal(
            "fp8_gemm_nt", impl, a, b, output, c, compiled_dims
        )
    impl(
        a,
        b,
        output,
        c,
        compiled_dims=compiled_dims,
        # normal gemm tmp not use ue8m0 cast default
        disable_ue8m0_cast=(
            disable_ue8m0_cast if disable_ue8m0_cast is not None else True
        ),
    )


def m_grouped_fp8_gemm_nt_contiguous(
    a: Tuple[torch.Tensor, torch.Tensor],
    b: Tuple[torch.Tensor, torch.Tensor],
    output: torch.Tensor,
    m_indices: torch.Tensor,
    compiled_dims: str = "nk",
    disable_ue8m0_cast: Optional[bool] = None,
) -> None:
    """Execute grouped FP8 GEMM (A * B^T) with contiguous layout.

    Args:
        a (Tuple[torch.Tensor, torch.Tensor]): FP8 data and scales for the first matrix with contiguous layout.
        b (Tuple[torch.Tensor, torch.Tensor]): FP8 data and scales for the second matrix.
        output (torch.Tensor): Output tensor.
        m_indices (torch.Tensor): Grouped indices for valid tokens in each group.
            The length of m_indices is the a[0].shape[0], and the corresponding value of valid tokens is group_idx.
        compiled_dims (str, optional): Compiled dimensions. Defaults to "nk".
        disable_ue8m0_cast (bool, optional): Whether to disable E8M0 type cast for E8M0 scale.
            Defaults to None, which will be set to False if E8M0 scale is used, otherwise True.
    """

    impl = _resolve_deep_gemm_impl("m_grouped_fp8_gemm_nt_contiguous")
    if impl is None:
        return _missing_deep_gemm()
    if _deep_gemm_impl_uses_full_name.get(
        "m_grouped_fp8_gemm_nt_contiguous", False
    ):
        return _call_full_name_grouped_contiguous(
            "m_grouped_fp8_gemm_nt_contiguous",
            impl,
            a,
            b,
            output,
            m_indices,
            compiled_dims,
        )
    impl(
        a,
        b,
        output,
        m_indices,
        compiled_dims=compiled_dims,
        disable_ue8m0_cast=(
            disable_ue8m0_cast
            if disable_ue8m0_cast is not None
            else not is_deep_gemm_e8m0_used()
        ),
    )


def maybe_pack_ue8m0_scale(
    x: torch.Tensor, scale: torch.Tensor, disable_ue8m0_cast: bool
) -> torch.Tensor:
    # check pack conditions:
    # 1. sm=100
    # 2. sf.scalar_type() == torch::kFloat
    # 3. not disable_ue8m0_cast
    # 4. num_groups > 1
    arch_major, _ = torch.cuda.get_device_capability()
    if arch_major != 10:
        return scale
    if scale.dtype != torch.float32:
        return scale
    if disable_ue8m0_cast:
        return scale
    if scale.dim() != 3 or scale.shape[0] < 2:
        return scale

    gran_mn = x.shape[-2] // scale.shape[-2]
    if gran_mn != 1 and gran_mn != 128:
        return scale

    return pack_ue8m0_kernel_launcher(scale, gran_mn)


def m_grouped_fp8_gemm_nt_masked(
    a: Tuple[torch.Tensor, torch.Tensor],
    b: Tuple[torch.Tensor, torch.Tensor],
    output: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m: int,
    compiled_dims: str = "nk",
    disable_ue8m0_cast: Optional[bool] = None,
) -> None:
    """Execute grouped FP8 GEMM (A * B^T) with masked layout.

    Args:
        a (Tuple[torch.Tensor, torch.Tensor]): FP8 data and scales for the first matrix with masked layout.
        b (Tuple[torch.Tensor, torch.Tensor]): FP8 data and scales for the second matrix.
        output (torch.Tensor): Output tensor.
        masked_m (torch.Tensor): the number of valid tokens in each group.
        expected_m (int): Expected number of valid tokens in each group.
        compiled_dims (str, optional): Compiled dimensions. Defaults to "nk".
        disable_ue8m0_cast (bool, optional): Whether to disable E8M0 type cast for E8M0 scale.
            Defaults to None, which will be set to False if E8M0 scale is used, otherwise True.
    """
    impl = _resolve_deep_gemm_impl("m_grouped_fp8_gemm_nt_masked")
    if impl is None:
        return _missing_deep_gemm()
    if _deep_gemm_impl_uses_full_name.get(
        "m_grouped_fp8_gemm_nt_masked", False
    ):
        return _call_full_name_grouped_masked(
            "m_grouped_fp8_gemm_nt_masked",
            impl,
            a,
            b,
            output,
            masked_m,
            expected_m,
            compiled_dims,
        )

    disable_ue8m0_cast = (
        disable_ue8m0_cast
        if disable_ue8m0_cast is not None
        else not is_deep_gemm_e8m0_used()
    )

    a = (a[0], maybe_pack_ue8m0_scale(a[0], a[1], disable_ue8m0_cast))
    b = (b[0], maybe_pack_ue8m0_scale(b[0], b[1], disable_ue8m0_cast))

    impl(
        a,
        b,
        output,
        masked_m,
        expected_m,
        compiled_dims=compiled_dims,
        disable_ue8m0_cast=disable_ue8m0_cast,
    )


def bf16_gemm_nt(
    a: torch.Tensor,
    b: torch.Tensor,
    output: torch.Tensor,
    c: Optional[torch.Tensor] = None,
    compiled_dims: str = "nk",
) -> None:
    """Execute BF16 GEMM (A * B^T).

    Args:
        a (torch.Tensor): BF16 data for the first matrix.
        b (torch.Tensor): BF16 data for the second matrix.
        output (torch.Tensor): Output tensor.
        c (Optional[torch.Tensor], optional): Optional bias tensor. Defaults to None.
        compiled_dims (str, optional): Compiled dimensions. Defaults to "nk".
    """
    impl = _resolve_deep_gemm_impl("bf16_gemm_nt")
    if impl is None:
        return _missing_deep_gemm()
    if _deep_gemm_impl_uses_full_name.get("bf16_gemm_nt", False):
        return _call_full_name_normal(
            "bf16_gemm_nt", impl, a, b, output, c, compiled_dims
        )
    impl(a, b, output, c, compiled_dims)


def m_grouped_bf16_gemm_nt_contiguous(
    a: torch.Tensor,
    b: torch.Tensor,
    output: torch.Tensor,
    m_indices: torch.Tensor,
    compiled_dims: str = "nk",
) -> None:
    """Execute grouped BF16 GEMM (A * B^T) with contiguous layout.

    Args:
        a (torch.Tensor): BF16 data for the first matrix with contiguous layout.
        b (torch.Tensor): BF16 data for the second matrix.
        output (torch.Tensor): Output tensor.
        m_indices (torch.Tensor): Grouped indices for valid tokens in each group.
            The length of m_indices is the a.shape[0], and the corresponding value of valid tokens is group_idx.
        compiled_dims (str, optional): Compiled dimensions. Defaults to "nk".
    """
    impl = _resolve_deep_gemm_impl("m_grouped_bf16_gemm_nt_contiguous")
    if impl is None:
        return _missing_deep_gemm()
    if _deep_gemm_impl_uses_full_name.get(
        "m_grouped_bf16_gemm_nt_contiguous", False
    ):
        return _call_full_name_grouped_contiguous(
            "m_grouped_bf16_gemm_nt_contiguous",
            impl,
            a,
            b,
            output,
            m_indices,
            compiled_dims,
        )
    impl(a, b, output, m_indices, compiled_dims)


def m_grouped_bf16_gemm_nt_masked(
    a: torch.Tensor,
    b: torch.Tensor,
    output: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m: int,
    compiled_dims: str = "nk",
) -> None:
    """Execute grouped BF16 GEMM (A * B^T) with masked layout.

    Args:
        a (torch.Tensor): BF16 data for the first matrix with masked layout.
        b (torch.Tensor): BF16 data for the second matrix.
        output (torch.Tensor): Output tensor.
        masked_m (torch.Tensor): the number of valid tokens in each group.
        expected_m (int): Expected number of valid tokens in each group.
        compiled_dims (str, optional): Compiled dimensions. Defaults to "nk".
    """
    impl = _resolve_deep_gemm_impl("m_grouped_bf16_gemm_nt_masked")
    if impl is None:
        return _missing_deep_gemm()
    if _deep_gemm_impl_uses_full_name.get(
        "m_grouped_bf16_gemm_nt_masked", False
    ):
        return _call_full_name_grouped_masked(
            "m_grouped_bf16_gemm_nt_masked",
            impl,
            a,
            b,
            output,
            masked_m,
            expected_m,
            compiled_dims,
        )
    impl(a, b, output, masked_m, expected_m, compiled_dims)


def _require_sm100_packed_scale_for_fp8_fp4(
    a: Tuple[torch.Tensor, torch.Tensor],
    b: Tuple[torch.Tensor, torch.Tensor],
) -> None:
    if not torch.cuda.is_available():
        return
    arch_major, _ = torch.cuda.get_device_capability()
    if arch_major != 10 or not is_deep_gemm_e8m0_used():
        return
    if a[1].dtype != torch.int32 or b[1].dtype != torch.int32:
        raise RuntimeError(
            "SM100 FP8xFP4 DeepGEMM calls require prepacked int32 scales; "
            f"got a_scale={a[1].dtype}, b_scale={b[1].dtype}"
        )


# --- FP8 activation × FP4 weight (UE8M0 block-scale) ---
#
# DeepGEMM's fp8_fp4 family consumes packed-int8 FP4 weights (2 FP4/byte)
# with UE8M0 block-32 scale along K, matching the DeepSeek-native FP4
# recipe shipped by V3.2/V4 routed experts. Activation is FP8 e4m3fn with
# per-token block-128 UE8M0 scale. SM100 only.


def fp8_fp4_gemm_nt(
    a: Tuple[torch.Tensor, torch.Tensor],
    b: Tuple[torch.Tensor, torch.Tensor],
    output: torch.Tensor,
    c: Optional[torch.Tensor] = None,
    recipe: Optional[Tuple[int, int, int]] = None,
    recipe_a: Optional[Tuple[int, int]] = None,
    recipe_b: Optional[Tuple[int, int]] = None,
    compiled_dims: str = "nk",
    disable_ue8m0_cast: Optional[bool] = None,
) -> None:
    """Dense FP8-act × packed-FP4-weight GEMM with UE8M0 block scales."""
    impl = _resolve_deep_gemm_impl("fp8_fp4_gemm_nt")
    if impl is None:
        return _missing_deep_gemm()
    _require_sm100_packed_scale_for_fp8_fp4(a, b)
    impl(
        a,
        b,
        output,
        c,
        recipe=recipe,
        recipe_a=recipe_a,
        recipe_b=recipe_b,
        compiled_dims=compiled_dims,
        disable_ue8m0_cast=(
            disable_ue8m0_cast
            if disable_ue8m0_cast is not None
            else not is_deep_gemm_e8m0_used()
        ),
    )


def m_grouped_fp8_fp4_gemm_nt_contiguous(
    a: Tuple[torch.Tensor, torch.Tensor],
    b: Tuple[torch.Tensor, torch.Tensor],
    output: torch.Tensor,
    grouped_layout: torch.Tensor,
    recipe: Optional[Tuple[int, int, int]] = None,
    recipe_a: Optional[Tuple[int, int]] = None,
    recipe_b: Optional[Tuple[int, int]] = None,
    compiled_dims: str = "nk",
    disable_ue8m0_cast: Optional[bool] = None,
    use_psum_layout: bool = False,
    expected_m_for_psum_layout: Optional[int] = None,
) -> None:
    """Grouped FP8×FP4 GEMM with contiguous per-expert token layout.

    `grouped_layout` is a `[num_tokens]` int tensor: `grouped_layout[i]` is
    the expert index owning token `i`. Tokens must already be permuted so
    each expert's rows are contiguous.
    """
    impl = _resolve_deep_gemm_impl("m_grouped_fp8_fp4_gemm_nt_contiguous")
    if impl is None:
        return _missing_deep_gemm()
    _require_sm100_packed_scale_for_fp8_fp4(a, b)
    impl(
        a,
        b,
        output,
        grouped_layout,
        recipe=recipe,
        recipe_a=recipe_a,
        recipe_b=recipe_b,
        compiled_dims=compiled_dims,
        disable_ue8m0_cast=(
            disable_ue8m0_cast
            if disable_ue8m0_cast is not None
            else not is_deep_gemm_e8m0_used()
        ),
        use_psum_layout=use_psum_layout,
        expected_m_for_psum_layout=expected_m_for_psum_layout,
    )


def m_grouped_fp8_fp4_gemm_nt_masked(
    a: Tuple[torch.Tensor, torch.Tensor],
    b: Tuple[torch.Tensor, torch.Tensor],
    output: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m: int,
    recipe: Optional[Tuple[int, int, int]] = None,
    recipe_a: Optional[Tuple[int, int]] = None,
    recipe_b: Optional[Tuple[int, int]] = None,
    compiled_dims: str = "nk",
    disable_ue8m0_cast: Optional[bool] = None,
) -> None:
    """Grouped FP8×FP4 GEMM with masked layout (data-dependent per-expert
    token counts; avoids a D2H sync, suitable for decode)."""
    impl = _resolve_deep_gemm_impl("m_grouped_fp8_fp4_gemm_nt_masked")
    if impl is None:
        return _missing_deep_gemm()
    _require_sm100_packed_scale_for_fp8_fp4(a, b)
    impl(
        a,
        b,
        output,
        masked_m,
        expected_m,
        recipe=recipe,
        recipe_a=recipe_a,
        recipe_b=recipe_b,
        compiled_dims=compiled_dims,
        disable_ue8m0_cast=(
            disable_ue8m0_cast
            if disable_ue8m0_cast is not None
            else not is_deep_gemm_e8m0_used()
        ),
    )


def fp8_fp4_paged_mqa_logits(
    q: Tuple[torch.Tensor, Optional[torch.Tensor]],
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_table: torch.Tensor,
    schedule_meta: torch.Tensor,
    max_context_len: int,
    clean_logits: bool = False,
    logits_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """FP8-query × FP4-packed paged-KV MQA logits — same kernel V3.2 DSA
    uses for the lightning indexer score step."""
    impl = _resolve_deep_gemm_impl("fp8_fp4_paged_mqa_logits")
    if impl is None:
        return _missing_deep_gemm()
    return impl(
        q,
        kv_cache,
        weights,
        context_lens,
        block_table,
        schedule_meta,
        max_context_len,
        clean_logits,
        logits_dtype,
    )


def per_token_cast_to_fp4(*args: Any, **kwargs: Any) -> Any:
    """DeepGEMM helper: cast BF16 activations to packed-FP4 + UE8M0 scale.
    Thin passthrough — signature/kwargs owned by deep_gemm."""
    impl = _resolve_deep_gemm_impl("per_token_cast_to_fp4")
    if impl is None:
        return _missing_deep_gemm()
    return impl(*args, **kwargs)


def cast_back_from_fp4(*args: Any, **kwargs: Any) -> Any:
    """DeepGEMM helper: dequant packed-FP4 → BF16 (debug/inspection path)."""
    impl = _resolve_deep_gemm_impl("cast_back_from_fp4")
    if impl is None:
        return _missing_deep_gemm()
    return impl(*args, **kwargs)


def transpose_packed_fp4(*args: Any, **kwargs: Any) -> Any:
    """DeepGEMM helper: transpose a packed-FP4 weight in its int8 storage,
    preserving nibble ordering."""
    impl = _resolve_deep_gemm_impl("transpose_packed_fp4")
    if impl is None:
        return _missing_deep_gemm()
    return impl(*args, **kwargs)


def tf32_hc_prenorm_gemm(
    x: torch.Tensor,
    fn: torch.Tensor,
    out: torch.Tensor,
    sqrsum: torch.Tensor,
    num_split: int,
) -> torch.Tensor:
    """DeepGEMM mHC pre helper: out=x.float()@fn.T and sqrsum=x^2.

    This symbol is optional in DeepGEMM, so it is not part of the eager
    wrapper initialization used by the unrelated GEMM paths.
    """
    impl = _resolve_deep_gemm_impl("tf32_hc_prenorm_gemm")
    if impl is None:
        return _missing_deep_gemm()
    return impl(x, fn, out, sqrsum, num_split)
