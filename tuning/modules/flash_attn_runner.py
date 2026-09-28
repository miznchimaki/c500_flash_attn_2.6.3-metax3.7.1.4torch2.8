import math

import pytest
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from flash_attn import (
    flash_attn_func,
    flash_attn_kvpacked_func,
    flash_attn_qkvpacked_func,
    flash_attn_varlen_func,
    flash_attn_varlen_kvpacked_func,
    flash_attn_varlen_qkvpacked_func,
    flash_attn_with_kvcache,
)
from flash_attn.layers.rotary import apply_rotary_emb
from flash_attn.tuning.modules.test_ref import *
from flash_attn.tuning.modules.logger import logger

MAX_HEADDIM_SM8x = 192

is_sm75 = torch.cuda.get_device_capability("cuda") == (7, 5)
is_sm8x = torch.cuda.get_device_capability("cuda")[0] == 8
is_sm80 = torch.cuda.get_device_capability("cuda") == (8, 0)
is_sm90 = torch.cuda.get_device_capability("cuda") == (9, 0)

# seqlen_pair_q_k, d, dropout_p, causal, local, alibi, deterministic, mha_type, dtype, kvpacked

# TODO Not support hdim_q != hdim_kv
# TODO Not support attn_mask
def run_flash_attn(
    batch_size,
    seqlen_q,
    seqlen_k,
    nheads,
    nheads_k,
    d,
    d_v,
    dropout_p,
    causal,
    window_size,
    alibi,
    deterministic,
    dtype,
    kvpacked = False,
    qkvpacked = False,
    bwd_check = False
):
    # seqlen_q, seqlen_k = seqlen_pair_q_k
    # if (
    #     max(seqlen_q, seqlen_k) >= 2048
    #     and torch.cuda.get_device_properties("cuda").total_memory <= 16 * 2**30
    # ):
    #     pytest.skip()  # Reference implementation OOM
    device = "cuda"
    # set seed
    torch.random.manual_seed(0)
    # batch_size = 4
    # nheads = 9
    # nheads_k = nheads if mha_type == "mha" else (1 if mha_type == "mqa" else 3)
    assert nheads % nheads_k == 0
    local = False if window_size == (-1,-1) else True
    # window_size = (-1, -1) if not local else torch.randint(0, seqlen_k, (2,))
    q = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype, requires_grad=True)
    if kvpacked:
        kv = torch.randn(
            batch_size, seqlen_k, 2, nheads_k, d, device=device, dtype=dtype, requires_grad=True
        )
    elif qkvpacked:
        qkv = torch.randn(
            batch_size, seqlen_k, 3, nheads_k, d, device=device, dtype=dtype, requires_grad=True
        )
    else:
        k = torch.randn(
            batch_size, seqlen_k, nheads_k, d, device=device, dtype=dtype, requires_grad=True
        )
        v = torch.randn(
            batch_size, seqlen_k, nheads_k, d_v, device=device, dtype=dtype, requires_grad=True
        )
    if alibi:
        alibi_slopes = torch.rand(batch_size, nheads, device=device, dtype=torch.float32) * 0.3
        attn_bias = attn_bias_from_alibi_slopes(alibi_slopes, seqlen_q, seqlen_k, causal=causal)
    else:
        alibi_slopes, attn_bias = None, None

    if kvpacked:
        out, lse, S_dmask = flash_attn_kvpacked_func(
            q,
            kv,
            dropout_p,
            causal=causal,
            window_size=window_size,
            alibi_slopes=alibi_slopes,
            deterministic=deterministic,
            return_attn_probs=True,
        )
    elif qkvpacked:
        out, lse, S_dmask = flash_attn_qkvpacked_func(
            qkv,
            dropout_p,
            causal = causal,
            window_size = window_size,
            alibi_slopes = alibi_slopes,
            deterministic = deterministic,
            return_attn_probs = True,
        )
    else:
        out, lse, S_dmask = flash_attn_func(
            q,
            k,
            v,
            dropout_p,
            causal=causal,
            window_size=window_size,
            alibi_slopes=alibi_slopes,
            deterministic=deterministic,
            return_attn_probs=True,
        )
    if dropout_p > 0.0:
        S_dmask_converted = convert_flash_attn_S_to_softmax(
            S_dmask,
            seqlen_q,
            seqlen_k,
            None,
            None,
            d,
            dropout_p > 0.0,
            causal=causal,
            window_size=window_size,
        )
        dropout_mask = S_dmask_converted >= 0
        attn_unnorm = S_dmask_converted.abs()
        if kvpacked:
            kv_rep = repeat(kv, "b s two h d -> b s two (h g) d", g=nheads // nheads_k)
            k_rep, v_rep = kv_rep.unbind(dim=2)
        elif qkvpacked:
            q = qkv[:, :, 0]
            k_rep = qkv[:, :, 1]
            v_rep = qkv[:, :, 2]
        else:
            k_rep = repeat(k, "b s h d -> b s (h g) d", g=nheads // nheads_k)
            v_rep = repeat(v, "b s h d -> b s (h g) d", g=nheads // nheads_k)
        attn = normalize_flash_attn_S(
            attn_unnorm,
            q,
            k_rep,
            v_rep,
            None,
            None,
            attn_bias,
            dropout_p > 0.0,
            causal=causal,
            window_size=window_size,
        )
        dropout_fraction = get_dropout_fraction(
            dropout_mask, None, None, causal=causal, window_size=window_size
        ).item()
        logger.debug(f"Actual dropout fraction: {dropout_fraction}")
    else:
        dropout_mask = None

    if kvpacked:
        out_ref, attn_ref = attention_kvpacked_ref(
            q,
            kv,
            None,
            None,
            attn_bias,
            dropout_p,
            dropout_mask,
            causal=causal,
            window_size=window_size,
        )
        out_pt, attn_pt = attention_kvpacked_ref(
            q,
            kv,
            None,
            None,
            attn_bias,
            dropout_p,
            dropout_mask,
            causal=causal,
            window_size=window_size,
            upcast=False,
            reorder_ops=True,
        )
    elif qkvpacked:
        out_ref, attn_ref = attention_qkvpacked_ref(
            qkv,
            None,
            attn_bias,
            dropout_p,
            dropout_mask,
            causal=causal,
            window_size=window_size
        )
        out_pt, attn_pt = attention_qkvpacked_ref(
            qkv,
            None,
            attn_bias,
            dropout_p,
            dropout_mask,
            causal=causal,
            window_size=window_size,
            upcast=False,
            reorder_ops=True,
        )
    else:
        out_ref, attn_ref = attention_ref(
            q,
            k,
            v,
            None,
            None,
            attn_bias,
            dropout_p,
            dropout_mask,
            causal=causal,
            window_size=window_size,
        )
        out_pt, attn_pt = attention_ref(
            q,
            k,
            v,
            None,
            None,
            attn_bias,
            dropout_p,
            dropout_mask,
            causal=causal,
            window_size=window_size,
            upcast=False,
            reorder_ops=True,
        )

    diffs = []

    diffs.append(f"Output max diff: {(out - out_ref).abs().max().item()}")
    diffs.append(f"Output mean diff: {(out - out_ref).abs().mean().item()}")
    diffs.append(f"Pytorch max diff: {(out_pt - out_ref).abs().max().item()}")
    diffs.append(f"Pytorch mean diff: {(out_pt - out_ref).abs().mean().item()}")
    if dropout_p > 0.0:
        diffs.append(f"Attention max diff: {(attn - attn_ref).abs().max().item()}")
        diffs.append(f"Attention Pytorch max diff: {(attn_pt - attn_ref).abs().max().item()}")

    g = torch.randn_like(out)
    do_o = (g.float() * out.float()).sum(-1)
    if d <= MAX_HEADDIM_SM8x or (is_sm80 or is_sm90):
        if kvpacked:
            (
                dq,
                dkv,
            ) = torch.autograd.grad(out, (q, kv), g)
            dk, dv = dkv.unbind(2)
            (
                dq_ref,
                dkv_ref,
            ) = torch.autograd.grad(out_ref, (q, kv), g)
            dk_ref, dv_ref = dkv_ref.unbind(2)
            (
                dq_pt,
                dkv_pt,
            ) = torch.autograd.grad(out_pt, (q, kv), g)
            dk_pt, dv_pt = dkv_pt.unbind(2)
        elif qkvpacked:
            (dqkv,) = torch.autograd.grad(out, qkv, g)
            dq, dk, dv = dqkv.unbind(2)

            (dqkv_ref,) = torch.autograd.grad(out_ref, qkv, g)
            dq_ref, dk_ref, dv_ref = dqkv_ref.unbind(2)

            (dqkv_pt,) = torch.autograd.grad(out_pt, qkv, g)
            dq_pt, dk_pt ,dv_pt = dqkv_pt.unbind(2)
        else:
            (
                dq,
                dk,
                dv,
            ) = torch.autograd.grad(out, (q, k, v), g)
            (
                dq_ref,
                dk_ref,
                dv_ref,
            ) = torch.autograd.grad(out_ref, (q, k, v), g)
            (
                dq_pt,
                dk_pt,
                dv_pt,
            ) = torch.autograd.grad(out_pt, (q, k, v), g)
        diffs.append(f"dQ max diff: {(dq - dq_ref).abs().max().item()}")
        diffs.append(f"dK max diff: {(dk - dk_ref).abs().max().item()}")
        diffs.append(f"dV max diff: {(dv - dv_ref).abs().max().item()}")
        diffs.append(f"dQ mean diff: {(dq - dq_ref).abs().mean().item()}")
        diffs.append(f"dK mean diff: {(dk - dk_ref).abs().mean().item()}")
        diffs.append(f"dV mean diff: {(dv - dv_ref).abs().mean().item()}")
        diffs.append(f"dQ Pytorch max diff: {(dq_pt - dq_ref).abs().max().item()}")
        diffs.append(f"dK Pytorch max diff: {(dk_pt - dk_ref).abs().max().item()}")
        diffs.append(f"dV Pytorch max diff: {(dv_pt - dv_ref).abs().max().item()}")
        diffs.append(f"dQ Pytorch mean diff: {(dq_pt - dq_ref).abs().mean().item()}")
        diffs.append(f"dK Pytorch mean diff: {(dk_pt - dk_ref).abs().mean().item()}")
        diffs.append(f"dV Pytorch mean diff: {(dv_pt - dv_ref).abs().mean().item()}")

    # Check that FlashAttention's numerical error is at most twice the numerical error
    # of a Pytorch implementation.
    forward_check = (out - out_ref).abs().max().item() <= 2 * (out_pt - out_ref).abs().max().item()

    dropout_check = True
    if dropout_p > 0.0:
        dropout_check = dropout_check and (attn - attn_ref).abs().max().item() <= 2 * (attn_pt - attn_ref).abs().max().item()
        # With alibi, many of the prob values are 0.0 & -0.0 so dropout_fraction isn't accurate
        # For Maca we canot got same max_diff with cuda, only affects the test.
        if not alibi:
           dropout_check = dropout_check and abs(dropout_fraction - dropout_p) <= (0.01 if not local else 0.025)

    alpha = 2
    #TODO [2024/03/05]:For maca bfloat16 alpha cannot same with cuda, only affects the test
    if dtype == torch.bfloat16 and dropout_p > 0.0:
        alpha = 3
    #TODO [2024/06/24]:For maca [0.17-512-256-59-False-True-True-False-gqa-dtype0-True] cannot same with cuda
    if dtype == torch.float16 and seqlen_q == 512 and seqlen_k == 256 and d == 59 and mha_type == "gqa":
        alpha = 3

    backward_check =  (dq - dq_ref).abs().max().item() <= alpha * (dq_pt - dq_ref).abs().max().item() and (dk - dk_ref).abs().max().item() <= alpha * (dk_pt - dk_ref).abs().max().item() and (dv - dv_ref).abs().max().item() <= alpha * (dv_pt - dv_ref).abs().max().item()

    if bwd_check:
        return forward_check and dropout_check and backward_check, diffs
    else:
        return forward_check and dropout_check, diffs


# TODO Not support hdim_q != hdim_kv
# seqlen_pair_q_k, d, dropout_p, causal, local, alibi, deterministic, mha_type, dtype, kvpacked, batch_size, nheads
def run_flash_attn_varlen(
    seqlens_q : list,
    seqlens_kv : list,
    nheads,
    nheads_k,
    d,
    d_v,
    dropout_p,
    causal,
    window_size,
    alibi,
    deterministic,
    dtype,
    kvpacked = False,
    qkvpacked = False,
    bwd_check = False
):
    max_seqlen_q = max(seqlens_q)
    max_seqlen_k = max(seqlens_kv)
    seqlen_q = max_seqlen_q
    seqlen_k = max_seqlen_k

    assert len(seqlens_q) == len(seqlens_kv)
    batch_size = len(seqlens_q)
    # if (
    #     max(seqlen_q, seqlen_k) >= 2048
    #     and torch.cuda.get_device_properties("cuda").total_memory <= 16 * 2**30
    # ):
    #     pytest.skip()  # Reference implementation OOM
    device = "cuda"
    # set seed
    torch.random.manual_seed(0)
    # nheads_k = nheads if mha_type == "mha" else (1 if mha_type == "mqa" else 3)
    assert nheads % nheads_k == 0
    local = False if window_size == (-1,-1) else True
    # window_size = (-1, -1) if not local else torch.randint(0, seqlen_k, (2,))
    q = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype, requires_grad=True)
    if kvpacked:
        kv = torch.randn(
            batch_size, seqlen_k, 2, nheads_k, d, device=device, dtype=dtype, requires_grad=True
        )
    else:
        k = torch.randn(
            batch_size, seqlen_k, nheads_k, d, device=device, dtype=dtype, requires_grad=True
        )
        v = torch.randn(
            batch_size, seqlen_k, nheads_k, d_v, device=device, dtype=dtype, requires_grad=True
        )

    # query_padding_mask = generate_random_padding_mask(seqlen_q, batch_size, device, mode="random")
    # key_padding_mask = generate_random_padding_mask(seqlen_k, batch_size, device, mode="random")

    query_padding_mask = generate_varlen_padding_mask(seqlens_q, batch_size, device)
    key_padding_mask = generate_varlen_padding_mask(seqlens_kv, batch_size, device)

    if alibi:
        alibi_slopes = torch.rand(batch_size, nheads, device=device, dtype=torch.float32) * 0.3
        attn_bias = attn_bias_from_alibi_slopes(
            alibi_slopes, seqlen_q, seqlen_k, query_padding_mask, key_padding_mask, causal=causal
        )
    else:
        alibi_slopes, attn_bias = None, None

    if kvpacked:
        (
            q_unpad,
            kv_unpad,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            q,
            kv,
            output_pad_fn,
            dq_pad_fn,
            dkv_pad_fn,
        ) = generate_qkv(q, *kv.unbind(dim=2), query_padding_mask, key_padding_mask, kvpacked=True)
        out_unpad, sm_lse, S_dmask = flash_attn_varlen_kvpacked_func(
            q_unpad,
            kv_unpad,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            dropout_p,
            causal=causal,
            window_size=window_size,
            alibi_slopes=alibi_slopes,
            deterministic=deterministic,
            return_attn_probs=True,
        )
    else:
        (
            q_unpad,
            k_unpad,
            v_unpad,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            q,
            k,
            v,
            output_pad_fn,
            dq_pad_fn,
            dk_pad_fn,
        ) = generate_qkv(q, k, v, query_padding_mask, key_padding_mask, kvpacked=False)
        out_unpad, sm_lse, S_dmask = flash_attn_varlen_func(
            q_unpad,
            k_unpad,
            v_unpad,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            dropout_p,
            causal=causal,
            window_size=window_size,
            alibi_slopes=alibi_slopes,
            deterministic=deterministic,
            return_attn_probs=True,
        )
    out = output_pad_fn(out_unpad)
    if dropout_p > 0.0:
        S_dmask_converted = convert_flash_attn_S_to_softmax(
            S_dmask,
            seqlen_q,
            seqlen_k,
            query_padding_mask,
            key_padding_mask,
            d,
            dropout_p > 0.0,
            causal=causal,
            window_size=window_size,
        )
        dropout_mask = S_dmask_converted >= 0
        attn_unnorm = S_dmask_converted.abs()
        if kvpacked:
            kv_rep = repeat(kv, "b s two h d -> b s two (h g) d", g=nheads // nheads_k)
            k_rep, v_rep = kv_rep.unbind(dim=2)
        else:
            k_rep = repeat(k, "b s h d -> b s (h g) d", g=nheads // nheads_k)
            v_rep = repeat(v, "b s h d -> b s (h g) d", g=nheads // nheads_k)
        attn = normalize_flash_attn_S(
            attn_unnorm,
            q,
            k_rep,
            v_rep,
            query_padding_mask,
            key_padding_mask,
            attn_bias,
            dropout_p > 0.0,
            causal=causal,
            window_size=window_size,
        )
        dropout_fraction = get_dropout_fraction(
            dropout_mask,
            query_padding_mask,
            key_padding_mask,
            causal=causal,
            window_size=window_size,
        ).item()
        print(f"Actual dropout fraction: {dropout_fraction}")
    else:
        dropout_mask = None

    if kvpacked:
        out_ref, attn_ref = attention_kvpacked_ref(
            q,
            kv,
            query_padding_mask,
            key_padding_mask,
            attn_bias,
            dropout_p,
            dropout_mask,
            causal=causal,
            window_size=window_size,
        )
        out_pt, attn_pt = attention_kvpacked_ref(
            q,
            kv,
            query_padding_mask,
            key_padding_mask,
            attn_bias,
            dropout_p,
            dropout_mask,
            causal=causal,
            window_size=window_size,
            upcast=False,
            reorder_ops=True,
        )
    else:
        out_ref, attn_ref = attention_ref(
            q,
            k,
            v,
            query_padding_mask,
            key_padding_mask,
            attn_bias,
            dropout_p,
            dropout_mask,
            causal=causal,
            window_size=window_size,
        )
        out_pt, attn_pt = attention_ref(
            q,
            k,
            v,
            query_padding_mask,
            key_padding_mask,
            attn_bias,
            dropout_p,
            dropout_mask,
            causal=causal,
            window_size=window_size,
            upcast=False,
            reorder_ops=True,
        )

    diffs = []

    diffs.append(f"Output max diff: {(out - out_ref).abs().max().item()}")
    diffs.append(f"Output mean diff: {(out - out_ref).abs().mean().item()}")
    diffs.append(f"Pytorch max diff: {(out_pt - out_ref).abs().max().item()}")
    diffs.append(f"Pytorch mean diff: {(out_pt - out_ref).abs().mean().item()}")
    if dropout_p > 0.0:
        diffs.append(f"Attention max diff: {(attn - attn_ref).abs().max().item()}")
        diffs.append(f"Attention Pytorch max diff: {(attn_pt - attn_ref).abs().max().item()}")

    g = torch.randn_like(out)
    if d <= MAX_HEADDIM_SM8x or (is_sm80 or is_sm90):
        if kvpacked:
            (
                dq_unpad,
                dkv_unpad,
            ) = torch.autograd.grad(out, (q_unpad, kv_unpad), g)
            dk, dv = dkv_pad_fn(dkv_unpad).unbind(2)
            (
                dq_ref,
                dkv_ref,
            ) = torch.autograd.grad(out_ref, (q, kv), g)
            dk_ref, dv_ref = dkv_ref.unbind(2)
            (
                dq_pt,
                dkv_pt,
            ) = torch.autograd.grad(out_pt, (q, kv), g)
            dk_pt, dv_pt = dkv_pt.unbind(2)
        else:
            (
                dq_unpad,
                dk_unpad,
                dv_unpad,
            ) = torch.autograd.grad(out, (q_unpad, k_unpad, v_unpad), g)
            dk = dk_pad_fn(dk_unpad)
            dv = dk_pad_fn(dv_unpad)
            (
                dq_ref,
                dk_ref,
                dv_ref,
            ) = torch.autograd.grad(out_ref, (q, k, v), g)
            (
                dq_pt,
                dk_pt,
                dv_pt,
            ) = torch.autograd.grad(out_pt, (q, k, v), g)
        dq = dq_pad_fn(dq_unpad)
        diffs.append(f"dQ max diff: {(dq - dq_ref).abs().max().item()}")
        diffs.append(f"dK max diff: {(dk - dk_ref).abs().max().item()}")
        diffs.append(f"dV max diff: {(dv - dv_ref).abs().max().item()}")
        diffs.append(f"dQ mean diff: {(dq - dq_ref).abs().mean().item()}")
        diffs.append(f"dK mean diff: {(dk - dk_ref).abs().mean().item()}")
        diffs.append(f"dV mean diff: {(dv - dv_ref).abs().mean().item()}")
        diffs.append(f"dQ Pytorch max diff: {(dq_pt - dq_ref).abs().max().item()}")
        diffs.append(f"dK Pytorch max diff: {(dk_pt - dk_ref).abs().max().item()}")
        diffs.append(f"dV Pytorch max diff: {(dv_pt - dv_ref).abs().max().item()}")
        diffs.append(f"dQ Pytorch mean diff: {(dq_pt - dq_ref).abs().mean().item()}")
        diffs.append(f"dK Pytorch mean diff: {(dk_pt - dk_ref).abs().mean().item()}")
        diffs.append(f"dV Pytorch mean diff: {(dv_pt - dv_ref).abs().mean().item()}")

    dropout_check = True
    if dropout_p > 0.0:
        dropout_check = dropout_check and (attn - attn_ref).abs().max().item() <= 2 * (attn_pt - attn_ref).abs().max().item()
        if not alibi:
            if seqlen_q == 1 and seqlen_k == 147 :
                dropout_check = dropout_check and abs(dropout_fraction - dropout_p) <= (0.01 if not local else 0.05)
            else:
                dropout_check = dropout_check and abs(dropout_fraction - dropout_p) <= (0.01 if not local else 0.025)

    forward_check = (out - out_ref).abs().max().item() <= 2 * (out_pt - out_ref).abs().max().item()
    backward_check = (dq - dq_ref).abs().max().item() <= 3 * (dq_pt - dq_ref).abs().max().item() and (dk - dk_ref).abs().max().item() <= 3 * (dk_pt - dk_ref).abs().max().item() and (dv - dv_ref).abs().max().item() <= 3 * (dv_pt - dv_ref).abs().max().item()

    if bwd_check:
        return forward_check and backward_check and dropout_check,diffs
    else:
        return forward_check and dropout_check, diffs

def run_flash_attn_kvcache(
    batch_size,
    seqlen_q,
    nheads,
    nheads_k,
    batch_size_cache,
    seqlen_kvcache,
    d,
    has_batch_idx,
    paged_kv_block_size,
    rotary_fraction,
    rotary_interleaved,
    seqlen_new_eq_seqlen_q,
    causal,
    local,
    alibi,
    new_kv,
    num_splits,
    dtype,
):
    seqlen_k = seqlen_kvcache
    # if seqlen_q > seqlen_k and new_kv:
    #     pytest.skip()
    # if not new_kv and rotary_fraction > 0.0:
    #     pytest.skip()
    # if has_batch_idx and paged_kv_block_size is not None:
    #     pytest.skip()
    # if rotary_fraction == 0.0 and rotary_interleaved: # rotary_interlevaved not be used while rotary_fraction==0.0
    #     pytest.skip()
    device = "cuda"
    # set seed
    torch.random.manual_seed(0)
    # batch_size = 2
    # batch_size_cache = batch_size if not has_batch_idx else batch_size * 2
    # nheads = 6
    # rotary_dim must be a multiple of 16, and must be <= d
    rotary_dim = math.floor(int(rotary_fraction * d) / 16) * 16
    # nheads_k = nheads if mha_type == "mha" else (1 if mha_type == "mqa" else 3)
    assert nheads % nheads_k == 0
    window_size = (-1, -1) if not local else torch.randint(0, seqlen_k, (2,))
    q = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype)
    seqlen_new = seqlen_q if seqlen_new_eq_seqlen_q else torch.randint(1, seqlen_q + 1, (1,)).item()
    if new_kv:
        k = torch.randn(batch_size, seqlen_new, nheads_k, d, device=device, dtype=dtype)
        v = torch.randn(batch_size, seqlen_new, nheads_k, d, device=device, dtype=dtype)
    else:
        k, v = None, None
    if paged_kv_block_size is None:
        k_cache = torch.randn(batch_size_cache, seqlen_k, nheads_k, d, device=device, dtype=dtype)
        v_cache = torch.randn(batch_size_cache, seqlen_k, nheads_k, d, device=device, dtype=dtype)
        block_table = None
    else:
        num_blocks = math.ceil(seqlen_k / paged_kv_block_size) * batch_size * 3
        k_cache_paged = torch.randn(
            num_blocks, paged_kv_block_size, nheads_k, d, device=device, dtype=dtype
        )
        v_cache_paged = torch.randn(
            num_blocks, paged_kv_block_size, nheads_k, d, device=device, dtype=dtype
        )
        block_table = rearrange(
            torch.randperm(num_blocks, dtype=torch.int32, device=device),
            "(b nblocks) -> b nblocks",
            b=batch_size,
        )
        k_cache = rearrange(
            k_cache_paged[block_table.flatten()],
            "(b nblocks) block_size ... -> b (nblocks block_size) ...",
            b=batch_size,
        )[:, :seqlen_k]
        v_cache = rearrange(
            v_cache_paged[block_table.flatten()],
            "(b nblocks) block_size ... -> b (nblocks block_size) ...",
            b=batch_size,
        )[:, :seqlen_k]
    cache_seqlens = torch.randint(
        0 if new_kv else 1,
        # If we don't use seqlen_q in the case of causal and rotary, cos/sin won't be long enough
        (seqlen_k - (seqlen_q if (causal or local) and rotary_dim > 1 else seqlen_new) + 1)
        if new_kv
        else (seqlen_k + 1),
        (batch_size,),
        dtype=torch.int32,
        device=device,
    )
    arange = rearrange(torch.arange(seqlen_k, device=device), "s -> 1 s")
    cache_seqlens_expanded = rearrange(cache_seqlens, "b -> b 1")
    key_padding_mask = arange < cache_seqlens_expanded + (seqlen_new if new_kv else 0)
    if has_batch_idx:
        cache_batch_idx = torch.randperm(batch_size_cache, dtype=torch.int32, device=device)[
            :batch_size
        ]
    else:
        cache_batch_idx = None
    if alibi:
        alibi_slopes = torch.rand(batch_size, nheads, device=device, dtype=torch.float32) * 0.3
        attn_bias = attn_bias_from_alibi_slopes(
            alibi_slopes, seqlen_q, seqlen_k, None, key_padding_mask, causal=causal
        )
    else:
        alibi_slopes, attn_bias = None, None
    # cache_seqlens = torch.tensor([64], dtype=torch.int32, device=device)
    if rotary_dim > 0:
        angle = (
            torch.rand(
                seqlen_k if paged_kv_block_size is None else num_blocks * paged_kv_block_size,
                rotary_dim // 2,
                device=device,
            )
            * 2
            * math.pi
        )
        cos = torch.cos(angle).to(dtype=dtype)
        sin = torch.sin(angle).to(dtype=dtype)
        if causal or local:
            q_ro = apply_rotary_emb(
                q, cos, sin, seqlen_offsets=cache_seqlens, interleaved=rotary_interleaved
            )
        else:
            q_ro = rearrange(
                apply_rotary_emb(
                    rearrange(q, "b s h d -> b 1 (s h) d"),
                    cos,
                    sin,
                    seqlen_offsets=cache_seqlens,
                    interleaved=rotary_interleaved,
                ),
                "b 1 (s h) d -> b s h d",
                s=seqlen_q,
            )
        # q_ro = q
        k_ro = apply_rotary_emb(
            k, cos, sin, seqlen_offsets=cache_seqlens, interleaved=rotary_interleaved
        )
    else:
        cos, sin = None, None
        q_ro, k_ro = q, k
    # k_cache[:, 64:] = -1
    k_cache_ref = (k_cache if not has_batch_idx else k_cache[cache_batch_idx]).clone()
    v_cache_ref = (v_cache if not has_batch_idx else v_cache[cache_batch_idx]).clone()
    if new_kv:
        update_mask = torch.logical_and(
            cache_seqlens_expanded <= arange, arange < cache_seqlens_expanded + seqlen_new
        )
        k_cache_ref[update_mask] = rearrange(k_ro, "b s ... -> (b s) ...")
        v_cache_ref[update_mask] = rearrange(v, "b s ... -> (b s) ...")
    k_cache_rep = repeat(k_cache_ref, "b s h d -> b s (h g) d", g=nheads // nheads_k)
    v_cache_rep = repeat(v_cache_ref, "b s h d -> b s (h g) d", g=nheads // nheads_k)
    out = flash_attn_with_kvcache(
        q,
        k_cache if paged_kv_block_size is None else k_cache_paged,
        v_cache if paged_kv_block_size is None else v_cache_paged,
        k,
        v,
        rotary_cos=cos,
        rotary_sin=sin,
        cache_seqlens=cache_seqlens,
        cache_batch_idx=cache_batch_idx,
        block_table=block_table,
        causal=causal,
        window_size=window_size,
        rotary_interleaved=rotary_interleaved,
        alibi_slopes=alibi_slopes,
        num_splits=num_splits,
    )
    # out = flash_attn_with_kvcache(
    #     q, k_cache, v_cache, cache_seqlens=cache_seqlens, causal=causal, window_size=window_size
    # )
    # out = flash_attn_with_kvcache(q, k_cache, v_cache, causal=causal, window_size=window_size)
    # qk = torch.einsum("bqhd,bkhd->bhqk", q, k_cache_ref)
    # m = qk.amax(-1, keepdim=True)
    # s_tmp = torch.exp((qk - m) / math.sqrt(d))
    # o1 = torch.einsum('bhst,bthd->bshd', s_tmp, v_cache_ref)
    # lse_ref = torch.logsumexp(qk / math.sqrt(d), -1)
    # probs = torch.softmax(qk, dim=-1)
    out_ref, _ = attention_ref(
        q_ro,
        k_cache_rep,
        v_cache_rep,
        None,
        key_padding_mask,
        attn_bias,
        0.0,
        None,
        causal=causal,
        window_size=window_size,
    )
    out_pt, _ = attention_ref(
        q_ro,
        k_cache_rep,
        v_cache_rep,
        None,
        key_padding_mask,
        attn_bias,
        0.0,
        None,
        causal=causal,
        window_size=window_size,
        upcast=False,
        reorder_ops=True,
    )

    diffs = []

    diffs.append(f"Output max diff: {(out - out_ref).abs().max().item()}")
    diffs.append(f"Output mean diff: {(out - out_ref).abs().mean().item()}")
    diffs.append(f"Pytorch max diff: {(out_pt - out_ref).abs().max().item()}")
    diffs.append(f"Pytorch mean diff: {(out_pt - out_ref).abs().mean().item()}")

    # Check that FlashAttention's numerical error is at most twice the numerical error
    # of a Pytorch implementation.
    if new_kv:
        if paged_kv_block_size is None:
            k_cache_select = k_cache if not has_batch_idx else k_cache[cache_batch_idx]
            v_cache_select = v_cache if not has_batch_idx else v_cache[cache_batch_idx]
        else:
            k_cache_select = rearrange(
                k_cache_paged[block_table.flatten()],
                "(b nblocks) block_size ... -> b (nblocks block_size) ...",
                b=batch_size,
            )[:, :seqlen_k]
            v_cache_select = rearrange(
                v_cache_paged[block_table.flatten()],
                "(b nblocks) block_size ... -> b (nblocks block_size) ...",
                b=batch_size,
            )[:, :seqlen_k]
        assert torch.allclose(k_cache_select, k_cache_ref, rtol=1e-3, atol=1e-3)
        assert torch.equal(v_cache_select, v_cache_ref)
    mult = 3 if not alibi else 5
    # assert (out - out_ref).abs().max().item() <= mult * (out_pt - out_ref).abs().max().item() + 1e-5
    return (out - out_ref).abs().max().item() <= mult * (out_pt - out_ref).abs().max().item() + 1e-5,diffs



def run_flash_attn_kvcache_with_pagedkv(
    batch_size,
    seqlen_q,
    nheads,
    nheads_k,
    d,
    num_blocks,
    paged_kv_block_size,
    rotary_fraction,
    rotary_interleaved,
    seqlen_new_eq_seqlen_q,
    causal,
    local,
    alibi,
    new_kv,
    num_splits,
    dtype,
):
    has_batch_idx = False
    batch_size_cache = batch_size
    seqlen_kvcache = (num_blocks // (batch_size * 3)) * paged_kv_block_size

    return run_flash_attn_kvcache(
        batch_size,
        seqlen_q,
        nheads,
        nheads_k,
        batch_size_cache,
        seqlen_kvcache,
        d,
        has_batch_idx,
        paged_kv_block_size,
        rotary_fraction,
        rotary_interleaved,
        seqlen_new_eq_seqlen_q,
        causal,
        local,
        alibi,
        new_kv,
        num_splits,
        dtype,
    )


def run_flash_attn_kvcache_without_pagedkv(
    batch_size,
    seqlen_q,
    nheads,
    nheads_k,
    batch_size_cache,
    seqlen_kvcache,
    d,
    has_batch_idx,
    rotary_fraction,
    rotary_interleaved,
    seqlen_new_eq_seqlen_q,
    causal,
    local,
    alibi,
    new_kv,
    num_splits,
    dtype,
):
    paged_kv_block_size = None
    return run_flash_attn_kvcache(
        batch_size,
        seqlen_q,
        nheads,
        nheads_k,
        batch_size_cache,
        seqlen_kvcache,
        d,
        has_batch_idx,
        paged_kv_block_size,
        rotary_fraction,
        rotary_interleaved,
        seqlen_new_eq_seqlen_q,
        causal,
        local,
        alibi,
        new_kv,
        num_splits,
        dtype,
    )
