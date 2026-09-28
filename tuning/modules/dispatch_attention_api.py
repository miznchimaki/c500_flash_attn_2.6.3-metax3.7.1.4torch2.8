from flash_attn.tuning.FlashTuningData.KernelType import KernelType
import torch
from flash_attn.flash_attn_interface import (
    _flash_attn_forward,
    _flash_attn_varlen_forward,
    _flash_attn_backward,
    _flash_attn_varlen_backward,
    flash_attn_with_kvcache,
    flash_attn_func,
    flash_attn_varlen_func,
)
from flash_attn.tuning.modules.profiler import pytorch_profiler as profiler
from itertools import accumulate
import flash_attn_2_cuda as flash_attn_cuda
from flash_attn.tuning.modules.flash_api_constants import (
    FLASH_FWD,
    FLASH_VARLEN_FWD,
    FLASH_BWD,
    FLASH_VARLEN_BWD,
    FLASH_KVCACHE,
)

from flash_attn.tuning.modules.tuning_verifier import (
    verify_flash_attn,
    verify_flash_attn_varlen,
    verify_flash_attn_fwd_kvcache
)
from flash_attn.tuning.modules.logger import logger
from flash_attn.tuning.modules.data_generate import generate_qkv, generate_varlen_qkv

def set_solution_params(kernel_id, kernel_type, alg, num_splits=1):
    flash_attn_cuda.ks_set_solution(
        kernel_id, kernel_type, num_splits, alg
    )

def run_forward_interface(params, config, verbose=False):
    head_dim_qk = params["head_dim"]
    head_dim_v = params["head_dim_v"]
    num_heads_q = params["num_heads_q"]
    num_heads_kv = params["num_heads_kv"]

    dropout_p = 0.17 if params["dropout"] else 0.0
    causal = params["causal"]

    softmax_scale = head_dim_qk ** (-0.5)
    window_size = (-1, -1)  # TODO: not support
    alibi_slopes = None  # TODO: not support
    attn_mask = None  # TODO: not support
    return_softmax = False

    q,k,v = generate_qkv(params)

    ret, diffs = verify_flash_attn(params,bwd_check=False, max_seqlen=1024)
    logger.debug(f"ret:{ret},diffs:{diffs}")
    if not ret: # TODO: log diffs
        return None

    softcap = 0.0
    warmups = config["warmups"]
    repeats = config["repeats"]

    return profiler(
        _flash_attn_forward,
        q,
        k,
        v,
        dropout_p = dropout_p,
        softmax_scale = softmax_scale,
        causal = causal,
        window_size = window_size,
        softcap = softcap,
        alibi_slopes = alibi_slopes,
        attn_mask = attn_mask,
        return_softmax = return_softmax,
        warmups = warmups,
        repeats = repeats,
        verbose = verbose,
    )


def run_varlen_forward_interface(params, config=None, verbose=False):
    head_dim_qk = params["head_dim"]
    head_dim_v = params["head_dim_v"]
    num_heads_q = params["num_heads_q"]
    num_heads_kv = params["num_heads_kv"]
    batch_size = params["batch_size"]
    seqlens_q = params["seqlens_q"]
    seqlens_kv = params["seqlens_kv"]
    assert len(seqlens_q) == batch_size
    assert len(seqlens_kv) == batch_size
    max_seqlen_q = max(seqlens_q)
    max_seqlen_kv = max(seqlens_kv)
    dropout_p = 0.17 if params["dropout"] else 0.0
    causal = params["causal"]
    dtype = torch.float16 if params["is_fp16"] else torch.bfloat16

    softmax_scale = head_dim_qk ** (-0.5)
    window_size = (-1, -1)  # TODO: not support
    alibi_slopes = None  # TODO: not support
    attn_mask = None  # TODO: not support
    return_softmax = False

    q,k,v = generate_varlen_qkv(params)

    cu_seqlens_q = torch.tensor(
        [0] + list(accumulate(seqlens_q)), dtype=torch.int32, device="cuda"
    )
    cu_seqlens_kv = torch.tensor(
        [0] + list(accumulate(seqlens_kv)), dtype=torch.int32, device="cuda"
    )
    max_seqlen = config.get("max_seqlen")
    if (batch_size>=128):
        max_seqlen = max(1024, max_seqlen)
    ret, diffs = verify_flash_attn(params, bwd_check=False, max_seqlen=max_seqlen)
    logger.debug(f"ret:{ret},diffs:{diffs}")
    if not ret: # TODO: log diffs
        return None

    softcap=0.0
    warmups = config["warmups"]
    repeats = config["repeats"]

    return profiler(
        _flash_attn_varlen_forward,
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_kv,
        max_seqlen_q,
        max_seqlen_kv,
        dropout_p = dropout_p,
        softmax_scale = softmax_scale,
        causal = causal,
        window_size = window_size,
        softcap = softcap,
        alibi_slopes = alibi_slopes,
        return_softmax = return_softmax,
        warmups = warmups,
        repeats = repeats,
        verbose=verbose,
    )


def run_backward_interface(params, config=None, verbose=False):
    head_dim_qk = params["head_dim"]
    head_dim_v = params["head_dim_v"]
    num_heads_q = params["num_heads_q"]
    num_heads_kv = params["num_heads_kv"]
    batch_size = params["batch_size"]
    seqlens_q = params["seqlens_q"]
    seqlens_kv = params["seqlens_kv"]
    assert len(seqlens_q) == batch_size
    assert len(seqlens_kv) == batch_size
    max_seqlen_q = max(seqlens_q)
    max_seqlen_kv = max(seqlens_kv)
    dropout_p = 0.17 if params["dropout"] else 0.0
    causal = params["causal"]
    deterministic = params["deterministic"]
    dtype = torch.float16 if params["is_fp16"] else torch.bfloat16

    q,k,v = generate_qkv(params,requires_grad = True)

    softmax_lse = torch.empty(
        (batch_size, num_heads_q, max_seqlen_q), device="cuda", dtype=dtype
    )
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    softmax_scale = head_dim_qk ** (-0.5)
    window_size = (-1, -1)  # TODO: not support
    alibi_slopes = None  # TODO: not support
    attn_mask = None  # TODO: not support
    rng_state = torch.empty((2), dtype=torch.int64)

    max_seqlen = config.get("max_seqlen")
    if (batch_size>=128):
        max_seqlen = max(1024, max_seqlen)
    ret, diffs = verify_flash_attn(params, bwd_check=True, max_seqlen=max_seqlen)
    logger.debug(f"ret:{ret},diffs:{diffs}")
    if not ret: # TODO: log diffs
        return None

    softcap = 0.0
    warmups = config["warmups"]
    repeats = config["repeats"]

    out, softmax_lse, _ = flash_attn_func(
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

    # out = torch.empty(batch_size,seqlens_q[0],num_heads_q,head_dim_v,dtype=q.dtype,device='cuda')
    dout = torch.empty_like(out)

    logger.debug(f"_flash_attn_backward,q:{q.shape},v:{v.shape},out:{out.shape},dout:{dout.shape}")

    return profiler(
        _flash_attn_backward,
        dout,
        q,
        k,
        v,
        out,
        softmax_lse,
        dq,
        dk,
        dv,
        dropout_p = dropout_p,
        softmax_scale = softmax_scale,
        causal = causal,
        window_size = window_size,
        softcap = softcap,
        alibi_slopes = alibi_slopes,
        attn_mask = attn_mask,
        deterministic = deterministic,
        rng_state = rng_state,
        warmups = warmups,
        repeats = repeats,
        verbose=verbose,
    )


def run_varlen_backward_interface(params, config=None, is_varlen=True, verbose=False):
    head_dim_qk = params["head_dim"]
    head_dim_v = params["head_dim_v"]
    num_heads_q = params["num_heads_q"]
    num_heads_kv = params["num_heads_kv"]
    batch_size = params["batch_size"]
    seqlens_q = params["seqlens_q"]
    seqlens_kv = params["seqlens_kv"]
    assert len(seqlens_q) == batch_size
    assert len(seqlens_kv) == batch_size
    max_seqlen_q = max(seqlens_q)
    max_seqlen_kv = max(seqlens_kv)
    dropout_p = 0.17 if params["dropout"] else 0.0
    causal = params["causal"]
    deterministic = params["deterministic"]
    dtype = torch.float16 if params["is_fp16"] else torch.bfloat16

    q,k,v = generate_varlen_qkv(params,requires_grad = True)
    cu_seqlens_q = torch.tensor(
        [0] + list(accumulate(seqlens_q)), dtype=torch.int32, device="cuda"
    )
    cu_seqlens_kv = torch.tensor(
        [0] + list(accumulate(seqlens_kv)), dtype=torch.int32, device="cuda"
    )

    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    softmax_scale = head_dim_qk ** (-0.5)
    window_size = (-1, -1)  # TODO: not support
    alibi_slopes = None  # TODO: not support
    attn_mask = None  # TODO: not support
    rng_state = torch.empty((2), dtype=torch.int64)

    max_seqlen = config.get("max_seqlen")
    if (batch_size>=128):
        max_seqlen = max(1024, max_seqlen)
    ret, diffs = verify_flash_attn_varlen(params, bwd_check=True, max_seqlen=max_seqlen)
    logger.debug(f"ret:{ret},diffs:{diffs}")
    if not ret: # TODO: log diffs
        return None

    softcap=0.0
    warmups = config["warmups"]
    repeats = config["repeats"]

    out, softmax_lse, _ = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_kv,
        max_seqlen_q,
        max_seqlen_kv,
        dropout_p,
        causal=causal,
        window_size=window_size,
        alibi_slopes=alibi_slopes,
        deterministic=deterministic,
        return_attn_probs=True,
    )

    dout = torch.empty_like(out)

    return profiler(
        _flash_attn_varlen_backward,
        dout,
        q,
        k,
        v,
        out,
        softmax_lse,
        dq,
        dk,
        dv,
        cu_seqlens_q,
        cu_seqlens_kv,
        max_seqlen_q,
        max_seqlen_kv,
        dropout_p = dropout_p,
        softmax_scale = softmax_scale,
        causal = causal,
        window_size = window_size,
        softcap = softcap,
        alibi_slopes = alibi_slopes,
        deterministic = deterministic,
        rng_state = rng_state,
        warmups = warmups,
        repeats = repeats,
        verbose=verbose,
    )


# TODO: support paged kv cache and rope
def run_kvcache_interface(params, config, verbose=False):
    head_dim_qk = params["head_dim"]
    num_heads_q = params["num_heads_q"]
    batch_size = params["batch_size"]
    seqlens_q = params["seqlens_q"]

    assert len(seqlens_q) == batch_size

    causal = params["causal"]
    dtype = torch.float16 if params["is_fp16"] else torch.bfloat16

    max_seqlen = config.get("max_seqlen")

    ret, diffs = verify_flash_attn_fwd_kvcache(params,max_seqlen=max_seqlen)
    if not ret: # TODO: log diffs
        return None

    warmups = config["warmups"]
    repeats = config["repeats"]

    return profiler(verify_flash_attn_fwd_kvcache,params,warmups=warmups,repeats=repeats,verbose=verbose)


def dispatch_flash_attention_api(params, config, verbose=False):
    api = params["api"]

    if api == "_flash_attn_forward" :
        return run_forward_interface(params, config, verbose=verbose)
    elif api == "_flash_attn_varlen_forward":
        return run_varlen_forward_interface(params, config, verbose=verbose)
    elif api == "_flash_attn_backward":
        return run_backward_interface(params, config, verbose=verbose)
    elif api == "_flash_attn_varlen_backward":
        return run_varlen_backward_interface(params, config, verbose=verbose)
    elif api == "flash_attn_with_kvcache":
        return run_kvcache_interface(params, config, verbose=verbose)
    else:
        raise ValueError(f"Unsupported API: {api}")
