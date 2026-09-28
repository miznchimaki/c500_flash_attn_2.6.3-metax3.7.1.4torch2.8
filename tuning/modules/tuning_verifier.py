import torch

from flash_attn.tuning.modules.flash_attn_runner import (
    run_flash_attn,
    run_flash_attn_varlen,
    run_flash_attn_kvcache,
    run_flash_attn_kvcache_with_pagedkv,
    run_flash_attn_kvcache_without_pagedkv
)

def verify_flash_attn(params,bwd_check, max_seqlen=None):
    head_dim_qk = params["head_dim"]
    head_dim_v = params["head_dim_v"]
    nheads = params["num_heads_q"]
    nheads_k = params["num_heads_kv"]
    batch_size = params["batch_size"]
    seqlens_q = params["seqlens_q"]
    seqlens_kv = params["seqlens_kv"]
    dtype = torch.float16 if params["is_fp16"] else torch.bfloat16

    if batch_size > 3:
        batch_size = 3
        seqlens_q = seqlens_q[:3]
        seqlens_kv = seqlens_kv[:3]

    assert len(seqlens_q) == batch_size
    assert len(seqlens_kv) == batch_size

    qkvpacked = params.get("qkvpacked", False)
    kvpacked = params.get("kvpacked", False)

    if head_dim_qk != head_dim_v:
        qkvpacked = False
        kvpacked = False

    assert not (qkvpacked and kvpacked)

    dropout_p = 0.3 if params["dropout"] else 0.0
    causal = params["causal"]
    deterministic = params["deterministic"]
    alibi = params["alibi"]
    has_attn_mask = params["attn_mask"] #TODO Not support now

    # window_size = (params["window_size_left"], params["window_size_right"])
    window_size = (-1,-1)

    seqlen_q = seqlens_q[0]
    seqlen_kv = seqlens_kv[0]
    # If the seqlen > max_seqlen , set to max_seqlen
    if max_seqlen is not None:
        seqlen_q = seqlen_q if seqlen_q < max_seqlen else max_seqlen
        seqlen_kv = seqlen_kv if seqlen_kv < max_seqlen else max_seqlen

    return run_flash_attn(
        batch_size,
        seqlen_q,
        seqlen_kv,
        nheads,
        nheads_k,
        head_dim_qk,
        head_dim_v,
        dropout_p,
        causal,
        window_size,
        alibi,
        deterministic,
        dtype,
        kvpacked,
        qkvpacked,
        bwd_check=bwd_check
    )


def verify_flash_attn_varlen(params, bwd_check, max_seqlen=None):
    head_dim_qk = params["head_dim"]
    head_dim_v = params["head_dim_v"]
    nheads = params["num_heads_q"]
    nheads_k = params["num_heads_kv"]
    batch_size = params["batch_size"]
    seqlens_q = params["seqlens_q"]
    seqlens_kv = params["seqlens_kv"]

    if batch_size > 3:
        batch_size = 3
        seqlens_q = seqlens_q[:3]
        seqlens_kv = seqlens_kv[:3]

    assert len(seqlens_q) == batch_size
    assert len(seqlens_kv) == batch_size

    # If the seqlen > max_seqlen , set to max_seqlen
    if max_seqlen is not None:
        seqlens_q = [max_seqlen if x > max_seqlen else x for x in seqlens_q]
        seqlens_kv = [max_seqlen if x > max_seqlen else x for x in seqlens_kv]

    max_seqlen_q = max(seqlens_q)
    max_seqlen_kv = max(seqlens_kv)
    dtype = torch.float16 if params["is_fp16"] else torch.bfloat16

    qkvpacked = params.get("qkvpacked", False)
    kvpacked = params.get("kvpacked", False)
    if head_dim_qk != head_dim_v:
        qkvpacked = False
        kvpacked = False
    assert not (qkvpacked and kvpacked)

    dropout_p = 0.3 if params["dropout"] else 0.0
    causal = params["causal"]
    deterministic = params["deterministic"]
    alibi = params["alibi"]

    # window_size = (params["window_size_left"], params["window_size_right"])
    window_size = (-1, -1)

    return run_flash_attn_varlen(
        seqlens_q,
        seqlens_kv,
        nheads,
        nheads_k,
        head_dim_qk,
        head_dim_v,
        dropout_p,
        causal,
        window_size,
        alibi,
        deterministic,
        dtype,
        kvpacked,
        qkvpacked,
        bwd_check=bwd_check
    )

def verify_flash_attn_fwd_kvcache(params, max_seqlen=None):
    head_dim_qk = params["head_dim"]
    num_heads_q = params["num_heads_q"]
    batch_size = params["batch_size"]
    seqlens_q = params["seqlens_q"]

    num_heads_kv = params["num_heads_kv"]
    head_dim_v = params["head_dim_v"]

    casual = params["causal"]
    alibi = params["alibi"]

    has_batch_idx = False
    rotary_fraction = 0.0
    rotary_interleaved = False
    seqlen_new_eq_seqlen_q = True
    new_kv = False
    local = True
    num_splits = 1

    dtype = torch.float16 if params["is_fp16"] else torch.bfloat16

    paged_block_size = None
    paged_kv = params["paged_kv"]
    if paged_kv:
        num_blocks = params["paged_num_blocks"]
        paged_block_size = params["paged_block_size"]
        seqlens_kv = [num_blocks * paged_block_size]

        seqlen_q = seqlens_q[0]
        if max_seqlen is not None:
            seqlen_q = seqlen_q if seqlen_q < max_seqlen else max_seqlen

        return run_flash_attn_kvcache_with_pagedkv(
            batch_size,
            seqlen_q,
            num_heads_q,
            num_heads_kv,
            head_dim_qk,
            num_blocks,
            paged_block_size,
            rotary_fraction,
            rotary_interleaved,
            seqlen_new_eq_seqlen_q,
            casual,
            local,
            alibi,
            new_kv,
            num_splits,
            dtype
        )

    else:
        batch_size_cache = params["batch_size_c"]
        seqlens_kv = params["seqlens_kv"]

        seqlen_q = seqlens_q[0]
        seqlen_kv = seqlens_kv[0]
        if max_seqlen is not None:
            seqlen_q = seqlen_q if seqlen_q < max_seqlen else max_seqlen
            seqlen_kv = seqlen_kv if seqlen_kv < max_seqlen else max_seqlen

        return run_flash_attn_kvcache_without_pagedkv(
            batch_size,
            seqlen_q,
            num_heads_q,
            num_heads_kv,
            batch_size_cache,
            seqlen_kv,
            head_dim_qk,
            has_batch_idx,
            rotary_fraction,
            rotary_interleaved,
            seqlen_new_eq_seqlen_q,
            casual,
            local,
            alibi,
            new_kv,
            num_splits,
            dtype
        )

def dispatch_flash_attn_verifier(problem, max_seqlen=None):
    func_name = problem["api"]

    flash_attn = [
        "_flash_attn_forward",
        "_flash_attn_backward",
        "_flash_attn_forward_inference"
    ]
    flash_attn_varlen = [
        "_flash_attn_varlen_forward",
        "_flash_attn_varlen_backward",
        "_flash_attn_varlen_forward_inference"
    ]
    flash_attn_fwd_kvcache = ["flash_attn_with_kvcache"]

    bwd_check = True if "backward" in func_name else False

    if func_name in flash_attn:
        return verify_flash_attn(problem,bwd_check, max_seqlen=max_seqlen)
    elif func_name in flash_attn_varlen:
        return verify_flash_attn_varlen(problem,bwd_check, max_seqlen=max_seqlen)
    elif func_name in flash_attn_fwd_kvcache:
        return verify_flash_attn_fwd_kvcache(problem, max_seqlen=max_seqlen)
    else:
        logger.error(f"No function name:{func_name}")
        return False