import torch

def generate_qkv(params, requires_grad = False):
    head_dim_qk = params["head_dim"]
    head_dim_v = params["head_dim_v"]
    num_heads_q = params["num_heads_q"]
    num_heads_kv = params["num_heads_kv"]
    batch_size = params["batch_size"]
    seqlens_q = params["seqlens_q"]
    seqlens_kv = params["seqlens_kv"]
    dtype = torch.float16 if params["is_fp16"] else torch.bfloat16

    assert len(seqlens_q) == batch_size
    assert len(seqlens_kv) == batch_size

    qkvpacked = params.get("qkvpacked", False)
    kvpacked = params.get("kvpacked", False)
    assert not (qkvpacked and kvpacked)

    if qkvpacked:
        assert seqlens_q[0] == seqlens_kv[0]
        assert num_heads_q == num_heads_kv
        assert head_dim_qk == head_dim_v
        qkv = torch.randn(
            batch_size,
            seqlens_q[0],
            3,
            num_heads_q,
            head_dim_qk,
            device="cuda",
            dtype=dtype,
            requires_grad=requires_grad
        )
        return qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

    if kvpacked:
        assert head_dim_qk == head_dim_v
        q = torch.randn(
            batch_size,
            seqlens_kv[0],
            num_heads_kv,
            head_dim_qk,
            device="cuda",
            dtype=dtype,
            requires_grad=requires_grad
        )
        kv = torch.randn(
            batch_size,
            seqlens_kv[0],
            2,
            num_heads_kv,
            head_dim_v,
            device="cuda",
            dtype=dtype,
            requires_grad=requires_grad
        )
        return q, kv[:, :, 0], kv[:, :, 1]

    q = torch.randn(
        batch_size,
        seqlens_q[0],
        num_heads_q,
        head_dim_qk,
        device="cuda",
        dtype=dtype,
        requires_grad=requires_grad
    )
    k = torch.randn(
        batch_size,
        seqlens_kv[0],
        num_heads_kv,
        head_dim_qk,
        device="cuda",
        dtype=dtype,
        requires_grad=requires_grad
    )
    v = torch.randn(
        batch_size,
        seqlens_kv[0],
        num_heads_kv,
        head_dim_v,
        device="cuda",
        dtype=dtype,
        requires_grad=requires_grad
    )
    return q,k,v

def generate_varlen_qkv(params, requires_grad=False):
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
    dtype = torch.float16 if params["is_fp16"] else torch.bfloat16

    qkvpacked = params.get("qkvpacked", False)
    kvpacked = params.get("kvpacked", False)
    assert not (qkvpacked and kvpacked)

    if qkvpacked:
        assert sum(seqlens_q) == sum(seqlens_kv)
        assert head_dim_qk == head_dim_v
        assert num_heads_q == num_heads_kv
        qkv = torch.randn(
            sum(seqlens_q),
            3,
            num_heads_q,
            head_dim_qk,
            device="cuda",
            dtype=dtype,
            requires_grad=requires_grad,
        )
        return qkv[:,0],qkv[:,1],qkv[:,2]

    if kvpacked:
        assert head_dim_qk == head_dim_v
        q = torch.randn(
            sum(seqlens_q),
            num_heads_q,
            head_dim_qk,
            device="cuda",
            dtype=dtype,
            requires_grad=requires_grad
        )
        kv = torch.randn(
            sum(seqlens_kv),
            2,
            num_heads_kv,
            head_dim_qk,
            device="cuda",
            dtype=dtype,
            requires_grad=requires_grad
        )
        return q, kv[:, 0], kv[:, 1]

    q = torch.randn(
        sum(seqlens_q),
        num_heads_q,
        head_dim_qk,
        device="cuda",
        dtype=dtype,
        requires_grad=requires_grad
    )
    k = torch.randn(
        sum(seqlens_kv),
        num_heads_kv,
        head_dim_qk,
        device="cuda",
        dtype=dtype,
        requires_grad=requires_grad
    )
    v = torch.randn(
        sum(seqlens_kv),
        num_heads_kv,
        head_dim_v,
        device="cuda",
        dtype=dtype,
        requires_grad=requires_grad
    )
    return q,k,v