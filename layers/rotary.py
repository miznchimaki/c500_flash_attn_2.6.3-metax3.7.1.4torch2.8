# Copyright (c) 2023, Tri Dao.

import math
from typing import Optional, Tuple, Union

import torch
from einops import rearrange, repeat

torch_version=torch.__version__
torch_version=torch_version.split("+",1)[0]

if torch_version < "2.1.0":
    import rotary_emb
    def rotate_half(x, interleaved=False):
        if not interleaved:
            x1, x2 = x.chunk(2, dim=-1)
            return torch.cat((-x2, x1), dim=-1)
        else:
            x1, x2 = x[..., ::2], x[..., 1::2]
            return rearrange(torch.stack((-x2, x1), dim=-1), '... d two -> ... (d two)', two=2)

    def apply_rotary_emb_torch(x, cos, sin, interleaved=False):
        """
        x: (batch_size, seqlen, nheads, headdim)
        cos, sin: (seqlen, rotary_dim / 2)
        """
        ro_dim = cos.shape[-1] * 2
        assert ro_dim <= x.shape[-1]
        cos = repeat(cos, 's d -> s 1 (2 d)')
        sin = repeat(sin, 's d -> s 1 (2 d)')
        return torch.cat([x[..., :ro_dim] * cos + rotate_half(x[..., :ro_dim], interleaved) * sin,
                        x[..., ro_dim:]], dim=-1)


    class ApplyRotaryEmb(torch.autograd.Function):

        @staticmethod
        def forward(ctx, x, cos, sin, interleaved=False, inplace=False):
            """
                x: (batch_size, seqlen, nheads, headdim)
                cos, sin: (seqlen, rotary_dim / 2)
                interleaved: if True, rotate pairs of even and odd dimensions (GPT-J style) instead
                    of 1st half and 2nd half (GPT-NeoX style).
            rotary_dim must be <= headdim
            Apply rotary embedding to the first rotary_dim of x.
            """
            batch, seqlen, nheads, headdim = x.shape
            rotary_seqlen, rotary_dim = cos.shape
            rotary_dim *= 2
            assert rotary_dim <= headdim
            assert seqlen <= rotary_seqlen
            assert sin.shape == (rotary_seqlen, rotary_dim // 2)
            x_ro = x[..., :rotary_dim]
            x1, x2 = x_ro.chunk(2, dim=-1) if not interleaved else (x_ro[..., ::2], x_ro[..., 1::2])
            out = torch.empty_like(x) if not inplace else x
            out_ro = out[..., :rotary_dim]
            if inplace:
                o1, o2 = x1, x2
            else:
                o1, o2 = (out_ro.chunk(2, dim=-1) if not interleaved
                        else (out_ro[..., ::2], out_ro[..., 1::2]))
            rotary_emb.apply_rotary(x1, x2, rearrange(cos[:seqlen], 's d -> s 1 d'),
                                    rearrange(sin[:seqlen], 's d -> s 1 d'), o1, o2, False)
            if not inplace and rotary_dim < headdim:
                out[..., rotary_dim:].copy_(x[..., rotary_dim:])
            ctx.save_for_backward(cos, sin)
            ctx.interleaved = interleaved
            ctx.inplace = inplace
            return out if not inplace else x

        @staticmethod
        def backward(ctx, do):
            cos, sin = ctx.saved_tensors
            _, seqlen, _, headdim = do.shape
            rotary_dim = cos.shape[-1]
            rotary_dim *= 2
            inplace = ctx.inplace
            do_ro = do[..., :rotary_dim]
            do1, do2 = (do_ro.chunk(2, dim=-1) if not ctx.interleaved
                        else (do_ro[..., ::2], do_ro[..., 1::2]))
            dx = torch.empty_like(do) if not inplace else do
            if inplace:
                dx1, dx2 = do1, do2
            else:
                dx_ro = dx[..., :rotary_dim]
                dx1, dx2 = (dx_ro.chunk(2, dim=-1) if not ctx.interleaved
                            else (dx_ro[..., ::2], dx_ro[..., 1::2]))
            rotary_emb.apply_rotary(do1, do2, rearrange(cos[:seqlen], 's d -> s 1 d'),
                                    rearrange(sin[:seqlen], 's d -> s 1 d'), dx1, dx2, True)
            if not inplace and rotary_dim < headdim:
                dx[..., rotary_dim:].copy_(do[..., rotary_dim:])
            return dx, None, None, None, None


    def apply_rotary_emb(
        x,
        cos,
        sin,
        interleaved=False,
        inplace=False,
        seqlen_offsets: Union[int, torch.Tensor] = 0,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
    ):
        """
        Arguments:
            x: (batch_size, seqlen, nheads, headdim) if cu_seqlens is None
                else (total_seqlen, nheads, headdim)
            cos, sin: (seqlen_rotary, rotary_dim / 2)
            interleaved: if True, rotate pairs of even and odd dimensions (GPT-J style) instead
                of 1st half and 2nd half (GPT-NeoX style).
            inplace: if True, apply rotary embedding in-place.
            seqlen_offsets: (batch_size,) or int. Each sequence in x is shifted by this amount.
                Most commonly used in inference when we have KV cache.
            cu_seqlens: (batch + 1,) or None
            max_seqlen: int
        Return:
            out: (batch_size, seqlen, nheads, headdim) if cu_seqlens is None
                else (total_seqlen, nheads, headdim)
        rotary_dim must be <= headdim
        Apply rotary embedding to the first rotary_dim of x.
        """
        '''
        return ApplyRotaryEmb.apply(
            x, cos, sin, interleaved, inplace, seqlen_offsets, cu_seqlens, max_seqlen
        )
        '''
        return ApplyRotaryEmb.apply(
            x, cos, sin, interleaved, inplace
        )

    apply_rotary_emb_func = ApplyRotaryEmb.apply


    class ApplyRotaryEmbQKV_(torch.autograd.Function):

        @staticmethod
        def forward(ctx, qkv, cos, sin, cos_k=None, sin_k=None, interleaved=False):
            """
                qkv: (batch_size, seqlen, 3, nheads, headdim)
                cos, sin: (seqlen, rotary_dim / 2)
                cos_k, sin_k: (seqlen, rotary_dim / 2), optional
                interleaved: if True, rotate pairs of even and odd dimensions (GPT-J style) instead of
                    1st half and 2nd half (GPT-NeoX style).
            rotary_dim must be <= headdim
            Apply rotary embedding *inplace* to the first rotary_dim of q and k.
            """
            batch, seqlen, three, nheads, headdim = qkv.shape
            assert three == 3
            rotary_seqlen, rotary_dim = cos.shape
            rotary_dim *= 2
            assert rotary_dim <= headdim
            assert seqlen <= rotary_seqlen
            cos_k = cos if cos_k is None else cos_k
            sin_k = sin if sin_k is None else sin_k
            assert sin.shape == cos_k.shape == sin_k.shape == (rotary_seqlen, rotary_dim // 2)
            q_ro = qkv[:, :, 0, :, :rotary_dim]
            q1, q2 = q_ro.chunk(2, dim=-1) if not interleaved else (q_ro[..., ::2], q_ro[..., 1::2])
            rotary_emb.apply_rotary(q1, q2, rearrange(cos[:seqlen], 's d -> s 1 d'),
                                    rearrange(sin[:seqlen], 's d -> s 1 d'), q1, q2, False)
            k_ro = qkv[:, :, 1, :, :rotary_dim]
            k1, k2 = k_ro.chunk(2, dim=-1) if not interleaved else (k_ro[..., ::2], k_ro[..., 1::2])
            rotary_emb.apply_rotary(k1, k2, rearrange(cos_k[:seqlen], 's d -> s 1 d'),
                                    rearrange(sin_k[:seqlen], 's d -> s 1 d'), k1, k2, False)
            ctx.save_for_backward(cos, sin, cos_k, sin_k)
            ctx.interleaved = interleaved
            return qkv

        @staticmethod
        def backward(ctx, dqkv):
            cos, sin, cos_k, sin_k = ctx.saved_tensors
            _, seqlen, _, _, headdim = dqkv.shape
            rotary_dim = cos.shape[-1]
            rotary_dim *= 2
            dq_ro = dqkv[:, :, 0, :, :rotary_dim]
            dq1, dq2 = (dq_ro.chunk(2, dim=-1) if not ctx.interleaved
                        else (dq_ro[..., ::2], dq_ro[..., 1::2]))
            rotary_emb.apply_rotary(dq1, dq2, rearrange(cos[:seqlen], 's d -> s 1 d'),
                                    rearrange(sin[:seqlen], 's d -> s 1 d'), dq1, dq2, True)
            dk_ro = dqkv[:, :, 1, :, :rotary_dim]
            dk1, dk2 = (dk_ro.chunk(2, dim=-1) if not ctx.interleaved
                        else (dk_ro[..., ::2], dk_ro[..., 1::2]))
            rotary_emb.apply_rotary(dk1, dk2, rearrange(cos_k[:seqlen], 's d -> s 1 d'),
                                    rearrange(sin_k[:seqlen], 's d -> s 1 d'), dk1, dk2, True)
            return dqkv, None, None, None, None, None


    apply_rotary_emb_qkv_ = ApplyRotaryEmbQKV_.apply


    class ApplyRotaryEmbKV_(torch.autograd.Function):

        @staticmethod
        def forward(ctx, kv, cos, sin, interleaved=False):
            """
                kv: (batch_size, seqlen, 2, nheads, headdim)
                cos, sin: (seqlen, rotary_dim / 2)
                interleaved: if True, rotate pairs of even and odd dimensions (GPT-J style) instead of
                    1st half and 2nd half (GPT-NeoX style).
            rotary_dim must be <= headdim
            Apply rotary embedding *inplace* to the first rotary_dim of k.
            """
            batch, seqlen, two, nheads, headdim = kv.shape
            assert two == 2
            rotary_seqlen, rotary_dim = cos.shape
            rotary_dim *= 2
            assert rotary_dim <= headdim
            assert seqlen <= rotary_seqlen
            k_ro = kv[:, :, 0, :, :rotary_dim]
            k1, k2 = k_ro.chunk(2, dim=-1) if not interleaved else (k_ro[..., ::2], k_ro[..., 1::2])
            rotary_emb.apply_rotary(k1, k2, rearrange(cos[:seqlen], 's d -> s 1 d'),
                                    rearrange(sin[:seqlen], 's d -> s 1 d'), k1, k2,
                                    False)  # conj=False since this is the forward pass
            ctx.save_for_backward(cos, sin)
            ctx.interleaved = interleaved
            return kv

        @staticmethod
        def backward(ctx, dkv):
            cos, sin = ctx.saved_tensors
            _, seqlen, _, _, headdim = dkv.shape
            rotary_dim = cos.shape[-1]
            rotary_dim *= 2
            dk_ro = dkv[:, :, 0, :, :rotary_dim]
            dk1, dk2 = (dk_ro.chunk(2, dim=-1) if not ctx.interleaved
                        else (dk_ro[..., ::2], dk_ro[..., 1::2]))
            rotary_emb.apply_rotary(dk1, dk2, rearrange(cos[:seqlen], 's d -> s 1 d'),
                                    rearrange(sin[:seqlen], 's d -> s 1 d'), dk1, dk2,
                                    True)  # conj=True since this is the backward pass
            return dkv, None, None, None


    apply_rotary_emb_kv_ = ApplyRotaryEmbKV_.apply


    class RotaryEmbedding(torch.nn.Module):
        """
        The rotary position embeddings from RoFormer_ (Su et. al).
        A crucial insight from the method is that the query and keys are
        transformed by rotation matrices which depend on the relative positions.

        Other implementations are available in the Rotary Transformer repo_ and in
        GPT-NeoX_, GPT-NeoX was an inspiration

        .. _RoFormer: https://arxiv.org/abs/2104.09864
        .. _repo: https://github.com/ZhuiyiTechnology/roformer
        .. _GPT-NeoX: https://github.com/EleutherAI/gpt-neox

        If scale_base is not None, this implements XPos (Sun et al., https://arxiv.org/abs/2212.10554).
        A recommended value for scale_base is 512: https://github.com/HazyResearch/flash-attention/issues/96
        Reference: https://github.com/sunyt32/torchscale/blob/main/torchscale/component/xpos_relative_position.py
        """

        def __init__(self, dim: int, base=10000.0, interleaved=False, scale_base=None,
                    pos_idx_in_fp32=True, device=None):
            """
                interleaved: if True, rotate pairs of even and odd dimensions (GPT-J style) instead
                    of 1st half and 2nd half (GPT-NeoX style).
                pos_idx_in_fp32: if True, the position indices [0.0, ..., seqlen - 1] are in fp32,
                    otherwise they might be in lower precision.
                    This option was added because previously (before 2023-07-02), when we construct
                    the position indices, we use the dtype of self.inv_freq. In most cases this would
                    be fp32, but if the model is trained in pure bf16 (not mixed precision), then
                    self.inv_freq would be bf16, and the position indices are also in bf16.
                    Because of the limited precision of bf16 (e.g. 1995.0 is rounded to 2000.0), the
                    embeddings for some positions will coincide.
                    To maintain compatibility with models previously trained in pure bf16,
                    we add this option.
            """
            super().__init__()
            self.dim = dim
            self.base = float(base)
            self.pos_idx_in_fp32 = pos_idx_in_fp32
            # Generate and save the inverse frequency buffer (non trainable)
            inv_freq = self._compute_inv_freq(device)
            self.register_buffer("inv_freq", inv_freq, persistent=False)
            self.interleaved = interleaved
            self.scale_base = scale_base
            scale = ((torch.arange(0, dim, 2, device=device, dtype=torch.float32) + 0.4 * dim)
                    / (1.4 * dim) if scale_base is not None else None)
            self.register_buffer("scale", scale, persistent=False)

            self._seq_len_cached = 0
            self._cos_cached = None
            self._sin_cached = None
            self._cos_k_cached = None
            self._sin_k_cached = None

        def _compute_inv_freq(self, device=None):
            return 1.0 / (self.base ** (torch.arange(0, self.dim, 2, device=device,
                                                    dtype=torch.float32) / self.dim))


        def _update_cos_sin_cache(self, seqlen, device=None, dtype=None):
            # Reset the tables if the sequence length has changed,
            # if we're on a new device (possibly due to tracing for instance),
            # or if we're switching from inference mode to training
            if (seqlen > self._seq_len_cached or self._cos_cached.device != device
                or self._cos_cached.dtype != dtype
                or (self.training and self._cos_cached.is_inference())):
                self._seq_len_cached = seqlen
                # We want fp32 here, not self.inv_freq.dtype, since the model could be loaded in bf16
                # And the output of arange can be quite large, so bf16 would lose a lot of precision.
                # However, for compatibility reason, we add an option to use the dtype of self.inv_freq.
                if self.pos_idx_in_fp32:
                    t = torch.arange(seqlen, device=device, dtype=torch.float32)
                    # We want fp32 here as well since inv_freq will be multiplied with t, and the output
                    # will be large. Having it in bf16 will lose a lot of precision and cause the
                    # cos & sin output to change significantly.
                    # We want to recompute self.inv_freq if it was not loaded in fp32
                    if self.inv_freq.dtype != torch.float32:
                        inv_freq = self._compute_inv_freq(device=device)
                    else:
                        inv_freq = self.inv_freq
                else:
                    t = torch.arange(seqlen, device=device, dtype=self.inv_freq.dtype)
                    inv_freq = self.inv_freq
                # Don't do einsum, it converts fp32 to fp16 under AMP
                # freqs = torch.einsum("i,j->ij", t, self.inv_freq)
                freqs = torch.outer(t, inv_freq)
                if self.scale is None:
                    self._cos_cached = torch.cos(freqs).to(dtype)
                    self._sin_cached = torch.sin(freqs).to(dtype)
                else:
                    power = ((torch.arange(seqlen, dtype=self.scale.dtype, device=self.scale.device)
                            - seqlen // 2) / self.scale_base)
                    scale = self.scale.to(device=power.device) ** rearrange(power, 's -> s 1')
                    # We want the multiplication by scale to happen in fp32
                    self._cos_cached = (torch.cos(freqs) * scale).to(dtype)
                    self._sin_cached = (torch.sin(freqs) * scale).to(dtype)
                    self._cos_k_cached = (torch.cos(freqs) / scale).to(dtype)
                    self._sin_k_cached = (torch.sin(freqs) / scale).to(dtype)

        def forward(self, qkv: torch.Tensor, kv: Optional[torch.Tensor] = None,
                    seqlen_offset: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            qkv: (batch, seqlen, 3, nheads, headdim) if kv is none,
                else it's just q of shape (batch, seqlen, nheads, headdim)
            kv: (batch, seqlen, 2, nheads, headdim)
            seqlen_offset: can be used in generation where the qkv being passed in is only the last
            token in the batch.
            """
            seqlen = qkv.shape[1]
            self._update_cos_sin_cache(seqlen + seqlen_offset, device=qkv.device, dtype=qkv.dtype)
            if kv is None:
                if self.scale is None:
                    return apply_rotary_emb_qkv_(
                        qkv, self._cos_cached[seqlen_offset:], self._sin_cached[seqlen_offset:],
                        None, None, self.interleaved
                    )
                else:
                    return apply_rotary_emb_qkv_(
                        qkv, self._cos_cached[seqlen_offset:], self._sin_cached[seqlen_offset:],
                        self._cos_k_cached[seqlen_offset:], self._sin_k_cached[seqlen_offset:],
                        self.interleaved
                    )
            else:
                q = qkv
                q = apply_rotary_emb_func(
                    q, self._cos_cached[seqlen_offset:], self._sin_cached[seqlen_offset:],
                    self.interleaved, True
                )
                if self.scale is None:
                    kv = apply_rotary_emb_kv_(
                        kv, self._cos_cached[seqlen_offset:], self._sin_cached[seqlen_offset:],
                        self.interleaved
                    )
                else:
                    kv = apply_rotary_emb_kv_(
                        kv, self._cos_k_cached[seqlen_offset:], self._sin_k_cached[seqlen_offset:],
                        self.interleaved
                    )
                return q, kv

else:
    from flash_attn.ops.triton.rotary import apply_rotary


    def rotate_half(x, interleaved=False):
        if not interleaved:
            x1, x2 = x.chunk(2, dim=-1)
            return torch.cat((-x2, x1), dim=-1)
        else:
            x1, x2 = x[..., ::2], x[..., 1::2]
            return rearrange(torch.stack((-x2, x1), dim=-1), "... d two -> ... (d two)", two=2)


    def apply_rotary_emb_torch(x, cos, sin, interleaved=False):
        """
        x: (batch_size, seqlen, nheads, headdim)
        cos, sin: (seqlen, rotary_dim / 2) or (batch_size, seqlen, rotary_dim / 2)
        """
        ro_dim = cos.shape[-1] * 2
        assert ro_dim <= x.shape[-1]
        cos = repeat(cos, "... d -> ... 1 (2 d)" if not interleaved else "... d -> ... 1 (d 2)")
        sin = repeat(sin, "... d -> ... 1 (2 d)" if not interleaved else "... d -> ... 1 (d 2)")
        return torch.cat(
            [x[..., :ro_dim] * cos + rotate_half(x[..., :ro_dim], interleaved) * sin, x[..., ro_dim:]],
            dim=-1,
        )


    class ApplyRotaryEmb(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx,
            x,
            cos,
            sin,
            interleaved=False,
            inplace=False,
            seqlen_offsets: Union[int, torch.Tensor] = 0,
            cu_seqlens: Optional[torch.Tensor] = None,
            max_seqlen: Optional[int] = None,
        ):
            out = apply_rotary(
                x,
                cos,
                sin,
                seqlen_offsets=seqlen_offsets,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                interleaved=interleaved,
                inplace=inplace,
            )
            if isinstance(seqlen_offsets, int):
                ctx.save_for_backward(cos, sin, cu_seqlens)  # Can't save int with save_for_backward
                ctx.seqlen_offsets = seqlen_offsets
            else:
                ctx.save_for_backward(cos, sin, cu_seqlens, seqlen_offsets)
                ctx.seqlen_offsets = None
            ctx.interleaved = interleaved
            ctx.inplace = inplace
            ctx.max_seqlen = max_seqlen
            return out if not inplace else x

        @staticmethod
        def backward(ctx, do):
            seqlen_offsets = ctx.seqlen_offsets
            if seqlen_offsets is None:
                cos, sin, cu_seqlens, seqlen_offsets = ctx.saved_tensors
            else:
                cos, sin, cu_seqlens = ctx.saved_tensors
            # TD [2023-09-02]: For some reason Triton (2.0.0.post1) errors with
            # "[CUDA]: invalid device context", and cloning makes it work. Idk why. Triton 2.1.0 works.
            if not ctx.interleaved and not ctx.inplace:
                do = do.clone()
            dx = apply_rotary(
                do,
                cos,
                sin,
                seqlen_offsets=seqlen_offsets,
                cu_seqlens=cu_seqlens,
                max_seqlen=ctx.max_seqlen,
                interleaved=ctx.interleaved,
                inplace=ctx.inplace,
                conjugate=True,
            )
            return dx, None, None, None, None, None, None, None


    def apply_rotary_emb(
        x,
        cos,
        sin,
        interleaved=False,
        inplace=False,
        seqlen_offsets: Union[int, torch.Tensor] = 0,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
    ):
        """
        Arguments:
            x: (batch_size, seqlen, nheads, headdim) if cu_seqlens is None
                else (total_seqlen, nheads, headdim)
            cos, sin: (seqlen_rotary, rotary_dim / 2)
            interleaved: if True, rotate pairs of even and odd dimensions (GPT-J style) instead
                of 1st half and 2nd half (GPT-NeoX style).
            inplace: if True, apply rotary embedding in-place.
            seqlen_offsets: (batch_size,) or int. Each sequence in x is shifted by this amount.
                Most commonly used in inference when we have KV cache.
            cu_seqlens: (batch + 1,) or None
            max_seqlen: int
        Return:
            out: (batch_size, seqlen, nheads, headdim) if cu_seqlens is None
                else (total_seqlen, nheads, headdim)
        rotary_dim must be <= headdim
        Apply rotary embedding to the first rotary_dim of x.
        """
        return ApplyRotaryEmb.apply(
            x, cos, sin, interleaved, inplace, seqlen_offsets, cu_seqlens, max_seqlen
        )


    # For backward compatibility
    apply_rotary_emb_func = apply_rotary_emb


    class ApplyRotaryEmbQKV_(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx,
            qkv,
            cos,
            sin,
            cos_k=None,
            sin_k=None,
            interleaved=False,
            seqlen_offsets: Union[int, torch.Tensor] = 0,
        ):
            batch, seqlen, three, nheads, headdim = qkv.shape
            assert three == 3
            if cos_k is None and sin_k is None and qkv.is_contiguous():
                # Call 1 kernel instead of 2 kernels
                # We need qkv to be contiguous so that when we reshape to combine (3, nheads)
                # dimensions, we get the same tensor
                # qk = rearrange(qkv[:, :, :2], "b s t h d -> b s (t h) d")
                qk = qkv[:, :, :2].reshape(batch, seqlen, -1, headdim)
                apply_rotary(
                    qk, cos, sin, seqlen_offsets=seqlen_offsets, interleaved=interleaved, inplace=True
                )
            else:
                cos_k = cos if cos_k is None else cos_k
                sin_k = sin if sin_k is None else sin_k
                q, k = qkv[:, :, 0], qkv[:, :, 1]
                apply_rotary(q, cos, sin, seqlen_offsets, interleaved=interleaved, inplace=True)
                apply_rotary(k, cos_k, sin_k, seqlen_offsets, interleaved=interleaved, inplace=True)
                ctx.save_for_backward(cos, sin, cos_k, sin_k)
            if isinstance(seqlen_offsets, int):
                ctx.save_for_backward(cos, sin, cos_k, sin_k)
                ctx.seqlen_offsets = seqlen_offsets
            else:
                ctx.save_for_backward(cos, sin, cos_k, sin_k, seqlen_offsets)
                ctx.seqlen_offsets = None
            ctx.interleaved = interleaved
            return qkv

        @staticmethod
        def backward(ctx, dqkv):
            seqlen_offsets = ctx.seqlen_offsets
            if seqlen_offsets is None:
                cos, sin, cos_k, sin_k, seqlen_offsets = ctx.saved_tensors
            else:
                cos, sin, cos_k, sin_k = ctx.saved_tensors
            if cos_k is None and sin_k is None and dqkv.is_contiguous():
                # Call 1 kernel instead of 2 kernels
                # We need dqkv to be contiguous so that when we reshape to combine (3, nheads)
                # dimensions, we get the same tensor
                dqk = rearrange(dqkv[:, :, :2], "b s t h d -> b s (t h) d")
                apply_rotary(
                    dqk,
                    cos,
                    sin,
                    seqlen_offsets=seqlen_offsets,
                    interleaved=ctx.interleaved,
                    inplace=True,
                    conjugate=True,
                )
            else:
                cos_k = cos if cos_k is None else cos_k
                sin_k = sin if sin_k is None else sin_k
                dq, dk = dqkv[:, :, 0], dqkv[:, :, 1]
                apply_rotary(
                    dq, cos, sin, seqlen_offsets, interleaved=ctx.interleaved, inplace=True, conjugate=True
                )
                apply_rotary(
                    dk,
                    cos_k,
                    sin_k,
                    seqlen_offsets,
                    interleaved=ctx.interleaved,
                    inplace=True,
                    conjugate=True,
                )
            return dqkv, None, None, None, None, None, None


    def apply_rotary_emb_qkv_(
        qkv,
        cos,
        sin,
        cos_k=None,
        sin_k=None,
        interleaved=False,
        seqlen_offsets: Union[int, torch.Tensor] = 0,
    ):
        """
        Arguments:
            qkv: (batch_size, seqlen, 3, nheads, headdim)
            cos, sin: (seqlen, rotary_dim / 2)
            cos_k, sin_k: (seqlen, rotary_dim / 2), optional
            interleaved: if True, rotate pairs of even and odd dimensions (GPT-J style) instead of
                1st half and 2nd half (GPT-NeoX style).
            seqlen_offsets: (batch_size,) or int. Each sequence in Q and K is shifted by this amount.
                Most commonly used in inference when we have KV cache.
        Return:
            qkv: (batch_size, seqlen, 3, nheads, headdim)
        rotary_dim must be <= headdim
        Apply rotary embedding *inplace* to the first rotary_dim of Q and K.
        """
        return ApplyRotaryEmbQKV_.apply(qkv, cos, sin, cos_k, sin_k, interleaved, seqlen_offsets)


    class ApplyRotaryEmbKV_(torch.autograd.Function):
        @staticmethod
        def forward(ctx, kv, cos, sin, interleaved=False, seqlen_offsets: Union[int, torch.Tensor] = 0):
            batch, seqlen, two, nheads, headdim = kv.shape
            assert two == 2
            k = kv[:, :, 0]
            apply_rotary(
                k, cos, sin, seqlen_offsets=seqlen_offsets, interleaved=interleaved, inplace=True
            )
            if isinstance(seqlen_offsets, int):
                ctx.save_for_backward(cos, sin)  # Can't save int with save_for_backward
                ctx.seqlen_offsets = seqlen_offsets
            else:
                ctx.save_for_backward(cos, sin, seqlen_offsets)
                ctx.seqlen_offsets = None
            ctx.interleaved = interleaved
            return kv

        @staticmethod
        def backward(ctx, dkv):
            seqlen_offsets = ctx.seqlen_offsets
            if seqlen_offsets is None:
                cos, sin, seqlen_offsets = ctx.saved_tensors
            else:
                cos, sin = ctx.saved_tensors
            apply_rotary(
                dkv[:, :, 0],
                cos,
                sin,
                seqlen_offsets=seqlen_offsets,
                interleaved=ctx.interleaved,
                inplace=True,
                conjugate=True,
            )
            return dkv, None, None, None, None


    apply_rotary_emb_kv_ = ApplyRotaryEmbKV_.apply


    def apply_rotary_emb_kv_(
        kv,
        cos,
        sin,
        interleaved=False,
        seqlen_offsets: Union[int, torch.Tensor] = 0,
    ):
        """
        Arguments:
            kv: (batch_size, seqlen, 2, nheads, headdim)
            cos, sin: (seqlen, rotary_dim / 2)
            interleaved: if True, rotate pairs of even and odd dimensions (GPT-J style) instead of
                1st half and 2nd half (GPT-NeoX style).
            seqlen_offsets: (batch_size,) or int. Each sequence in Q and K is shifted by this amount.
                Most commonly used in inference when we have KV cache.
        Return:
            kv: (batch_size, seqlen, 2, nheads, headdim)
        rotary_dim must be <= headdim
        Apply rotary embedding *inplace* to the first rotary_dim of K.
        """
        return ApplyRotaryEmbKV_.apply(kv, cos, sin, interleaved, seqlen_offsets)


    class RotaryEmbedding(torch.nn.Module):
        """
        The rotary position embeddings from RoFormer_ (Su et. al).
        A crucial insight from the method is that the query and keys are
        transformed by rotation matrices which depend on the relative positions.

        Other implementations are available in the Rotary Transformer repo_ and in
        GPT-NeoX_, GPT-NeoX was an inspiration

        .. _RoFormer: https://arxiv.org/abs/2104.09864
        .. _repo: https://github.com/ZhuiyiTechnology/roformer
        .. _GPT-NeoX: https://github.com/EleutherAI/gpt-neox

        If scale_base is not None, this implements XPos (Sun et al., https://arxiv.org/abs/2212.10554).
        A recommended value for scale_base is 512: https://github.com/HazyResearch/flash-attention/issues/96
        Reference: https://github.com/sunyt32/torchscale/blob/main/torchscale/component/xpos_relative_position.py
        """

        def __init__(
            self,
            dim: int,
            base=10000.0,
            interleaved=False,
            scale_base=None,
            pos_idx_in_fp32=True,
            device=None,
        ):
            """
            interleaved: if True, rotate pairs of even and odd dimensions (GPT-J style) instead
                of 1st half and 2nd half (GPT-NeoX style).
            pos_idx_in_fp32: if True, the position indices [0.0, ..., seqlen - 1] are in fp32,
                otherwise they might be in lower precision.
                This option was added because previously (before 2023-07-02), when we construct
                the position indices, we use the dtype of self.inv_freq. In most cases this would
                be fp32, but if the model is trained in pure bf16 (not mixed precision), then
                self.inv_freq would be bf16, and the position indices are also in bf16.
                Because of the limited precision of bf16 (e.g. 1995.0 is rounded to 2000.0), the
                embeddings for some positions will coincide.
                To maintain compatibility with models previously trained in pure bf16,
                we add this option.
            """
            super().__init__()
            self.dim = dim
            self.base = float(base)
            self.pos_idx_in_fp32 = pos_idx_in_fp32
            # Generate and save the inverse frequency buffer (non trainable)
            inv_freq = self._compute_inv_freq(device)
            self.register_buffer("inv_freq", inv_freq, persistent=False)
            self.interleaved = interleaved
            self.scale_base = scale_base
            scale = (
                (torch.arange(0, dim, 2, device=device, dtype=torch.float32) + 0.4 * dim) / (1.4 * dim)
                if scale_base is not None
                else None
            )
            self.register_buffer("scale", scale, persistent=False)

            self._seq_len_cached = 0
            self._cos_cached = None
            self._sin_cached = None
            self._cos_k_cached = None
            self._sin_k_cached = None

        def _compute_inv_freq(self, device=None):
            return 1.0 / (
                self.base
                ** (torch.arange(0, self.dim, 2, device=device, dtype=torch.float32) / self.dim)
            )

        def _update_cos_sin_cache(self, seqlen, device=None, dtype=None):
            # Reset the tables if the sequence length has changed,
            # if we're on a new device (possibly due to tracing for instance),
            # or if we're switching from inference mode to training
            if (
                seqlen > self._seq_len_cached
                or self._cos_cached is None
                or self._cos_cached.device != device
                or self._cos_cached.dtype != dtype
                or (self.training and self._cos_cached.is_inference())
            ):
                self._seq_len_cached = seqlen
                # We want fp32 here, not self.inv_freq.dtype, since the model could be loaded in bf16
                # And the output of arange can be quite large, so bf16 would lose a lot of precision.
                # However, for compatibility reason, we add an option to use the dtype of self.inv_freq.
                if self.pos_idx_in_fp32:
                    t = torch.arange(seqlen, device=device, dtype=torch.float32)
                    # We want fp32 here as well since inv_freq will be multiplied with t, and the output
                    # will be large. Having it in bf16 will lose a lot of precision and cause the
                    # cos & sin output to change significantly.
                    # We want to recompute self.inv_freq if it was not loaded in fp32
                    if self.inv_freq.dtype != torch.float32:
                        inv_freq = self._compute_inv_freq(device=device)
                    else:
                        inv_freq = self.inv_freq
                else:
                    t = torch.arange(seqlen, device=device, dtype=self.inv_freq.dtype)
                    inv_freq = self.inv_freq
                # Don't do einsum, it converts fp32 to fp16 under AMP
                # freqs = torch.einsum("i,j->ij", t, self.inv_freq)
                freqs = torch.outer(t, inv_freq)
                if self.scale is None:
                    self._cos_cached = torch.cos(freqs).to(dtype)
                    self._sin_cached = torch.sin(freqs).to(dtype)
                else:
                    power = (
                        torch.arange(seqlen, dtype=self.scale.dtype, device=self.scale.device)
                        - seqlen // 2
                    ) / self.scale_base
                    scale = self.scale.to(device=power.device) ** rearrange(power, "s -> s 1")
                    # We want the multiplication by scale to happen in fp32
                    self._cos_cached = (torch.cos(freqs) * scale).to(dtype)
                    self._sin_cached = (torch.sin(freqs) * scale).to(dtype)
                    self._cos_k_cached = (torch.cos(freqs) / scale).to(dtype)
                    self._sin_k_cached = (torch.sin(freqs) / scale).to(dtype)

        def forward(
            self,
            qkv: torch.Tensor,
            kv: Optional[torch.Tensor] = None,
            seqlen_offset: Union[int, torch.Tensor] = 0,
            max_seqlen: Optional[int] = None,
        ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
            """
            qkv: (batch, seqlen, 3, nheads, headdim) if kv is none,
                else it's just q of shape (batch, seqlen, nheads, headdim)
            kv: (batch, seqlen, 2, nheads, headdim)
            seqlen_offset: (batch_size,) or int. Each sequence in x is shifted by this amount.
                Most commonly used in inference when we have KV cache.
                If it's a tensor of shape (batch_size,), then to update the cos / sin cache, one
                should pass in max_seqlen, which will update the cos / sin cache up to that length.
            Apply rotary embedding *inplace* to qkv and / or kv.
            """
            seqlen = qkv.shape[1]
            if max_seqlen is not None:
                self._update_cos_sin_cache(max_seqlen, device=qkv.device, dtype=qkv.dtype)
            elif isinstance(seqlen_offset, int):
                self._update_cos_sin_cache(seqlen + seqlen_offset, device=qkv.device, dtype=qkv.dtype)
            if kv is None:
                if self.scale is None:
                    return apply_rotary_emb_qkv_(
                        qkv,
                        self._cos_cached,
                        self._sin_cached,
                        interleaved=self.interleaved,
                        seqlen_offsets=seqlen_offset,
                    )
                else:
                    return apply_rotary_emb_qkv_(
                        qkv,
                        self._cos_cached,
                        self._sin_cached,
                        self._cos_k_cached,
                        self._sin_k_cached,
                        interleaved=self.interleaved,
                        seqlen_offsets=seqlen_offset,
                    )
            else:
                q = qkv
                q = apply_rotary_emb_func(
                    q,
                    self._cos_cached,
                    self._sin_cached,
                    interleaved=self.interleaved,
                    inplace=True,
                    seqlen_offsets=seqlen_offset,
                )
                if self.scale is None:
                    kv = apply_rotary_emb_kv_(
                        kv,
                        self._cos_cached,
                        self._sin_cached,
                        interleaved=self.interleaved,
                        seqlen_offsets=seqlen_offset,
                    )
                else:
                    kv = apply_rotary_emb_kv_(
                        kv,
                        self._cos_k_cached,
                        self._sin_k_cached,
                        interleaved=self.interleaved,
                        seqlen_offsets=seqlen_offset,
                    )
                return q, kv
