"""FlashAttention (Dao et al. 2022, arXiv 2205.14135; Dao 2023 for the backward loop order), written out.

Standard attention materializes S = Q K^T, an N x N matrix, in slow GPU memory (HBM), then reads it back for
softmax and again for P V. FlashAttention never builds S. It walks over blocks of keys/values, keeps for
every query row a running max m, a running softmax denominator l and an unnormalized output o, and rescales
them whenever a new block raises the max ("online softmax", Milakov & Gimelshein 2018):

    m_new = max(m, rowmax(S_ij))
    l     = exp(m - m_new) * l + rowsum(exp(S_ij - m_new))
    o     = exp(m - m_new) * o + exp(S_ij - m_new) @ V_j
    after the last block: O = o / l,  and L = m + log(l) is saved for the backward pass

Memory is O(N) instead of O(N^2). The backward pass recomputes each block of P from Q, K and L instead of
storing it: more FLOPs, far fewer memory reads, which is the trade the paper makes.

These PyTorch loops show the algorithm and are checked against standard attention; `triton_flash_forward`
is the same forward pass as one fused GPU kernel (CUDA only).
"""

import math

import torch


def reference_attention(q, k, v, causal: bool) -> torch.Tensor:
    s = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    if causal:
        n, m = s.shape[-2:]
        s = s.masked_fill(torch.ones(n, m, dtype=torch.bool, device=s.device).triu(1), float("-inf"))
    return s.softmax(-1) @ v


def flash_forward(q, k, v, causal: bool, block_q: int = 64, block_k: int = 64):
    """q, k, v: (..., N, d). Returns (O, L) with L the row-wise logsumexp of the scaled scores."""
    N, d = q.shape[-2:]
    scale = 1.0 / math.sqrt(d)
    O = torch.empty_like(q)
    L = torch.empty(q.shape[:-1], dtype=torch.float32, device=q.device)
    for i in range(0, N, block_q):
        qi = q[..., i : i + block_q, :].float()
        rows = torch.arange(i, i + qi.shape[-2], device=q.device)
        m = torch.full(qi.shape[:-1], float("-inf"), device=q.device)
        l = torch.zeros(qi.shape[:-1], device=q.device)
        o = torch.zeros_like(qi)
        for j in range(0, (i + qi.shape[-2]) if causal else k.shape[-2], block_k):  # causal: skip blocks fully in the future
            kj, vj = k[..., j : j + block_k, :].float(), v[..., j : j + block_k, :].float()
            s = qi @ kj.transpose(-2, -1) * scale
            if causal:
                cols = torch.arange(j, j + kj.shape[-2], device=q.device)
                s = s.masked_fill(cols[None, :] > rows[:, None], float("-inf"))
            m_new = torch.maximum(m, s.amax(-1))
            p = torch.exp(s - m_new[..., None])
            alpha = torch.exp(m - m_new)  # how much the old running sums shrink under the new max
            l = alpha * l + p.sum(-1)
            o = alpha[..., None] * o + p @ vj
            m = m_new
        O[..., i : i + qi.shape[-2], :] = (o / l[..., None]).to(q.dtype)
        L[..., i : i + qi.shape[-2]] = m + torch.log(l)
    return O, L


def flash_backward(q, k, v, O, L, dO, causal: bool, block_q: int = 64, block_k: int = 64):
    """Gradients from the saved logsumexp L only: every P block is recomputed as exp(S - L).
    dV = P^T dO;  dP = dO V^T;  dS = P * (dP - D) with D = rowsum(dO * O);  dQ = dS K * scale;  dK = dS^T Q * scale."""
    N, d = q.shape[-2:]
    scale = 1.0 / math.sqrt(d)
    D = (dO.float() * O.float()).sum(-1)
    dq, dk, dv = torch.zeros_like(q, dtype=torch.float32), torch.zeros_like(k, dtype=torch.float32), torch.zeros_like(v, dtype=torch.float32)
    for j in range(0, k.shape[-2], block_k):  # outer loop over key blocks: dK_j, dV_j accumulate locally
        kj, vj = k[..., j : j + block_k, :].float(), v[..., j : j + block_k, :].float()
        cols = torch.arange(j, j + kj.shape[-2], device=q.device)
        for i in range(j - j % block_q if causal else 0, N, block_q):
            qi, dOi = q[..., i : i + block_q, :].float(), dO[..., i : i + block_q, :].float()
            rows = torch.arange(i, i + qi.shape[-2], device=q.device)
            s = qi @ kj.transpose(-2, -1) * scale
            if causal:
                s = s.masked_fill(cols[None, :] > rows[:, None], float("-inf"))
            p = torch.exp(s - L[..., i : i + qi.shape[-2], None])
            dv[..., j : j + kj.shape[-2], :] += p.transpose(-2, -1) @ dOi
            dp = dOi @ vj.transpose(-2, -1)
            ds = p * (dp - D[..., i : i + qi.shape[-2], None])
            dq[..., i : i + qi.shape[-2], :] += ds @ kj * scale
            dk[..., j : j + kj.shape[-2], :] += ds.transpose(-2, -1) @ qi * scale
    return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype)


class FlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal: bool = False):
        O, L = flash_forward(q, k, v, causal)
        ctx.save_for_backward(q, k, v, O, L)  # O(N) extra memory: no attention matrix is kept
        ctx.causal = causal
        return O

    @staticmethod
    def backward(ctx, dO):
        q, k, v, O, L = ctx.saved_tensors
        return (*flash_backward(q, k, v, O, L, dO, ctx.causal), None)


def flash_attention(q, k, v, causal: bool = False) -> torch.Tensor:
    return FlashAttention.apply(q, k, v, causal)


try:  # the fused kernel needs Triton and an NVIDIA GPU
    import triton
    import triton.language as tl

    @triton.jit
    def _fwd_kernel(Q, K, V, Out, Lse, sm_scale, N, stride_b, stride_n, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                    D: tl.constexpr, CAUSAL: tl.constexpr):
        # One program = one block of BLOCK_M query rows of one (batch x head). Q, K, V, Out: (BH, N, D), contiguous.
        start_m, bh = tl.program_id(0), tl.program_id(1)
        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, D)
        base = bh * stride_b
        q = tl.load(Q + base + offs_m[:, None] * stride_n + offs_d[None, :], mask=offs_m[:, None] < N, other=0.0)
        m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
        l_i = tl.zeros([BLOCK_M], tl.float32)
        acc = tl.zeros([BLOCK_M, D], tl.float32)
        hi = tl.minimum((start_m + 1) * BLOCK_M, N) if CAUSAL else N
        for start_n in range(0, hi, BLOCK_N):
            cols = start_n + offs_n
            k = tl.load(K + base + cols[:, None] * stride_n + offs_d[None, :], mask=cols[:, None] < N, other=0.0)
            v = tl.load(V + base + cols[:, None] * stride_n + offs_d[None, :], mask=cols[:, None] < N, other=0.0)
            s = tl.dot(q, tl.trans(k)) * sm_scale
            s = tl.where(cols[None, :] < N, s, float("-inf"))
            if CAUSAL:
                s = tl.where(offs_m[:, None] >= cols[None, :], s, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(s, 1))
            p = tl.exp(s - m_new[:, None])
            alpha = tl.exp(m_i - m_new)
            l_i = alpha * l_i + tl.sum(p, 1)
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            m_i = m_new
        tl.store(Out + base + offs_m[:, None] * stride_n + offs_d[None, :], (acc / l_i[:, None]).to(Out.dtype.element_ty),
                 mask=offs_m[:, None] < N)
        tl.store(Lse + bh * N + offs_m, m_i + tl.log(l_i), mask=offs_m < N)

    def triton_flash_forward(q, k, v, causal: bool = False, block_m: int = 64, block_n: int = 64):
        """q, k, v: (B, H, N, D) CUDA tensors, D a power of two >= 16. Returns (O, logsumexp)."""
        B, H, N, D = q.shape
        q, k, v = (t.reshape(B * H, N, D).contiguous() for t in (q, k, v))
        out = torch.empty_like(q)
        lse = torch.empty(B * H, N, device=q.device, dtype=torch.float32)
        grid = (triton.cdiv(N, block_m), B * H)
        _fwd_kernel[grid](q, k, v, out, lse, 1.0 / math.sqrt(D), N, q.stride(0), q.stride(1),
                          BLOCK_M=block_m, BLOCK_N=block_n, D=D, CAUSAL=causal)
        return out.view(B, H, N, D), lse.view(B, H, N)

except ImportError:
    triton_flash_forward = None
