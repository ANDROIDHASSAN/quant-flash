"""Checks for the three papers. CPU, about 20 seconds, no downloads.

    python check.py
"""

import torch

from flash import flash_attention, flash_forward, reference_attention, triton_flash_forward
from quant import gptq, llm_int8_matmul, rtn

torch.manual_seed(0)


def check_rtn_error_is_within_half_a_step() -> None:
    w = torch.randn(16, 256)
    for bits in (8, 4, 3):
        err = (rtn(w, bits, 128) - w).abs()
        steps = torch.stack([(w[:, g : g + 128].amax(1) - w[:, g : g + 128].amin(1)) / (2**bits - 1) for g in (0, 128)], 1)
        assert (err <= steps.repeat_interleave(128, 1) / 2 + 1e-5).all(), bits


def check_gptq_with_identity_hessian_is_rtn() -> None:
    """Uncorrelated inputs (H = I): no error is worth pushing to other columns, so GPTQ must equal RTN."""
    w = torch.randn(32, 256)
    assert torch.allclose(gptq(w, torch.eye(256), 4, 128, damp=0.0), rtn(w, 4, 128), atol=1e-6)


def check_gptq_beats_rtn_on_correlated_inputs() -> None:
    """What GPTQ optimizes: the layer's output error ||W X - Q X|| on the calibration inputs."""
    w = torch.randn(64, 256)
    mix = torch.randn(256, 256) * 0.3 + torch.eye(256)
    x = torch.randn(256, 2048).T @ mix  # (tokens, features) with correlated features, as real activations are
    H = 2 * x.T @ x / len(x)
    out = lambda q: ((x @ w.T - x @ q.T) ** 2).mean().item()
    for bits in (4, 3):
        e_rtn, e_gptq = out(rtn(w, bits, 128)), out(gptq(w, H, bits, 128))
        assert e_gptq < 0.7 * e_rtn, (bits, e_rtn, e_gptq)


def check_llm_int8() -> None:
    """Without outliers, vector-wise int8 is accurate. One huge feature dimension ruins plain int8 (it sets the
    scale of every row); the mixed-precision decomposition fixes it."""
    x, w = torch.randn(64, 512), torch.randn(256, 512) * 0.02
    exact = x @ w.T
    rel = lambda y: ((y - exact).norm() / exact.norm()).item()
    assert rel(llm_int8_matmul(x, w)) < 0.02  # ~1%: an int8 step is absmax / 127
    x[:, 7] *= 60  # an emergent outlier feature, as in the paper's Figure 3
    exact = x @ w.T
    plain, decomposed = rel(llm_int8_matmul(x, w, threshold=0)), rel(llm_int8_matmul(x, w, threshold=6.0))
    assert decomposed < 0.01 and plain > 5 * decomposed, (plain, decomposed)  # measured: 4.7% vs 0.36%


def check_flash_forward_equals_attention() -> None:
    for N, causal, bq, bk in [(100, False, 32, 16), (100, True, 32, 16), (64, True, 64, 64), (37, True, 16, 8)]:
        q, k, v = (torch.randn(2, 3, N, 32) for _ in range(3))
        O, L = flash_forward(q, k, v, causal, bq, bk)
        assert torch.allclose(O, reference_attention(q, k, v, causal), atol=1e-5), (N, causal, bq, bk)
        s = q @ k.transpose(-2, -1) / 32**0.5
        if causal:
            s = s.masked_fill(torch.ones(N, N, dtype=torch.bool).triu(1), float("-inf"))
        assert torch.allclose(L, s.logsumexp(-1), atol=1e-5)


def check_flash_backward_equals_autograd() -> None:
    for causal in (False, True):
        q, k, v = (torch.randn(2, 2, 90, 16, dtype=torch.float64, requires_grad=True) for _ in range(3))
        dO = torch.randn(2, 2, 90, 16, dtype=torch.float64)
        ours = torch.autograd.grad(flash_attention(q, k, v, causal), (q, k, v), dO)
        ref = torch.autograd.grad(reference_attention(q, k, v, causal), (q, k, v), dO)
        for a, b in zip(ours, ref):  # the loops accumulate in float32: agreement to ~1e-6
            assert torch.allclose(a, b, atol=1e-5), causal


def check_triton_kernel() -> str | None:
    if triton_flash_forward is None or not torch.cuda.is_available():
        return "skip"  # needs Triton and an NVIDIA GPU
    for causal in (False, True):
        q, k, v = (torch.randn(2, 4, 300, 64, device="cuda", dtype=torch.float16) for _ in range(3))
        O, _ = triton_flash_forward(q, k, v, causal)
        assert torch.allclose(O.float(), reference_attention(q.float(), k.float(), v.float(), causal), atol=2e-2), causal


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("check_"):
            print(f"{fn() or 'ok'}  {name[6:]}")
