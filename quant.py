"""Weight quantization: round-to-nearest, LLM.int8() and GPTQ, written out.

Round-to-nearest (RTN), asymmetric, per group of `group` input columns:
    scale = (max - min) / (2^bits - 1),  zero = round(-min / scale)
    q = clamp(round(w / scale) + zero, 0, 2^bits - 1),  w_hat = (q - zero) * scale

LLM.int8() (Dettmers et al. 2022, arXiv 2208.07339), section 3:
    vector-wise int8: each row of X and each output row of W gets its own absmax scale, the product is an
    int8 x int8 matmul rescaled by the outer product of the scales; and the mixed-precision decomposition:
    the few feature dimensions where any activation exceeds 6.0 ("emergent outliers") are multiplied in 16-bit.

GPTQ (Frantar et al. 2023, arXiv 2210.17323), Algorithm 1:
    quantize one column at a time and push its rounding error onto the columns not yet quantized, weighted
    by the inverse Hessian of the layer's reconstruction error, H = 2 X X^T. The quantized layer then
    reproduces the original layer's OUTPUTS on calibration data, not just its weights.
"""

import torch
import torch.nn as nn


def rtn_params(w: torch.Tensor, bits: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row asymmetric scale and zero point for w: (rows, cols)."""
    qmax = 2**bits - 1
    lo, hi = w.amin(1, keepdim=True).clamp(max=0), w.amax(1, keepdim=True).clamp(min=0)
    scale = ((hi - lo) / qmax).clamp(min=1e-8)
    return scale, torch.round(-lo / scale)


def quantize(w: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor, bits: int) -> torch.Tensor:
    """Round to the grid and map back (fake quantization: the values are exactly what int codes would give)."""
    q = torch.clamp(torch.round(w / scale) + zero, 0, 2**bits - 1)
    return (q - zero) * scale


def rtn(w: torch.Tensor, bits: int, group: int = 128) -> torch.Tensor:
    out = torch.empty_like(w)
    for g in range(0, w.shape[1], group):
        scale, zero = rtn_params(w[:, g : g + group], bits)
        out[:, g : g + group] = quantize(w[:, g : g + group], scale, zero, bits)
    return out


@torch.no_grad()
def gptq(w: torch.Tensor, H: torch.Tensor, bits: int, group: int = 128, block: int = 128, damp: float = 0.01) -> torch.Tensor:
    """w: (out, in); H: (in, in) = 2 X X^T summed over calibration inputs. Returns the quantized weight."""
    W, H = w.clone().float(), H.clone().float()
    dead = torch.diag(H) == 0  # input features that were always zero: their weights do not matter
    H[dead, dead] = 1.0
    W[:, dead] = 0.0
    H += damp * torch.diag(H).mean() * torch.eye(len(H), device=H.device)  # dampening, as in the paper
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)  # upper factor of H^-1
    Q = torch.zeros_like(W)
    cols = W.shape[1]
    scale = zero = None
    for i in range(0, cols, block):
        end = min(i + block, cols)
        W1, Hinv1 = W[:, i:end].clone(), Hinv[i:end, i:end]
        E = torch.zeros_like(W1)
        for j in range(end - i):
            c = i + j
            if c % group == 0:  # scales of the next group, from the weights after all earlier blocks' updates
                scale, zero = rtn_params(W[:, c : c + group], bits)
            q = quantize(W1[:, j : j + 1], scale, zero, bits)
            Q[:, c : c + 1] = q
            err = (W1[:, j : j + 1] - q) / Hinv1[j, j]
            W1[:, j + 1 :] -= err @ Hinv1[j : j + 1, j + 1 :]  # spread the error over the rest of the block
            E[:, j : j + 1] = err
        W[:, end:] -= E @ Hinv[i:end, end:]  # lazy batch update of all later columns
        W[:, i:end] = W1
    return Q.to(w.dtype)


def int8_vectorwise(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Absmax int8 per row: x ~= q * scale, q in [-127, 127]."""
    scale = (x.abs().amax(-1, keepdim=True) / 127.0).clamp(min=1e-8)
    return torch.round(x / scale), scale


def llm_int8_matmul(x: torch.Tensor, w: torch.Tensor, threshold: float = 6.0) -> torch.Tensor:
    """y = x @ w^T with LLM.int8(). x: (N, K) activations; w: (M, K) weight.
    The int8 x int8 product is emulated in float32: every term is an integer below 127^2, so the sum is the
    integer GEMM result (exactly while it stays under 2^24, to float32 rounding beyond)."""
    outliers = (x.abs() > threshold).any(0) if threshold else torch.zeros(x.shape[1], dtype=torch.bool, device=x.device)
    xq, sx = int8_vectorwise(x[:, ~outliers])
    wq, sw = int8_vectorwise(w[:, ~outliers])
    y = (xq @ wq.T) * sx * sw.T  # dequantize with the outer product of row and column scales
    if outliers.any():
        y = y + x[:, outliers] @ w[:, outliers].T  # the outlier dimensions in full precision
    return y


class Int8Linear(nn.Module):
    """Drop-in nn.Linear replacement running LLM.int8()."""

    def __init__(self, linear: nn.Linear, threshold: float = 6.0) -> None:
        super().__init__()
        self.weight, self.bias, self.threshold = linear.weight, linear.bias, threshold

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        y = llm_int8_matmul(x.reshape(-1, shape[-1]).float(), self.weight.float(), self.threshold)
        y = y.reshape(*shape[:-1], -1).to(x.dtype)
        return y + self.bias if self.bias is not None else y
