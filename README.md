# LLM.int8(), GPTQ and FlashAttention, from scratch

Three papers on making large models cheaper to run, implemented by hand in PyTorch (plus one Triton kernel)
and checked against exact references:

| Paper | What it does | Code |
|---|---|---|
| **LLM.int8()** — [Dettmers et al., 2022](https://arxiv.org/abs/2208.07339) | int8 matmul with per-row / per-column absmax scales, plus the mixed-precision decomposition that keeps "emergent outlier" feature dimensions (\|x\| > 6) in 16-bit | `llm_int8_matmul`, `Int8Linear` in [quant.py](quant.py) |
| **GPTQ** — [Frantar et al., 2023](https://arxiv.org/abs/2210.17323) | 3/4-bit weights: quantize one column at a time and push each rounding error onto the remaining columns through the Cholesky factor of the inverse Hessian H = 2XXᵀ; lazy block updates; per-group scales; layer-by-layer calibration of a real model | `gptq` in [quant.py](quant.py), `apply_gptq` in [evaluate.py](evaluate.py) |
| **FlashAttention** — [Dao et al., 2022](https://arxiv.org/abs/2205.14135) | Exact attention without the N×N matrix: tiled online softmax forward that saves only the row logsumexp, and a backward that recomputes each block | `flash_forward`, `flash_backward`, `FlashAttention` in [flash.py](flash.py) |

[evaluate.py](evaluate.py) applies each method to a real model (Qwen2.5-0.5B, loaded into a hand-written
Llama/Qwen2 in [llama.py](llama.py)) and measures WikiText-2 perplexity with the GPTQ paper's protocol.

## Results: Qwen2.5-0.5B on WikiText-2

Test perplexity over 298,938 tokens in non-overlapping 2,048-token windows (lower is better). GPTQ and RTN
quantize every linear layer's weights with group size 128; GPTQ is calibrated on 32 random 2,048-token windows
of the training split. M1 Pro (MPS).

| Method | Weight bits | Perplexity | vs. 16-bit | Minutes |
|---|---|---|---|---|
| 16-bit (reference) | 16 | **12.98** | | 2.6 |
| LLM.int8() | 8 | **13.02** | +0.04 | 11.8 |
| Round-to-nearest | 4 | 15.46 | +2.48 | 2.3 |
| GPTQ | 4 | **14.00** | +1.02 | 4.7 |
| Round-to-nearest | 3 | 50.50 | +37.52 | 2.6 |
| GPTQ | 3 | **19.51** | +6.53 | 4.8 |

- LLM.int8() is within 0.04 of 16-bit: the paper's claim of no degradation holds on this model.
- At 4 bits GPTQ cuts round-to-nearest's perplexity loss by 59% (2.48 → 1.02); at 3 bits by 83%
  (37.5 → 6.5), where plain rounding breaks the model and GPTQ keeps it usable.
- The int8 time is the cost of doing the decomposition in plain PyTorch with no int8 kernel; the paper's
  speed comes from CUDA kernels this repo does not have.

> **Status:** all code complete; `check.py` passes; the perplexity table above is measured. A GPU run of the
> Triton kernel is still to do.

## What the checks prove

`python check.py` (CPU, a few seconds):

| Check | Result |
|---|---|
| `gptq_with_identity_hessian_is_rtn` | With uncorrelated inputs (H = I) there is nothing to compensate: GPTQ returns exactly round-to-nearest |
| `gptq_beats_rtn_on_correlated_inputs` | On correlated inputs GPTQ minimizes what it should, the layer's **output** error: mean squared error 63.5 → 31.4 at 4-bit and 290.7 → 151.0 at 3-bit, versus round-to-nearest |
| `llm_int8` | Vector-wise int8 matmul is within ~1% of float; one feature dimension scaled 60× pushes plain int8 to 4.7% error, the outlier decomposition brings it back to 0.36% |
| `flash_forward_equals_attention` | Output and logsumexp equal standard attention, causal and not, block sizes that do not divide the sequence length |
| `flash_backward_equals_autograd` | dQ, dK, dV from the recomputation backward equal PyTorch autograd through standard attention (to float32 precision) |
| `rtn_error_is_within_half_a_step` | Round-to-nearest never errs by more than half a quantization step |
| `triton_kernel` | The fused Triton forward against standard attention, **only on an NVIDIA GPU** (skipped on this Mac; not yet run) |

Removing the online-softmax rescaling, the D = rowsum(dO ∘ O) term of the backward, or GPTQ's lazy block
update each makes a check fail.

## Run it

```bash
pip install -r requirements.txt
python check.py
uv run --with pyarrow python evaluate.py --prepare          # WikiText-2 raw text, once
python evaluate.py --methods fp rtn4 gptq4 rtn3 gptq3 int8  # perplexity table
```

## Notes and differences

- Quantization here is "fake": weights are rounded to the integer grid and stored dequantized, which gives the
  exact numerical effect of the int codes; packing into 4-bit storage and integer kernels are not included.
- The int8 x int8 product is emulated in float32 (every term is an integer below 127²; exact up to 2²⁴).
- GPTQ calibration uses WikiText-2 train windows; the paper used 128 C4 samples.
- The Triton kernel is the forward pass only; the backward runs through the PyTorch recomputation.

## References

- Dettmers, Lewis, Belkada & Zettlemoyer, 2022. *LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale.* [arXiv:2208.07339](https://arxiv.org/abs/2208.07339)
- Frantar, Ashkboos, Hoefler & Alistarh, 2023. *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers.* [arXiv:2210.17323](https://arxiv.org/abs/2210.17323)
- Dao, Fu, Ermon, Rudra & Ré, 2022. *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.* [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)
- Dao, 2023. *FlashAttention-2.* [arXiv:2307.08691](https://arxiv.org/abs/2307.08691)
- Milakov & Gimelshein, 2018. *Online normalizer calculation for softmax.* [arXiv:1805.02867](https://arxiv.org/abs/1805.02867)
