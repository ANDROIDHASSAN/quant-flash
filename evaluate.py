"""WikiText-2 perplexity of a real model under each quantization method (the GPTQ paper's Table 3 setup:
2048-token windows over the test set, 4-bit and 3-bit weights, groups of 128).

    uv run --with pyarrow python evaluate.py --prepare      # once: WikiText-2 raw -> data/*.txt
    python evaluate.py --methods fp rtn4 gptq4 rtn3 gptq3 int8

GPTQ runs layer by layer, as in the paper: calibration inputs go through layer i, the Hessian of each linear
layer is accumulated from its inputs, the layer is quantized, and the QUANTIZED layer's outputs become layer
i+1's inputs, so later layers learn to compensate for earlier rounding.
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn

from llama import LM, load_pretrained, tokenizer
from quant import Int8Linear, gptq, rtn

DATA = Path(__file__).parent / "data"


def prepare() -> None:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    DATA.mkdir(exist_ok=True)
    for split in ("train", "test"):
        path = hf_hub_download("Salesforce/wikitext", f"wikitext-2-raw-v1/{split}-00000-of-00001.parquet", repo_type="dataset")
        (DATA / f"wikitext2-{split}.txt").write_text("".join(pq.read_table(path).column("text").to_pylist()))


def linears(layer: nn.Module) -> dict[str, nn.Linear]:
    return {n: m for n, m in layer.named_modules() if isinstance(m, nn.Linear)}


@torch.no_grad()
def perplexity(model: LM, ids: torch.Tensor, seqlen: int, device: str) -> float:
    nll, count = 0.0, 0
    for i in range(0, len(ids) - seqlen, seqlen):  # non-overlapping windows, as in the GPTQ paper
        x = ids[i : i + seqlen][None].to(device)
        logits = model(x)
        nll += torch.nn.functional.cross_entropy(logits[0, :-1].float(), x[0, 1:], reduction="sum").item()
        count += seqlen - 1
    return float(torch.exp(torch.tensor(nll / count)))


@torch.no_grad()
def apply_rtn(model: LM, bits: int, group: int) -> None:
    for layer in model.model.layers:
        for lin in linears(layer).values():
            lin.weight.copy_(rtn(lin.weight.float(), bits, group).to(lin.weight.dtype))


@torch.no_grad()
def apply_gptq(model: LM, calib: torch.Tensor, bits: int, group: int, device: str) -> None:
    """calib: (n, seqlen) token ids."""
    x = model.model.embed_tokens(calib.to(device))
    pos = torch.arange(calib.shape[1], device=device)
    for li, layer in enumerate(model.model.layers):
        mods = linears(layer)
        H = {n: torch.zeros(m.in_features, m.in_features, device=device) for n, m in mods.items()}

        def hook(name):
            def accumulate(_, inp, __):
                a = inp[0].reshape(-1, inp[0].shape[-1]).float()
                H[name] += 2 * a.T @ a  # H = 2 X X^T, summed over every calibration token
            return accumulate

        handles = [m.register_forward_hook(hook(n)) for n, m in mods.items()]
        for b in range(len(x)):
            layer(x[b : b + 1], pos, None)
        for h in handles:
            h.remove()
        for n, m in mods.items():
            m.weight.copy_(gptq(m.weight.float(), H[n] / len(x), bits, group).to(m.weight.dtype))
        x = torch.cat([layer(x[b : b + 1], pos, None) for b in range(len(x))])  # quantized outputs feed the next layer
        print(f"  gptq layer {li + 1}/{len(model.model.layers)}", flush=True)


def apply_int8(model: LM, threshold: float) -> None:
    for layer in model.model.layers:
        for name, lin in linears(layer).items():
            parent = layer.get_submodule(name.rsplit(".", 1)[0])
            setattr(parent, name.rsplit(".", 1)[1], Int8Linear(lin, threshold))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--methods", nargs="+", default=["fp", "rtn4", "gptq4", "rtn3", "gptq3", "int8"])
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--nsamples", type=int, default=32, help="calibration windows (the paper used 128 from C4)")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = ap.parse_args()
    if args.prepare:
        prepare()
        return

    tok = tokenizer(args.model)
    test = torch.tensor(tok.encode((DATA / "wikitext2-test.txt").read_text()).ids)
    train = torch.tensor(tok.encode((DATA / "wikitext2-train.txt").read_text()).ids)
    g = torch.Generator().manual_seed(0)
    starts = torch.randint(0, len(train) - args.seqlen, (args.nsamples,), generator=g)
    calib = torch.stack([train[s : s + args.seqlen] for s in starts])
    print(f"{args.model}: {len(test):,} test tokens, {args.nsamples} x {args.seqlen} calibration tokens")
    print("| method | bits | WikiText-2 perplexity | minutes |\n|---|---|---|---|")
    for method in args.methods:
        t0 = time.time()
        model = load_pretrained(args.model, torch.float32, args.device)
        bits = {"fp": 16, "int8": 8}.get(method) or int(method[-1])
        if method.startswith("rtn"):
            apply_rtn(model, bits, args.group)
        elif method.startswith("gptq"):
            apply_gptq(model, calib, bits, args.group, args.device)
        elif method == "int8":
            apply_int8(model, 6.0)
        ppl = perplexity(model, test, args.seqlen, args.device)
        print(f"| {method} | {bits} | {ppl:.2f} | {(time.time() - t0) / 60:.1f} |", flush=True)
        del model


if __name__ == "__main__":
    main()
