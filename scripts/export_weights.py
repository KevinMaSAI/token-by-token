"""Export a small GPT-2 architecture to the hand-written browser Transformer.

This OFFLINE conversion utility may use torch/transformers. Neither package is
shipped to, imported by, or required by the web app. No backend service is used.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export GPT-2 weights and original byte BPE")
    parser.add_argument("--model", default="sshleifer/tiny-gpt2")
    parser.add_argument("--revision", default="main", help="Pin a model commit for reproducibility")
    parser.add_argument("--out", type=Path, default=Path("public/models"))
    parser.add_argument("--context", type=int, default=32)
    args = parser.parse_args()
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise SystemExit("Install conversion-only dependencies: pip install torch transformers huggingface_hub") from exc

    model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision, trust_remote_code=False).cpu().eval()
    hf = model.config
    if hf.model_type != "gpt2":
        raise SystemExit("Only the GPT-2 architecture is supported. Pythia/Qwen have different norms, gates and weight layouts; do not rename their weights to bypass this check.")
    d, ff, layers, heads = hf.n_embd, hf.n_inner or 4 * hf.n_embd, hf.n_layer, hf.n_head
    if d > 128 or ff > 512 or layers > 12:
        raise SystemExit("Model is too large for the inspectable JSON frontend. Use sshleifer/tiny-gpt2 or a compatible GPT-2 checkpoint with d<=128, FF<=512, layers<=12.")
    if hf.activation_function not in ("gelu_new", "gelu_fast", "gelu_pytorch_tanh"):
        raise SystemExit(f"Unsupported exact activation: {hf.activation_function}")
    if getattr(hf, "scale_attn_by_inverse_layer_idx", False) or not getattr(hf, "scale_attn_weights", True):
        raise SystemExit("Unsupported attention scaling convention")
    context = max(1, min(args.context, hf.n_positions, 128))
    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision, use_fast=False, trust_remote_code=False)
    raw_vocab = tok.get_vocab()
    vocab = [""] * hf.vocab_size
    for token, idx in raw_vocab.items():
        if idx >= len(vocab):
            raise SystemExit("Added tokenizer vocabulary has no matching model embedding")
        vocab[idx] = token
    if any(not s for s in vocab):
        raise SystemExit("The tokenizer vocabulary has gaps")
    merges_path = hf_hub_download(args.model, "merges.txt", revision=args.revision)
    merges = [line.split() for line in Path(merges_path).read_text(encoding="utf-8").splitlines() if line and not line.startswith("#")]
    if any(len(p) != 2 for p in merges):
        raise SystemExit("Invalid BPE merge file")
    sd, weights = model.state_dict(), {}

    def put(key: str, tensor) -> None:
        weights[key] = tensor.detach().float().contiguous().reshape(-1).tolist()

    put("embedding", sd["transformer.wte.weight"])
    put("position", sd["transformer.wpe.weight"][:context])
    put("final.gamma", sd["transformer.ln_f.weight"])
    put("final.beta", sd["transformer.ln_f.bias"])
    put("lm", sd["lm_head.weight"].T)
    put("lmBias", torch.zeros(hf.vocab_size))
    for layer in range(layers):
        src, dst = f"transformer.h.{layer}", f"L{layer}"
        # HF GPT-2 Conv1D weights already have [input, output] orientation.
        for i, name in enumerate(("q", "k", "v")):
            put(f"{dst}.{name}", sd[f"{src}.attn.c_attn.weight"][:, i*d:(i+1)*d])
            put(f"{dst}.{name}Bias", sd[f"{src}.attn.c_attn.bias"][i*d:(i+1)*d])
        put(f"{dst}.o", sd[f"{src}.attn.c_proj.weight"])
        put(f"{dst}.oBias", sd[f"{src}.attn.c_proj.bias"])
        for i in (1, 2):
            put(f"{dst}.n{i}.gamma", sd[f"{src}.ln_{i}.weight"])
            put(f"{dst}.n{i}.beta", sd[f"{src}.ln_{i}.bias"])
        put(f"{dst}.w1", sd[f"{src}.mlp.c_fc.weight"])
        put(f"{dst}.b1", sd[f"{src}.mlp.c_fc.bias"])
        put(f"{dst}.w2", sd[f"{src}.mlp.c_proj.weight"])
        put(f"{dst}.b2", sd[f"{src}.mlp.c_proj.bias"])
        # Dormant comparison parameters: the GPT-2 forward pass never uses SwiGLU.
        put(f"{dst}.gate", torch.zeros(d, ff))
        put(f"{dst}.gateBias", torch.zeros(ff))
    config = dict(vocabSize=hf.vocab_size, layers=layers, heads=heads, d=d, ff=ff, maxSeq=context, name=args.model, bosId=hf.bos_token_id, eosId=hf.eos_token_id)
    options = dict(position="learned", causal=True, scaled=True, attentionTemperature=1, dropout=0, norm="layer", normOrder="pre", eps=hf.layer_norm_epsilon, activation="gelu", ffWidth=ff)
    meta = dict(format="token-by-token-v1", config=config, vocab=vocab, tokenizer=dict(vocab=vocab, merges=merges, pattern="gpt2"), options=options, source=args.model, revision=getattr(hf, "_commit_hash", None) or args.revision)
    args.out.mkdir(parents=True, exist_ok=True)
    for name, value in (("config.json", meta), ("weights.json", weights), ("model.json", {**meta, "weights": weights})):
        (args.out / name).write_text(json.dumps(value, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    # This independent reference is useful for testing the JS export numerically.
    prompt = "hello world"
    batch = tok(prompt, return_tensors="pt")
    with torch.no_grad():
        logits = model(**batch).logits[0, -1].float().tolist()
    (args.out / "reference.json").write_text(json.dumps(dict(prompt=prompt, ids=batch.input_ids[0].tolist(), logits=logits)), encoding="utf-8")
    print(f"Exported {args.model}: {sum(map(len, weights.values())):,} stored parameters, context={context}.")
    print("Start npm run dev, then select the pretrained model; model.json can also be imported.")
    print("Changing position/activation/norm options creates an intervention and no longer reproduces the checkpoint.")


if __name__ == "__main__":
    main()
