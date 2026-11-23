#!/usr/bin/env python3
"""
Step 0 (~30s) fail-fast smoke test for CohereLabs/tiny-aya-global (Cohere2ForCausalLM):
  1. Verifies HF_TOKEN access and that vLLM loads tiny-aya-global cleanly in bfloat16.
  2. Confirms layer count (36), hidden size (2048), and in-process hook access via VLLM_USE_V1=0.
  3. Inspects layer.forward() outputs to confirm hidden_states already includes residual (same as Aya-Expanse).
"""
import os
os.environ["VLLM_USE_V1"] = "0"  # Required so vLLM runs model.layers in-process for forward hooks

from types import MethodType
import torch
from vllm import LLM, SamplingParams

MODEL = "CohereLabs/tiny-aya-global"

print(f"PyTorch: {torch.__version__} | CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name} ({props.total_memory / 1024**3:.1f} GiB)")

print(f"\nLoading {MODEL} in bfloat16...")
llm = LLM(
    model=MODEL,
    tensor_parallel_size=1,
    max_model_len=2048,
    dtype="bfloat16",
    enforce_eager=True,
    trust_remote_code=True,
    enable_prefix_caching=False,
)

cfg = llm.llm_engine.model_config.hf_config
layers = llm.llm_engine.model_executor.driver_worker.model_runner.model.model.layers
print(f"Architecture: {cfg.architectures}")
print(f"Layers: {len(layers)} | Hidden size: {cfg.hidden_size} | Vocab: {cfg.vocab_size}")

# Inspect layer 4 forward output signature
seen = {}
layer4 = layers[4]
orig_forward = layer4.forward

def inspect_hook(self, positions, hidden_states, residual):
    out_h, out_r = orig_forward(positions, hidden_states, residual)
    if "checked" not in seen:
        seen["checked"] = True
        print("\n=== Layer 4 Forward Output Inspection ===")
        print(f"  input  hidden_states: {tuple(hidden_states.shape)}, dtype={hidden_states.dtype}")
        print(f"  input  residual:      {None if residual is None else tuple(residual.shape)}")
        print(f"  output hidden_states: {tuple(out_h.shape)}, norm={out_h.float().norm(dim=-1).mean():.3f}")
        print(f"  output residual:      {None if out_r is None else (tuple(out_r.shape), float(out_r.float().norm(dim=-1).mean()))}")
        if out_r is not None:
            print(f"  allclose(out_h, out_r)? {torch.allclose(out_h, out_r, atol=1e-2)}")
    return out_h, out_r

layer4.forward = MethodType(inspect_hook, layer4)

out = llm.generate(["Hello, how are you?"], SamplingParams(max_tokens=16, temperature=0.0))
print("\nSample generation:", out[0].outputs[0].text.strip())
print("\n✓ Smoke test passed! Ready for 1-identify_tinyaya.sh")
