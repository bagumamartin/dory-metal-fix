# Dory on Apple Silicon: 2.6–4.5× faster than the model card's pinned build

> [!NOTE]
> **This bug is already fixed upstream.** The Dory llama.cpp PR
> ([ggml-org/llama.cpp#30306](https://github.com/ggml-org/llama.cpp/pull/30306)) fixed it on
> 2026-10-20 in commit `9e2f3be` ("fix incorrect CPU placement of Dory's BF16 scale tensors") by
> brianchmielNV, using the same approach as the patch here.
>
> NVIDIA's model card, however, still tells Mac users to build revision `98a74c9`, which predates
> that fix. If you follow the model card, you get the slow build measured below.
>
> **What to do:** build the PR's current head (or any revision at or after `9e2f3be`) instead of
> `98a74c9`. The patch in this repo is a backport for anyone who needs to stay on `98a74c9`.

I found and fixed the bug independently while benchmarking Dory on an M2 MacBook Air, before
noticing the upstream fix. The numbers below compare `98a74c9` with and without the backport.
Outputs are byte-identical.

## Results — MacBook Air M2, 8 GB

Model: `NVIDIA-Nemotron-Labs-Dory1.0-4B-Q5_K_M.gguf` (3.67 GB, SHA-256 verified).
Test: 268-token prompt, 128 generated tokens, greedy, per-loop KV cache, 90 s cool-down between runs.

| Loops | Generation before | Generation after | Speedup | Prefill after |
|---|---|---|---|---|
| 1 | 6.6 tok/s | **17.5 tok/s** | 2.6× | ~195 tok/s |
| 2 (recommended) | 3.4 tok/s | **12.7 tok/s** | 3.7× | ~137 tok/s |
| 4 | 1.8 tok/s | **8.1 tok/s** | 4.5× | ~80 tok/s |

Greedy outputs were compared byte-for-byte between both builds at 1, 2 and 4 loops: identical.

## The bug

Dory's per-layer scale vectors (`residual_alpha`, `attn_qk_scale`, `attn_gate_scale`,
`ffn_gate_scale`, `ffn_up_scale`, `output_scale`) are stored as BF16 and cast to F32 before use.
`src/llama-arch.cpp` declares them with the op hint `GGML_OP_MUL`. At load time llama.cpp asks each
backend whether it can run `MUL` with that BF16 weight. Metal can't, so the weights stay in CPU
memory, and the scheduler then runs every BF16→F32 cast on the CPU:

```
before: sched_reserve: graph: nodes = 2930, splits = 435
after:  sched_reserve: graph: nodes = 4555, splits = 2
```

435 GPU↔CPU round trips per token. CUDA accepts the BF16 `MUL` check, which is likely why this
doesn't show up on NVIDIA hardware.

## The fix (backport of upstream `9e2f3be`)

1. Declare those six tensors with `GGML_OP_CPY`, which is what the graph actually does with them.
2. Add a `GGML_OP_CPY` case to `weight_buft_supported` in `src/llama-model-loader.cpp` that tests a
   cast to F32.

Upstream `9e2f3be` makes the same two changes. See [`dory-metal-fix.patch`](dory-metal-fix.patch)
to apply them to `98a74c9`.

## Reproduce

```bash
./bench.sh /path/to/NVIDIA-Nemotron-Labs-Dory1.0-4B-Q5_K_M.gguf
```

The script fetches `98a74c9` (the model card's pinned revision), applies the backport, builds with
Metal and runs the benchmark.
Get the GGUF from
[nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-GGUF](https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-GGUF)
(SHA-256 `96080be49ffd67b916018b7bdef8c484762576fb52780e10ab98ac4978a660e0`).

## Notes

- The MacBook Air is fanless; back-to-back runs throttle (2 loops fell to ~7 tok/s without cool-down).
- Raw completion without the chat template tends to repeat the prompt; use `llama-server --jinja`
  for real chat.
- Only tested on an M2 (Metal). Not tested on CUDA or other backends.
