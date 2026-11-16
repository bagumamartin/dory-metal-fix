#!/usr/bin/env bash
# Build the Dory llama.cpp branch with the Metal fix and benchmark 1, 2 and 4 loops.
# Usage: ./bench.sh /path/to/NVIDIA-Nemotron-Labs-Dory1.0-4B-Q5_K_M.gguf [cooldown_seconds]
set -euo pipefail

MODEL="${1:?usage: $0 MODEL.gguf [cooldown_seconds]}"
COOLDOWN="${2:-90}"
HERE="$(cd "$(dirname "$0")" && pwd)"
REV=98a74c97ea09ac72f6ceda1559808fc0a2f447ce
SRC="$HERE/llama.cpp-dory"

if [ ! -d "$SRC" ]; then
  git init -q "$SRC"
  git -C "$SRC" remote add origin https://github.com/brianchmielNV/llama.cpp.git
  git -C "$SRC" fetch -q --depth 1 origin "$REV"
  git -C "$SRC" checkout -q --detach FETCH_HEAD
  git -C "$SRC" apply "$HERE/dory-metal-fix.patch"
fi

cmake -S "$SRC" -B "$SRC/build" -DCMAKE_BUILD_TYPE=Release \
  -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=Apple \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_OPENSSL=OFF >/dev/null
cmake --build "$SRC/build" --parallel --target llama-completion >/dev/null

PROMPT=$(printf 'Explain, step by step, how a CPU cache hierarchy works and why it matters for performance. %.0s' $(seq 14))

for L in 1 2 4; do
  echo "== loops=$L"
  "$SRC/build/bin/llama-completion" -m "$MODEL" -ngl 999 -c 4096 -b 512 -ub 512 \
    --flash-attn off -p "$PROMPT" -n 128 --ignore-eos --temp 0 --no-warmup -no-cnv \
    --override-kv "dory.num_loops=int:$L,dory.recurrent_kv_cache_mode=str:per_loop" \
    </dev/null 2>&1 | grep -E "eval time|splits =" | sed -E 's/^[0-9.]+ [A-Z] //'
  [ "$L" = 4 ] || sleep "$COOLDOWN"
done
