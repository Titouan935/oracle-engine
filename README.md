# ORACLE

An LLM inference engine written from scratch in C++17. Zero external dependencies.

No PyTorch. No BLAS. No llama.cpp. No ggml. Just the standard library (and,
optionally, OpenMP for multi-threaded attention).

## What it does

- **GGUF parser** — reads quantized model files directly, memory-mapped, no conversion step
- **Quantization** — direct GEMV/GEMM on quantized weights for Q4_0, Q8_0, Q4_K, Q6_K (plus F16/BF16), no full dequant-to-float expansion in the hot path
- **Transformer** — Grouped-Query Attention, RoPE, SwiGLU, RMSNorm, KV cache
- **Tokenizer** — a from-scratch BPE implementation (GPT-2 byte encoding) driven by the GGUF vocab
- **Sampler** — temperature, top-k, nucleus (top-p), repeat penalty
- **Speculative decoding** — optional draft model + verification pass
- **Embeddings** — mean-pooled hidden states + L2 norm, for retrieval

## What it does not do

- No GPU backend — CPU only
- No batching server / no continuous batching
- Tested against Llama 3.2 3B and Qwen2.5 7B (GGUF) only
- Not production-hardened. This is a systems exercise that turned into a working engine.

## Build

Requires a C++17 compiler. Nothing else.

**macOS / Linux:**

```bash
make            # builds ./oracle, ./oracle_bench, ./micro_gemm
# make OPENMP=0 # build without OpenMP (single-threaded attention, truly zero deps)
```

**Any platform (CMake):**

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

On Windows with MinGW-w64, the CMake path is the supported one.

## Run

```bash
./oracle -m models/your-model.gguf -p "Explain what a KV cache is."
```

Options: `-n <tokens>`, `--temp <f>`, `--top-p <f>`, `--system <text>`,
`--draft <draft.gguf>` (speculative decoding), `--no-stream`.

Bring your own GGUF weights (e.g. from Hugging Face). None are bundled.

## Performance

See [BENCHMARKS.md](BENCHMARKS.md) — with the exact machine, model, method, and a
reproducible harness (`./oracle_bench`). The honest notes there include where this
engine is slower than llama.cpp and why.

The engine is **memory-bandwidth bound**: throughput scales with how fast the
machine can read the model weights, not with raw compute. The isolated quantized
matmul kernel already runs near the machine's compute ceiling; the wins live in
memory traffic.

## Why

I wanted to know what actually happens between a prompt and a token. Importing a
library teaches you its API; writing the kernel teaches you where the time goes.

## License

Copyright (C) 2026 Titouan Ronalson SENATUS MANDIN

ORACLE is licensed under the **GNU Affero General Public License v3.0**
(AGPL-3.0) — see [LICENSE](LICENSE). In short: you may use, study, modify and
redistribute it, but any distributed **or network-served** derivative must also
be released under AGPL-3.0, with its source made available to users.

**Commercial licensing:** a separate commercial license — without AGPL's
copyleft obligations — is available for organizations that want to use ORACLE in
proprietary or closed-source products. Contact the author.
