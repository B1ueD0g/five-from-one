# Build and run the native numeric audit

The application implements the archived five-slot cyclic GSM8K collector. The original host was a Mac with Apple M3 Pro and 36 GiB of unified memory. Evaluation used Metal, a 192-token cap, top-k 8, temperature 1, and the final-line stopping rule. For an offline verification of the recorded results, run `python3 -B verify_release.py` from the repository root.

## External dependencies

- llama.cpp headers and macOS arm64 runtime, tag [b12428](https://github.com/ggml-org/llama.cpp/releases/tag/b12428), commit `b4c0e00a4fcc0287d0bb32fd7cb0d8b286af73ed`. Extract the release library directory with its symlinks intact. The captured backend used `libllama.1.0.0.dylib` SHA-256 `1b494800d12422ae0ce7f8b063947aa7143673d26bb95a426b576f19b32d2300` and `libggml-metal.1.0.0.dylib` SHA-256 `bdda07421fd0ec125e35a37d03332fb001dce22f86b36b2bcb070df4ed6b3ef3`.
- [Qwen2.5-3B-Instruct-GGUF Q4_K_M](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/blob/main/qwen2.5-3b-instruct-q4_k_m.gguf), SHA-256 `626b4a6678b86442240e33df819e00132d3ba7dddfe1cdc4fbb18e0a9615c62d`. Obtain the checkpoint directly from the maintainer under its research license. Check the three digests before attempting a native run.

With the release and model already downloaded, set `LLAMA_LIB` to the absolute release library directory and `MODEL` to the absolute GGUF path. From the artifact repository root:

```sh
git clone --depth 1 --branch b12428 https://github.com/ggml-org/llama.cpp.git build/llama.cpp
clang++ -std=c++17 -O3 -Wall -Wextra -Werror \
  -I build/llama.cpp/include -I build/llama.cpp/ggml/include \
  native/run_votes_cyclic.cpp -L "$LLAMA_LIB" -lllama \
  -Wl,-rpath,"$LLAMA_LIB" -o build/run_votes_cyclic

mkdir -p build/captures
build/run_votes_cyclic "$MODEL" qwen25-3b native/serial.tsv build/captures/serial.jsonl 0 120 192 serial final-line
build/run_votes_cyclic "$MODEL" qwen25-3b native/parallel.tsv build/captures/parallel.jsonl 0 30 192 parallel final-line
build/run_votes_cyclic "$MODEL" qwen25-3b native/parallel.tsv build/captures/repeat.jsonl 0 30 192 parallel final-line
build/run_votes_cyclic "$MODEL" qwen25-3b native/reverse.tsv build/captures/reverse.jsonl 0 30 192 parallel final-line
```

Each row runs both the separate and shared-prefix modes. Each command starts a fresh process and the collector rejects an output file that already exists. The TSV fields encode batch ID, item ID, split, row position, cold-control flag, UTF-8 prompt in hex and the five tape source rows. Convert or compare new outputs in a separate directory and retain exit status and hardware details. The archive checks always run on the frozen captures. Timing and other execution-dependent fields can change in a fresh native run.

The C++ collector is the original archived source, SHA-256 `dc3ec010a5ef905a7464ccd307e3d2ca5752e99443e1518b631f7bd846d59f19`. The archive verification reconstructs recorded outcomes and evidence checks. Native regeneration on a third party's independently prepared hardware is a separate test.
