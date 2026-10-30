# Parameter Golf: the naive baseline

This is the starting point for OpenAI's Parameter Golf challenge (github.com/openai/parameter-golf), main
track `track_10min_16mb`. The goal is the lowest bits per byte (`val_bpb`) on the FineWeb validation set,
for a language model that:

- trains in **at most 10 minutes on 8×H100 SXM**, and
- fits in a **16,000,000-byte artifact** (decimal 16 MB, not 16 MiB): the bytes of `train_gpt.py` plus the
  bytes of the compressed model file it writes.

`train_gpt.py` is the challenge's naive baseline: a 9-layer, 512-wide GPT (8 heads, 4 KV heads, 2× ReLU²
MLP, tied embeddings, 1024-token SentencePiece vocabulary, about 17M parameters). It trains with Muon and Adam
on 524,288 tokens per step until the 600-second wallclock cap, quantizes the weights to int8 per row,
compresses them with zlib, then loads the compressed file back and evaluates it. Its published result is
`val_bpb` 1.2244 after the round trip (1.2172 before quantization), with a 15,863,489-byte artifact
(47,642 bytes of code and 15,815,847 bytes of model).

## Running it

```bash
pip install -r requirements.txt
python3 data/cached_challenge_fineweb.py --variant sp1024   # val split + 80 train shards + tokenizer
RUN_ID=baseline \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

With `--nproc_per_node=1` it accumulates gradients over 8 micro-steps, so one GPU runs the same
optimization, about 8× slower. It needs a CUDA GPU.

## The rules

From the challenge's README and FAQ:

1. **Artifact size.** Code bytes plus compressed model bytes must be at most 16,000,000. All counted code
   lives in `train_gpt.py`; the artifact must be self-contained and reproducible.
2. **Training time.** Training stops within 600 seconds of wall time on 8×H100 SXM. Compiling and the
   warmup steps that are undone before training starts don't count; every step that changes the final
   weights does.
3. **Evaluation time.** Evaluating the artifact takes at most another 10 minutes on 8×H100. Evaluation at
   any sequence length, and any evaluation method, is allowed within that limit.
4. **No network.** No external downloads or network calls during training or evaluation.
5. **No training data during evaluation**, unless those bits are paid for inside the 16 MB.
6. **No validation data during training.** You can't train on the validation set, or compress it into the
   artifact. Test-time training is allowed only on validation tokens you have already scored.
7. **The metric must be computed correctly.** `val_bpb` is the token cross-entropy converted to bits per
   byte of the original text over the **full** `fineweb_val_*` split (the fixed first 50,000 documents).
   Submissions that change the tokenizer or the dataset must prove the bytes are counted correctly.
8. **Libraries** may be imported as long as they don't sneak in extra compute, capabilities or code size.
9. **No cheating on the test loss**, and no brute-forcing seeds or other hidden extra compute.

## How this project scores a candidate

- **Data and tokenizer are fixed** to `fineweb10B_sp1024` and `fineweb_1024_bpe.model` (the challenge's
  published export, pinned to its Hugging Face revision). The evals pass them in as `DATA_PATH` and
  `TOKENIZER_PATH`; nothing under `data/` may change.
- **The environment is fixed** to `requirements.txt` (torch 2.9.1+cu128 on `python:3.12`). Only those
  packages and the standard library exist on the eval machines.
- Each run is `torchrun --standalone --nproc_per_node=8 train_gpt.py` with `DATA_PATH`, `TOKENIZER_PATH`,
  `RUN_ID`, `SEED` and `MAX_WALLCLOCK_SECONDS=600` set in the environment. `train_gpt.py` must read those
  and honor the wallclock cap.
- After training, `train_gpt.py` must:
  - write the compressed model to a file named `final_model*.ptz` in its working directory (the baseline
    writes `final_model.int8.ptz`)
  - print `Total submission size <label>: <N> bytes`
  - load that file back, evaluate it, and print `final_<label>_exact val_loss:<x> val_bpb:<y>`.

  The last `final_…_exact` line is the score. The eval measures the artifact itself: the bytes of
  `train_gpt.py` plus the size of the `final_model*.ptz` file.
- The benchmark runs three seeds (1337, 42, 2024) on one 8×H100 SXM machine. It reports the mean `val_bpb`
  and `meets_rules` (1 only if every run's artifact is at most 16,000,000 bytes, training stopped within
  600 s, and evaluation took at most 600 s). It also runs the parent's code once on the same machine, for
  a same-machine delta.
- Before that, `pgolf-rules` checks these rules statically, and `pgolf-smoke` runs a short training on one
  H100 to check that the script trains, writes and round-trips its artifact under 16 MB, and prints its
  score.

## Files

| File | |
|---|---|
| `train_gpt.py` | The baseline: model, training, int8+zlib export, round-trip evaluation |
| `data/cached_challenge_fineweb.py` | Downloads the published FineWeb export and tokenizer from Hugging Face |
| `data/download_hf_docs_and_tokenize.py`, `data/tokenizer_specs.json` | Rebuilds tokenizers and shards from the published documents |
| `requirements.txt` | The eval environment |

Adapted from openai/parameter-golf (MIT License); its training code adapts modded-nanogpt (see
THIRD_PARTY_NOTICES.md).
