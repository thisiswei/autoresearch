# Autoresearch macOS Summary

## What We Did

- Ported the repo from CUDA-only Linux assumptions to run on this Mac using Apple `mps`.
- Added device fallback logic so the code can run on `cuda`, `mps`, or `cpu`.
- Replaced CUDA-only attention/kernels with a PyTorch fallback path for non-CUDA runs.
- Added a detached experiment loop plus monitoring/keepalive so short hyperparameter search runs could continue automatically.
- Ran a sequence of short training experiments and recorded the outcomes in `results.tsv`.
- Pushed the experiment branch `autoresearch/mar10` to GitHub.
- Stopped the live run at the end of the session on request.

## What Model This Is

This is **not stock GPT-2**.

It is a **custom GPT-like autoregressive language model** defined in `train.py`:

- transformer blocks with causal self-attention
- RMSNorm-style normalization
- rotary positional embeddings
- configurable local/full attention windows via `WINDOW_PATTERN`
- additional value-embedding path and per-layer scalar residual controls
- custom optimizer mix (`MuonAdamW`)

So the right description is:

> a small custom GPT-style language model, not an exact GPT-2 reproduction

## Where It Was Training

Training ran **locally on this MacBook** using Apple Silicon `mps`.

Data was stored under:

- `~/.cache/autoresearch/data`
- `~/.cache/autoresearch/tokenizer`

Downloaded data shards present during these runs:

- `shard_00000.parquet`
- `shard_00001.parquet`
- `shard_06542.parquet`

Training used:

- train split: all downloaded shards except `shard_06542.parquet`
- validation split: pinned shard `shard_06542.parquet`

## What The Eval Was

Each experiment:

- initialized a fresh small model
- trained for a fixed budget of about **5 minutes**
- then evaluated on the pinned validation shard

The metric was:

- `val_bpb` = **validation bits per byte**

Interpretation:

- lower is better
- it measures how well the model predicts held-out text
- it is a tokenizer-robust language-model quality metric

## Best Results So Far

Completed runs recorded in `results.tsv`: **35**

Best `keep` progression:

- `32031cf` -> `3.198667` (baseline)
- `bfd49dd` -> `3.188481`
- `f1d8264` -> `3.187949`
- `5141445` -> `3.160693`

Current best result:

- commit: `5141445`
- `val_bpb`: `3.160693`
- change: `SCALAR_LR=0.1, TOTAL_BATCH_SIZE=8192`

Interpretation:

- among the tried settings, lowering `SCALAR_LR` to `0.1`
- and using `TOTAL_BATCH_SIZE=8192`
- gave the best validation result so far

## Main Conclusions

- The hyperparameter search was working: it found multiple improvements over baseline.
- The strongest improvement found in this session was commit `5141445`.
- Some regions looked consistently weak, especially several `HEAD_DIM=64` variants.
- A few runs crashed; those were recorded in `results.tsv` but not kept.

## What We Have vs. What We Do Not Have

What we have:

- a working macOS/MPS port
- a ledger of experiment outcomes in `results.tsv`
- the best-known config so far (`5141445`)
- pushed branch history on `autoresearch/mar10`

What we do **not** have:

- saved trained model checkpoints
- reusable model weights from the winning run
- an inference-ready final model artifact

Important limitation:

The current training code does **not** save model checkpoints. The only `torch.save(...)` in the repo is for tokenizer metadata, not model weights.

So the output of this session is:

> best training configuration found so far

not:

> saved final model weights

## Files To Look At

- `results.tsv`: experiment scoreboard
- `run.log`: live training output for the active/in-progress run
- `mac_loop.log`: experiment loop history
- `train.py`: model and training code
- `prepare.py`: data prep and eval logic

## Recommended Next Step

If the goal is a usable model artifact, the next step is:

1. add model checkpoint saving
2. rerun training from the best-known config (`5141445`)
3. optionally train longer or on stronger hardware
