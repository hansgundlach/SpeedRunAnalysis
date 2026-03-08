# NanoGPT Speedrun Records CSV — Data Extraction Notes

## What This File Contains

`nanogpt_speedrun_records.csv` tracks all 74 world speed records for the NanoGPT speedrun (Track 1: GPT-2 Small, target ≤3.28 val loss on FineWeb using 8×H100s). Each row is a record with:

- **record_num** — sequential record index (1–74)
- **date** — date the record was set
- **record_time_min** — wall-clock training time in minutes (from the README table)
- **training_tokens_M** — total training tokens in millions, **only for log-backed records**
- **training_flops** — estimated training FLOPs, **only for log-backed records**
- **description** — brief description of the innovation
- **innovation_label** — categorical label (see below)
- **data_quality** — `exact` if derived from a log file, `est` if no log-backed token/FLOP value is provided

---

## Data Sources

### Record metadata (time, date, description)
All sourced directly from the README table. These values are exact.

### Training tokens and FLOPs
These required reading individual log files. Each log file contains the full Python training script followed by live training output. The relevant lines look like:

```
step:1393/1393 val_loss:3.2785 train_time:179527ms step_avg:129.81ms
```

From this, `step:X/X` gives the total number of training steps. Combined with the tokens-per-step (batch size × sequence length), total tokens can be computed.

**Records with exact log data** (marked `exact`): records 1, 2, 4, 5, 8, 9, 11, 12, 14, 18, 19, 20, 21, 29, 34, 40, 46, 49, 53, 62, 74. For these, step counts were read directly from the final line of the log.

**All other records** (marked `est`): `training_tokens_M` and `training_flops` are intentionally left blank in `nanogpt_speedrun_records.csv`.

This avoids circularity when comparing runtime against compute: no token/FLOP value in the main CSV is inferred from runtime.

---

## FLOPs Calculation

Training FLOPs are estimated as:

```
FLOPs = 6 × N_params × total_tokens
```

Where:
- **6** accounts for the forward pass (~2N) plus the backward pass (~4N)
- **N_params = 124,000,000** — a fixed estimate for the GPT-2 small-scale model throughout the entire speedrun

This is a standard approximation from the scaling laws literature (Chinchilla, etc.) and ignores attention FLOPs (which are small relative to matmul FLOPs for moderate sequence lengths) and embedding layers (which aren't counted in N here).

---

## Batch Configuration Eras (Context Only)

The tokens-per-step changed significantly over the course of the speedrun. Four distinct eras were identified by reading representative log files.

These era notes are retained as historical context and are **not** used to fill missing compute values in `nanogpt_speedrun_records.csv`.

### Era 1: Records 1–11 (May–Nov 2024)
- Sequence length: **1,024 tokens**
- Batch: **512 sequences** (8 GPUs × 64 sequences/GPU)
- **Tokens per step: 524,288**

### Era 2: Records 12–20 (Nov 2024–Jan 2025)
- Introduced FlexAttention with 64K context windows
- Sequence length: **65,536 tokens** (64 × 1024)
- Batch: **8 sequences** (1 per GPU)
- **Tokens per step: 524,288** (same total, just restructured)

### Era 3: Records 21–39 (Jan–Sep 2025)
- Record 21 introduced a reduced batch size
- Effective tokens per step: **393,216** (e.g., 48×1024 tokens per GPU × 8 GPUs)
- Records 22–28 are pure systems improvements with no batch size change

### Era 4: Records 40–74 (Oct 2025–Feb 2026)
- Record 40 (Backout) reverted to a smaller fixed batch: **262,144 tokens/step** (2048 × 16 × 8)
- Record 46 introduced a **3-stage batch size schedule**:
  - Stage 1 (first 1/3 of steps): 131,072 tokens/step
  - Stage 2 (middle 1/3): 262,144 tokens/step
  - Stage 3 (final 1/3): 393,216 tokens/step
  - Average: **262,144 tokens/step**
- Records 40–74 are generally in the 262,144 tokens/step regime (with schedule variation for some runs)

---

## Innovation Labels

Each record is assigned one of four categories based on the primary driver of improvement:

| Label | Meaning |
|---|---|
| **Architecture** | Changes to model structure: attention mechanisms, embeddings, MLP design, skip connections, normalization, positional encodings, etc. |
| **Optimizer** | Changes to the optimizer: Muon, NorMuon, Polar Express, Cautious Weight Decay, learning rate schedules, momentum, etc. |
| **Systems** | Software/hardware efficiency: PyTorch version upgrades, CUDA/Triton kernels, distributed communication, memory layout, Flash Attention integration, async data loading, FP8 matmul, etc. |
| **Hyperparameter** | Tuning of existing hyperparameters without structural changes: softcap values, batch size adjustments, LR values, etc. |

Records with multiple innovations were assigned the category of the primary contribution. This is inherently subjective for mixed records (e.g., record 20 introduces both architectural changes and systems improvements; it is labeled Architecture because the merged QKV and long-short attention pattern are the primary novelties).

---

## Known Inaccuracies and Limitations

### Fixed N_params assumption
The model architecture changed significantly across 74 records. Parameter count was not extracted from logs and is held constant at 124M throughout. In reality:
- Early records used a simpler architecture closer to GPT-2 small (~117–124M)
- Later records added value embeddings, bigram hash embeddings, and other components that likely push the parameter count up somewhat (rough estimate: 130–145M by the end)
- Using N=124M throughout means FLOPs are slightly understated for later records

### Batch size schedule averaging
For Era 4 records with a 3-stage batch size schedule, the 262,144 tokens/step figure is the exact arithmetic mean across equal-duration stages. In practice, the stage boundaries may not be exactly equal thirds of total steps, and the extension phase at the end of training uses the largest batch. The actual mean may be slightly higher than 262,144 for records near the end of the speedrun.

### Missing compute on non-logged records
For records without a usable log in this repository, `training_tokens_M` and `training_flops` are blank by design.

If you need a fully populated (inferred) compute series for separate exploratory work, see `nanogpt_s_inferred_records.csv` in this directory.

### Attention FLOPs
The FLOPs formula `6N × tokens` excludes attention computation. For the early records with 1,024-token sequences this is negligible, but record 12 onward used sequences up to 65,536 tokens. FlexAttention with sliding windows doesn't scale as O(n²) globally, but this component is still unaccounted for in the FLOPs estimates. For a 12-layer transformer attending over 64K tokens with a sliding window, attention FLOPs are still small relative to the linear layers, so the error is probably under 10%.

### Val-time FLOPs not included
The FLOPs count covers only training steps. Periodic validation evaluations are excluded. Given that validation runs every ~125 steps and involves a single forward pass over a fixed token sequence, the omission is small (roughly 1–2% of total compute).

---

## File Location

```
nanogpt_speedrun_records.csv     # the data
nanogpt_speedrun_records_README.md  # this file
```

Both are in the root of the `modded-nanogpt` repository.
