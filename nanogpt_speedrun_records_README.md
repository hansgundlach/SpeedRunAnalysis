# NanoGPT Speedrun Records CSV — Data Extraction Notes

## What This File Contains

`nanogpt_speedrun_records.csv` tracks all 74 world speed records for the NanoGPT speedrun (Track 1: GPT-2 Small, target ≤3.28 val loss on FineWeb using 8×H100s). Each row is a record with:

- **record_num** — sequential record index (1–74)
- **date** — date the record was set
- **record_time_min** — wall-clock training time in minutes (from the README table)
- **training_tokens_M** — total training tokens in millions, filled wherever the repository contains enough log information to compute or reasonably resolve it
- **total_params_M_est** — estimated total parameter count in millions, including embedding tables and other stored parameters
- **flop_params_M_est** — estimated effective parameter count in millions used in the FLOPs approximation
- **training_flops** — estimated training FLOPs derived from `training_tokens_M` and `flop_params_M_est`
- **description** — brief description of the innovation
- **innovation_label** — categorical label (see below)
- **data_quality** — `exact` if taken directly from a single resolved log, `est` if a small assumption was needed (for example selecting the winning run from a directory of logs or using a documented batch schedule)

---

## Data Sources

### Record metadata (time, date, description)
All sourced directly from the README table. These values are exact.

### Training tokens and FLOPs
These were extracted by reading the log links from the short-track table in `modded-nanogpt/README.md`. Each log contains the Python training script followed by live training output. The relevant lines look like:

```
step:1393/1393 val_loss:3.2785 train_time:179527ms step_avg:129.81ms
```

From this, `step:X/X` gives the total number of training steps. Combined with the tokens-per-step implied by the logged batch configuration, total tokens can be computed.

Most rows are now populated. Only records **3** and **66** remain blank because the short-track README does not link a usable training log for those records (`none` for record 3, `-` for record 66).

Rows marked `exact` were resolved from a single log file with an explicit final step count and explicit batch configuration. Rows marked `est` still use repository-backed evidence, but require at least one small judgment call documented below.

### Parameter estimates
The repository does not print a single canonical parameter-count number for every record, so parameter counts were estimated from the logged model definitions.

Two different parameter estimates are now tracked:

- **`total_params_M_est`**: total stored parameters, including embedding tables, value embeddings, bigram embeddings, gate banks, and other learned tensors
- **`flop_params_M_est`**: the effective dense-parameter count used in the `6N × tokens` FLOPs approximation

This distinction matters because several late records add very large embedding tables. Those increase model size substantially, but embedding lookups are not well-modeled by the standard `6N × tokens` dense-matmul approximation. So the FLOPs column uses `flop_params_M_est`, not `total_params_M_est`.

---

## FLOPs Calculation

Training FLOPs are estimated as:

```
FLOPs = 6 × flop_params × total_tokens
```

Where:
- **6** accounts for the forward pass (~2N) plus the backward pass (~4N)
- **flop_params** is the per-record value in `flop_params_M_est`

This is still the standard scaling-laws approximation and still ignores explicit attention FLOPs. It also intentionally does **not** charge raw embedding-table size one-for-one as dense FLOPs.

---

## Batch Configuration Eras (Context Only)

The tokens-per-step changed significantly over the course of the speedrun. Four distinct eras were identified by reading representative log files.

These era notes are retained as historical context. In a few `est` rows they also serve as a cross-check that the resolved tokens-per-step are consistent with neighboring records.

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

### Assumptions used for `est` rows
- **Directory log links**: some README entries point to a directory rather than a single file (for example records 14-17). In those cases, the filled value comes from the successful run in that directory whose final logged runtime best matches the README record time while also hitting the target loss.
- **Batch schedule averaging**: for late-era runs with `train_bs_schedule`, the token count uses the documented schedule as implemented in the logged script. This is close to, but not always identical to, using a single fixed average tokens-per-step value.
- **Era/regime carry-forward**: for some systems-only records, the logged code indicates the same batch regime as adjacent records. Those rows are marked `est` even when the resulting token count is very likely exact, because the fill relies on that continuity check rather than a fresh hand-audit of every per-step detail.
- **Architecture-family parameter mapping**: parameter counts were estimated from representative logged model definitions and then carried across records until the next architecture-changing record. This affects both `total_params_M_est` and `flop_params_M_est`.
- **Missing logs**: records 3 and 66 remain blank because the repository does not currently provide a usable linked log for them.

### Architecture-family parameter map
The following family-level estimates were used:

| Records | Total params (M) | FLOP params (M) | Notes |
|---|---:|---:|---|
| 1-7 | 123.6 | 123.6 | GPT-2-small-scale tied-head regime |
| 8-13 | 162.2 | 123.6 | Untied embed/lm_head increases stored params, not dense FLOPs |
| 14-16 | 394.0 | 123.6 | Large value-embedding tables |
| 17-29 | 275.7 | 121.2 | One attention layer removed; 3 value-embedding tables |
| 30-34 | 271.0 | 116.5 | First MLP layer removed |
| 35-48 | 268.7 | 114.1 | First attention layer also removed |
| 49-61 | 268.7 | 114.1 | 11-layer regime with similar dense footprint |
| 62 | 466.6 | 114.1 | Bigram hash embedding added |
| 63-64 | 543.8 | 114.1 | Untied to 5 value-embedding tables |
| 65-72 | 543.8 | 114.1 | Value embeddings fused into one parameter, same total size |
| 73-74 | 543.8 | 114.1 | Partitioned Hyperconnections add only negligible extra scalars |

### Batch size schedule approximation
For Era 4 records with a 3-stage batch size schedule, the average tokens-per-step is close to 262,144, but the exact mean depends on the precise schedule boundaries and extension phase. The filled values are therefore more accurate than a pure era-average, but still inherit small ambiguity from how the schedule is summarized at the dataset level.

### Total params versus FLOP params
The biggest source of parameter-count variation is embedding-heavy changes: untied embeddings, value embeddings, and bigram embeddings. These can change total model size by hundreds of millions of parameters while changing dense training FLOPs much less. The dataset therefore reports both quantities separately.

### Missing compute on non-logged records
For records without a usable linked log in this repository, `training_tokens_M` and `training_flops` are blank by design. At the moment this applies to records 3 and 66.

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
