# NanoGPT Medium Speedrun Records CSV — Data Extraction Notes

## What This File Contains

`nanogpt_medium_speedrun_records.csv` tracks all 18 world speed records for the NanoGPT speedrun Track 2 (GPT-2 Medium, target ≤2.92 val loss on FineWeb using 8×H100s). Each row is a record with:

- **record_num** — sequential record index (1–18)
- **date** — date the record was set
- **record_time_min** — wall-clock training time in minutes (from the README table)
- **total_params_M_est** — estimated total parameter count in millions, including embedding tables and other stored parameters
- **flop_params_M_est** — estimated effective parameter count in millions used in the FLOPs approximation
- **training_tokens_M** — total training tokens in millions (only when uncertainty is <=10%)
- **training_flops** — total training FLOPs (only when uncertainty is <=10%)
- **description** — brief description of the innovation
- **innovation_label** — categorical label (see below)
- **data_quality** — `exact` if derived from a log file with compute uncertainty <=10%, `est` otherwise

---

## Data Sources

### Record metadata (time, date, description)
All sourced directly from the README table. These values are exact.

### Training steps
Extracted from the final `step:X/X val_loss:` line of each log file. Log files are Python training scripts with the live training output appended. 17 of 18 records have log files; record 11 (PyTorch upgrade) has no log.

For the llm.c baseline (record 1), the log uses the older `s:X tel:X` format. The final logged entry is `s:60000 tel:2.9282`, but it does not include reliable batch-size metadata in the stored file.

### Batch configuration
Found by searching each log file for `batch_size`, `seq_len`, `train_bs_schedule`, etc. in the dataclass config section.

---

## FLOPs Calculation

```
training_flops = 6 × flop_params_M_est × 1e6 × training_tokens_M × 1e6
```

Where:
- **6** accounts for forward (~2N) + backward (~4N) passes
- **flop_params_M_est** is the per-record effective parameter count used for the dense-matmul approximation

### Critical caveat: the speedrun model is neither 350M nor 201M in total size

"Track 2" is called the "GPT-2 Medium" track because its **performance target** (≤2.92 val loss) matches what Andrej Karpathy's 350M-parameter llm.c baseline achieves. The actual speedrun model architecture is different throughout the track.

The early medium records do use a **16-layer, 1024-dim** backbone, but they also include substantial embedding-side parameters:
- untied `lm_head`
- multiple value-embedding tables
- later, a second input embedding
- later still, more value embeddings and additional small learned tensors

So the old single-number estimate of `201M` was only a rough count of the 16-layer hidden stack, not the full model size.

This file now tracks two parameter estimates:
- **`total_params_M_est`**: total stored parameters, including embeddings and other learned tensors
- **`flop_params_M_est`**: the effective parameter count used in the `6N × tokens` FLOPs approximation

This distinction matters because embedding-heavy changes can increase model size substantially without increasing dense training FLOPs by the same factor.

### Compute-quality gating used in this file

To avoid >10% compute uncertainty in downstream analysis:
- **Record 1** has `training_tokens_M` and `training_flops` blanked (batch-size ambiguity is large).
- **Record 11** has `training_tokens_M` and `training_flops` blanked (no log available).
- **Records 2–10 and 12–18** retain compute values from logs.

---

## Batch Configuration

### Records 1–17: Fixed batch, FlexAttention

All records from the llm.c baseline through record 17 used a fixed batch size of **524,288 tokens/step**:

- **llm.c baseline (record 1):** Standard llm.c batch = 2¹⁹ = 524,288 tokens/step; 60,000 steps → 31.46B tokens
- **Records 2–17:** FlexAttention with `batch_size = 8 × 64 × 1024 = 524,288` tokens/step (confirmed in log files). The FlexAttention implementation uses `B=1` (one combined variable-length sequence per device) with `train_seq_len = 64 × 1024 = 65,536` tokens per GPU. Total tokens per step = 8 GPUs × 65,536 = 524,288.

This differs from the small track, which went through multiple batch-config eras (512×1024, then FlexAttention, then reduced batches). The medium track started directly with FlexAttention and held the same batch size from record 2 all the way through record 17.

### Train vs. validation sequence length split

From record 3 onwards, the logs explicitly separate training and validation sequence lengths:

```python
train_seq_len = 64 * 1024   # 65,536 tokens per GPU per step
val_seq_len   = 4 * 64 * 1024  # 262,144 tokens for validation passes
```

Validation uses a 4× longer sequence, allowing the model to be evaluated at longer context than it was trained on. This is a meaningful difference from the small track, where train and val context lengths were the same. The val_seq_len is not counted in training tokens and does not affect FLOPs estimates.

### Sliding window warmup and attention FLOPs

The medium track uses a sliding window attention schedule that grows the window size from 0 to a maximum value over the course of training:

- **Records 2–7:** `window_size = round_to_128(1728 × step / total_steps)` — window grows linearly to a final size of 1,728 tokens
- **Record 8+:** `window_size = round_to_128(3456 × factor)` — window doubled to a final size of 3,456 tokens (record 8's key innovation)

This means in the **first ~10–20% of training steps the effective attention window is near zero**, reducing attention-related compute early in the run. For a local-window implementation at these dimensions, the run-averaged attention contribution is expected to be single-digit percent of `6N × tokens` for most records (roughly low-to-mid single digits for records 2–7 and high single digits for records 8+ after warmup averaging). On that basis, no additional per-record correction was applied.

### Record 18: Batch size schedule

Record 18 (Bulk Small Track Transfer) introduced a 12-bucket batch size schedule matching the one developed for the small track:

```python
train_bs_schedule = (131072, 262144, 393216, 524288,
                     524288, 524288, 524288, 524288,
                     524288, 524288, 524288, 524288)
```

Over **4,700 scheduled steps** + **40 extension steps**:
- Each of the 12 buckets covers 4700/12 ≈ 391.7 steps
- Buckets 1–3 ramp up; buckets 4–12 (9 buckets) stay at 524,288
- Average tokens/step over scheduled phase = (131072 + 262144 + 393216 + 9 × 524288) / 12 = **458,752**
- Extension: 40 steps × 524,288 tokens/step

**Total = 4,700 × 458,752 + 40 × 524,288 ≈ 2,177M tokens**

---

## Innovation Labels

Same taxonomy as the small track:

| Label | Meaning |
|---|---|
| **Architecture** | Model design: attention mechanism, embeddings, MLP, skip connections, positional encodings, etc. |
| **Optimizer** | Optimizer algorithm and hyperparameters: Muon, Snoo, EMA wrapper, LR schedules, Newton-Schulz coefficients, etc. |
| **Systems** | Software/hardware efficiency: PyTorch version upgrades, distributed communication, kernel optimizations, etc. |
| **Hyperparameter** | Tuning of existing hyperparameters without structural changes (none in this track) |

---

## Known Inaccuracies and Limitations

### Architecture-family parameter map
The medium-track records fall into a few clear model families:

| Records | Total params (M) | FLOP params (M) | Notes |
|---|---:|---:|---|
| 1 | 350.0 | 350.0 | Genuine llm.c GPT-2 Medium baseline |
| 2-8 | 454.5 | 248.6 | 16-layer medium backbone, 1 dropped attention layer, 3 value-embedding tables, untied head |
| 9 | 557.4 | 248.6 | Adds two more value-embedding tables |
| 10-17 | 608.9 | 248.6 | Adds a second input embedding; later records mostly preserve this parameter footprint |
| 18 | 613.4 | 252.8 | Bulk small-track transfer restores the missing attention layer and keeps the expanded embedding stack |

The main reason total parameter count is so much larger than the old `201M` estimate is that the medium speedrun architecture is embedding-heavy. The main reason `flop_params_M_est` stays much smaller than `total_params_M_est` is that raw embedding-table size is not charged one-for-one in the standard `6N × tokens` dense-matmul approximation.

### llm.c baseline token count (record 1)
The llm.c medium log is in the old `s:X tel:X` format and does not contain enough metadata to lock batch size with high confidence. Because that can create >10% compute uncertainty, record 1 compute fields are intentionally left blank.

### The old `201M` estimate was too low as a general model-size figure
The prior CSV treated most speedrun rows as `201M`, but the logs show that this omitted untied heads and large embedding-side additions. It was therefore not a good estimate of total model size, and it also understated dense FLOPs for the 16-layer medium backbone by excluding the output head.

### Record 7 step count uncertainty
The log file for record 7 has `num_iterations = 6450` in the config dataclass, but the final `step:X/X` line in the training output reads `6350/6350`. The training output may have been truncated before the final step, or the step count was overridden at launch via a command-line argument. The CSV uses 6,350 (from the log output). If the actual run completed all 6,450 steps, the token count would be ~1.5% higher (~3,382M instead of 3,329M).

### Record 11: No log file
Record 11 (PyTorch 2.7→2.10 upgrade) has no log in the repository and no date listed in the README. To avoid >10% risk from inferred compute, record 11 compute fields are intentionally left blank.

### Record 17: Higher step count than record 16
Record 17 ("Remove Redundant Block Mask Operation") has 5,960 steps — more than record 16's 5,535 — yet achieves essentially the same training time (22.98 vs. 22.99 minutes). This is consistent: removing the redundant op made each step slightly cheaper (~249ms → ~231ms), so more steps fit in the same wall-clock budget. The result is that record 17's total token count is actually slightly *higher* than records 15 and 16, despite being a systems improvement. This is worth noting as a counterintuitive data point.

### Record 18 batch schedule computation
The `get_bs(step)` function maps each step to one of 12 buckets via `int(12 * step / 4700)`. Details:
- The 12 buckets are not equal-width due to integer rounding (the last bucket absorbs the remainder)
- Buckets 1–3 use ramped-up batch sizes (131K, 262K, 393K tokens/step)
- Buckets 4–12 (nine buckets) all use the maximum 524,288 tokens/step
- The extension phase uses `train_bs_extension = 32 × 2048 × 8 = 524,288 tokens/step` for 40 steps

The computed total of ~2,177M tokens is exact given the code values.

### record 18 compresses an enormous leap in a single entry
Record 18 transferred a large batch of improvements from the small track simultaneously (NorMuon, Cautious Weight Decay, multi-token prediction, batch size schedule, Bigram Hash Embedding, and many more). Its 17.35-minute time represents a jump from 22.98 → 17.35 minutes — the largest single improvement in the medium track history. This is the only record whose token/compute efficiency reflects many compounding innovations at once, making it difficult to attribute the efficiency gains to any individual technique.

### Snoo optimizer (records 12–17)
Records 12–17 use the Snoo optimizer, which wraps Adam and Muon in an outer optimization loop. Snoo adds extra optimizer-step computation that is not captured in the `6N × tokens` FLOPs formula (which only covers the model forward and backward passes). The additional optimizer overhead is small relative to the model compute, but it is a source of unaccounted FLOPs unique to the medium track.

### `coordinate_descent_tuning` for records 6+
Records 6+ have `torch._inductor.config.coordinate_descent_tuning = True` enabled. This is explicitly allowed for the medium track by the competition rules (it is banned for the small track). It does not change FLOPs but adds ~25 minutes of untimed pre-run compilation overhead and makes wall-clock timings between medium records 1–5 and 6+ less directly comparable as a measure of pure training efficiency.

### Val-time FLOPs not included
Validation runs every 125 steps (confirmed in log files). Each validation pass processes `val_seq_len = 4 × 64 × 1024 × 8 = 2,097,152` tokens. Over the full run this is roughly:
- (steps / 125) validations × 2,097,152 tokens × (1/3 × N_params FLOPs/token for forward only)
- For record 2 (7,500 steps): 60 validations × ~140M FLOPs = ~8.4e9 FLOPs — negligible
- This is omitted from all estimates.

### Attention FLOPs grow during training (window warmup)
Because the sliding window size starts near zero and grows to ~1,728 or ~3,456 tokens, attention FLOPs are not constant per step. The `6N` formula intentionally remains a model-matmul baseline and does not add a separate attention term. Given the window warmup and observed configurations, this is treated as a sub-10% effect on populated medium-track rows.

---

## File Location

```
nanogpt_medium_speedrun_records.csv      # the data
nanogpt_medium_speedrun_records_README.md   # this file
```

See also the companion small-track files:
```
nanogpt_speedrun_records.csv
nanogpt_speedrun_records_README.md
```
