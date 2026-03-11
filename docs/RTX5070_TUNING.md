# RTX 5070 Training Tuning

Date: 2026-03-06  
GPU: NVIDIA GeForce RTX 5070 Ti Laptop GPU (12,227 MiB)

## Scope
- Dataset: `data/shards_global/fineweb-global-bpe-v1`
- Model architecture: `gpt_rope_rmsnorm_swiglu_v1` (default)
- Precision mode: `--precision auto` (bf16 on this GPU)
- Allocator: `PYTORCH_ALLOC_CONF=expandable_segments:True`

## Previous Small Model Result
- Shape: `n_layers=4`, `n_heads=4`, `d_model=256`, `context_length=256`
- Stable batch cap: `164`
- `166+` OOM
- Measured utilization: ~`45.4%` avg (peaks ~`91%`)
- Conclusion: memory-safe but compute-underutilized

## Big Model Sweep (12x12x768, ctx=512)

| Batch | Status | Peak VRAM (MiB) |
|---|---|---:|
| 20 | OK | 8177 |
| 24 | OK | 8999 |
| 28 | OK | 10779 |
| 30 | OK | 11441 |
| 32 | OK | 11481 |
| 34 | OK | 11403 |
| 35 | OK (short probe) |  |
| 36 | OOM | 11203 |
| 37 | OOM |  |

Probe artifact source: `artifacts/reports/rtx5070_probe_12x12x768_ctx512.tsv`

## Recommended Production Profile (RTX 5070)
- `n_layers=12`
- `n_heads=12`
- `d_model=768`
- `context_length=512`
- `batch_size=34` (production-safe)
- `learning_rate=1.5e-4`
- `eval_interval=2000`
- `eval_steps=5`
- `lr_schedule=cosine`
- `lr_warmup_steps=1000`
- `fail_on_eval_regression=true` (`eval_regression_tolerance=0.20`)
- `log_interval=100`
- `precision=auto`

Measured with this profile: `avg_util=99.1%`, `max_util=100%`, `min_util=96%`, `avg_mem=11474 MiB` over 30s.

## Launch
```bash
bash scripts/train_rtx5070_fineweb_350bt_bpe_v2.sh
```

350BT-first sweep and long-run launchers:
```bash
bash scripts/lr_sweep_rtx5070_fineweb_350bt_ctx512.sh
bash scripts/train_rtx5070_fineweb_350bt_bpe_v2.sh
```

Saved profile JSON:
- `configs/train/rtx5070/fineweb_350bt_bpe_v2_longrun.json`
- `configs/train/rtx5070/fineweb_350bt_bpe_v2_ctx1024_stage.json`

## Reproducible Context/Batch Benchmark
Run a short standardized sweep that records per-profile token throughput and GPU memory/utilization:

```bash
bash scripts/benchmark_rtx5070_context_profiles.sh --max-steps 1200 --compile-model
```

Default profile set:
- `512:34:1`
- `768:16:2`
- `1024:6:4`

Output:
- `artifacts/reports/rtx5070_ctx_bench_<timestamp>/summary.tsv`
