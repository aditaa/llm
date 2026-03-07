# Architecture Overview

## Pipeline
1. Data ingestion and cleanup
2. Tokenizer training and vocabulary export
3. Sequence packing and dataloading
4. Decoder-only transformer model
5. Training loop with checkpointing
6. Evaluation and text generation

## Current Scope
- Tokenizer baseline: byte-level BPE tokenizer in `src/llm/tokenizer.py`
- Tokenizer contract hashing (`tokenizer_hash`, `tokenizer_contract_hash`) enforced in sharding/training/integrity
- Data inspection: corpus stats CLI in `src/llm/cli.py`
- Model baseline in `src/llm/model.py`: RoPE + RMSNorm + SwiGLU (`gpt_rope_rmsnorm_swiglu_v1`)
- Legacy checkpoint compatibility path: `gpt_learnedpos_layernorm_gelu_v0`

## Planned Extensions
- Expanded eval benchmark suite and checkpoint regression tracking
- Larger-context training profiles + throughput benchmarks
- Fine-tuning/instruction-data training flows
