# Architecture and Roadmap

## Architecture
Pipeline progression:
1. Data ingestion and cleaning
2. Tokenizer training
3. Sequence packing/sharding
4. Decoder-only transformer model
5. Training loop with checkpointing
6. Evaluation + generation

Current implemented modules:
- `src/llm/zim.py`: ZIM extraction helpers
- `src/llm/tokenizer.py`: BPE tokenizer training/save/load + contract hashing
- `src/llm/data.py`: token window dataset scaffolding
- `src/llm/sharding.py`: shard writing + manifest generation with tokenizer hash enforcement
- `src/llm/model.py`: modern GPT block stack (RoPE + RMSNorm + SwiGLU)
- `src/llm/cli.py`: operational commands

## Roadmap Phases
### Phase 1 (Done)
- Repo scaffold, lint/type/test flow
- Tokenizer baseline + tests
- Data prep commands + sharding + manifests

### Phase 2 (Done)
- Model forward pass and loss computation
- Training loop with checkpoint save/resume
- Basic validation metrics + generation path

### Phase 3
- Sampling strategies (greedy, top-k, top-p)
- Experiment configs and reproducibility controls
- Runtime performance profiling (`scripts/benchmark_rtx5070_context_profiles.sh`)

### Phase 4
- CI automation
- Expanded eval benchmarks
- Fine-tuning/instruction flows
