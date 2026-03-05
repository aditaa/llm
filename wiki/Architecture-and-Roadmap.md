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
- `src/llm/tokenizer.py`: tokenizer training/save/load
- `src/llm/data.py`: token window dataset scaffolding
- `src/llm/sharding.py`: shard writing + manifest generation
- `src/llm/cli.py`: operational commands

## Roadmap Phases
### Phase 1 (Done)
- Repo scaffold, lint/type/test flow
- Tokenizer baseline + tests
- Data prep commands + sharding + manifests

### Phase 2 (In Progress)
- Model forward pass and loss computation
- Training loop with checkpoint save/resume
- Basic validation metrics

### Phase 3
- Sampling strategies (greedy, top-k, top-p)
- Experiment configs and reproducibility controls
- Runtime performance profiling

### Phase 4
- CI automation
- Expanded eval benchmarks
- Fine-tuning/instruction flows
