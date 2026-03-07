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
- Data inspection: corpus stats CLI in `src/llm/cli.py`
- Model config scaffold: `src/llm/model.py`

## Planned Extensions
- Attention mask and positional encodings
- Trainer module with optimizer, scheduler, and gradient clipping
- Evaluation module with perplexity and sample generation reports
