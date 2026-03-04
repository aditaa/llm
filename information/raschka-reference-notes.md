# Raschka Reference Notes (Heavy Cliff Notes)

Last updated: 2026-03-04

## 1) Source Index

### Primary sources used for this note
- Local PDF (book):  
  `information/Build a Large Language Model (From Scratch) - Sebastian Raschka.pdf`
- Blog/course post:  
  https://magazine.sebastianraschka.com/p/coding-llms-from-the-ground-up
- Official code repository:  
  https://github.com/rasbt/LLMs-from-scratch
- Local submodule mirror (for direct code reading):  
  `information/external/LLMs-from-scratch`

### Fast entry points
- Repo root README (chapter map):  
  https://github.com/rasbt/LLMs-from-scratch/blob/main/README.md
- Setup guide:  
  https://github.com/rasbt/LLMs-from-scratch/blob/main/setup/README.md
- Local setup guide mirror:  
  `information/external/LLMs-from-scratch/setup/README.md`

## 2) Big Picture (What this teaches)

Core progression across book + repo:
1. Text as tokens and training windows
2. Attention mechanics (self, causal, multi-head)
3. GPT model assembly (embedding -> blocks -> logits)
4. Pretraining loop (next-token objective)
5. Finetuning for classification
6. Instruction finetuning + response evaluation

Mental model:
- Build small, fully understandable components.
- Keep architecture close to production GPT patterns.
- Start with correctness and clarity, then add optimizations.

Design philosophy that is relevant for this repo:
- Prefer explicit implementations over abstraction-heavy frameworks.
- Keep “chapter snapshots” where each file is runnable and inspectable.
- Add optional accelerations only after baseline behavior is validated.

## 3) Blog/Course Cliff Notes

Post: **Coding LLMs from the Ground Up: A Complete Course**  
Link: https://magazine.sebastianraschka.com/p/coding-llms-from-the-ground-up

Useful details:
- It is a course-style index that follows the same sequencing as the book/repo.
- It includes module-level video segments for setup, data, attention, model, pretraining, finetuning.
- It directly points back to the book and GitHub repo as canonical references.

Module list (from page structure):
1. Set up your code environment
2. Working with text data
3. Coding attention mechanisms
4. Coding LLM architecture (title formatting on page can be inconsistent in scraped HTML)
5. Pretraining on unlabeled data
6. Finetuning for classification
7. Instruction finetuning

Practical usage:
- Use the blog for learning order and context.
- Use the repo files for implementation details and runnable code.
- Use the PDF/book for conceptual depth and chapter narrative.

## 4) Chapter-by-Chapter Cliff Notes + Links

### Chapter 2: Working with Text Data
Goal:
- Convert raw text into model-ready token sequences and training samples.

Key concepts:
- Tokenization choices impact sequence length and vocabulary behavior.
- Sliding window / chunking is the backbone of autoregressive training data.
- Input/target pairing is simply next-token shift.

Start here:
- Main notebook:  
  https://github.com/rasbt/LLMs-from-scratch/blob/main/ch02/01_main-chapter-code/ch02.ipynb
- Minimal dataloader notebook:  
  https://github.com/rasbt/LLMs-from-scratch/blob/main/ch02/01_main-chapter-code/dataloader.ipynb
- Local mirrors:  
  `information/external/LLMs-from-scratch/ch02/01_main-chapter-code/ch02.ipynb`  
  `information/external/LLMs-from-scratch/ch02/01_main-chapter-code/dataloader.ipynb`

High-value optional references:
- BPE comparison:  
  https://github.com/rasbt/LLMs-from-scratch/tree/main/ch02/02_bonus_bytepair-encoder
- BPE from scratch:  
  https://github.com/rasbt/LLMs-from-scratch/tree/main/ch02/05_bpe-from-scratch

What to port into this repo:
- `src/llm/data.py`: text -> token IDs -> fixed context windows -> batches
- deterministic split + batching utilities
- unit tests for shape, shift correctness, and edge windows

---

### Chapter 3: Coding Attention Mechanisms
Goal:
- Implement attention mechanics from basic form to causal multi-head attention.

Key concepts:
- Scaled dot-product attention
- Causal masking for autoregressive decoding
- Multi-head projection + concat + output projection
- Dropout as regularization for attention/MLP paths

Start here:
- Main notebook:  
  https://github.com/rasbt/LLMs-from-scratch/blob/main/ch03/01_main-chapter-code/ch03.ipynb
- Focused multi-head notebook:  
  https://github.com/rasbt/LLMs-from-scratch/blob/main/ch03/01_main-chapter-code/multihead-attention.ipynb
- Efficient MHA comparisons:  
  https://github.com/rasbt/LLMs-from-scratch/tree/main/ch03/02_bonus_efficient-multihead-attention

Local mirrors:
- `information/external/LLMs-from-scratch/ch03/01_main-chapter-code/ch03.ipynb`
- `information/external/LLMs-from-scratch/ch03/02_bonus_efficient-multihead-attention/mha-implementations.ipynb`

What to port into this repo:
- `src/llm/model.py`: `MultiHeadAttention` with strict causal mask behavior
- tests for:
  - mask correctness
  - output shape invariants
  - deterministic results under fixed seed

---

### Chapter 4: Implementing a GPT Model
Goal:
- Compose token embeddings, position embeddings, transformer blocks, and LM head.

Key concepts:
- LayerNorm + residual (“pre-norm” style in many GPT implementations)
- Feed-forward block after attention
- Repeated transformer block stack
- Parameter count awareness and memory implications

Start here:
- Main notebook:  
  https://github.com/rasbt/LLMs-from-scratch/blob/main/ch04/01_main-chapter-code/ch04.ipynb
- Standalone summary script:  
  https://github.com/rasbt/LLMs-from-scratch/blob/main/ch04/01_main-chapter-code/gpt.py
- KV cache bonus:  
  https://github.com/rasbt/LLMs-from-scratch/tree/main/ch04/03_kv-cache

Advanced architecture explorations:
- GQA: https://github.com/rasbt/LLMs-from-scratch/tree/main/ch04/04_gqa
- MLA: https://github.com/rasbt/LLMs-from-scratch/tree/main/ch04/05_mla
- SWA: https://github.com/rasbt/LLMs-from-scratch/tree/main/ch04/06_swa
- MoE: https://github.com/rasbt/LLMs-from-scratch/tree/main/ch04/07_moe

What to port into this repo:
- `ModelConfig` expansion in `src/llm/model.py`
- `TransformerBlock` + `GPTModel` baseline
- generation-ready forward pass returning logits `[B, T, V]`

---

### Chapter 5: Pretraining on Unlabeled Data
Goal:
- Train GPT with next-token objective, track losses, save/load checkpoints.

Key concepts:
- Cross-entropy on shifted targets
- train/validation loops
- optimizer + gradient flow stability
- checkpointing for resume and reproducibility

Start here:
- Main notebook:  
  https://github.com/rasbt/LLMs-from-scratch/blob/main/ch05/01_main-chapter-code/ch05.ipynb
- Standalone train script:  
  https://github.com/rasbt/LLMs-from-scratch/blob/main/ch05/01_main-chapter-code/gpt_train.py
- Standalone generation script:  
  https://github.com/rasbt/LLMs-from-scratch/blob/main/ch05/01_main-chapter-code/gpt_generate.py

High-value practical extras:
- LR scheduler ideas:  
  https://github.com/rasbt/LLMs-from-scratch/tree/main/ch05/04_learning_rate_schedulers
- Training speed tips:  
  https://github.com/rasbt/LLMs-from-scratch/tree/main/ch05/10_llm-training-speed
- Gutenberg pretraining example:  
  https://github.com/rasbt/LLMs-from-scratch/tree/main/ch05/03_bonus_pretraining_on_gutenberg

What to port into this repo:
- `src/llm/training.py` with:
  - training step
  - validation pass
  - checkpoint I/O
  - seed setup
- CLI entry points for train/eval/generate

---

### Chapter 6: Finetuning for Classification
Goal:
- Adapt pretrained GPT for supervised classification.

Key concepts:
- Task-head adaptation from language model backbone
- dataset preparation and label handling
- evaluation with task metrics (accuracy, etc.)

Start here:
- Main notebook:  
  https://github.com/rasbt/LLMs-from-scratch/blob/main/ch06/01_main-chapter-code/ch06.ipynb
- Standalone finetune script:  
  https://github.com/rasbt/LLMs-from-scratch/blob/main/ch06/01_main-chapter-code/gpt_class_finetune.py
- Bonus extra experiments:  
  https://github.com/rasbt/LLMs-from-scratch/tree/main/ch06/02_bonus_additional-experiments

What to port into this repo:
- `src/llm/finetune_classification.py`
- dataset adapters + evaluation report helpers
- smoke tests for tiny overfit batch behavior

---

### Chapter 7: Instruction Finetuning
Goal:
- Supervised instruction tuning and response quality evaluation.

Key concepts:
- Instruction dataset formatting
- supervised finetuning loop
- inference/evaluation workflow
- optional preference tuning pathways

Start here:
- Main notebook:  
  https://github.com/rasbt/LLMs-from-scratch/blob/main/ch07/01_main-chapter-code/ch07.ipynb
- Standalone instruction finetune script:  
  https://github.com/rasbt/LLMs-from-scratch/blob/main/ch07/01_main-chapter-code/gpt_instruction_finetuning.py
- Evaluation helper script:  
  https://github.com/rasbt/LLMs-from-scratch/blob/main/ch07/01_main-chapter-code/ollama_evaluate.py

Supporting utilities:
- Dataset utilities:  
  https://github.com/rasbt/LLMs-from-scratch/tree/main/ch07/02_dataset-utilities
- Model evaluation notebooks:  
  https://github.com/rasbt/LLMs-from-scratch/tree/main/ch07/03_model-evaluation
- Preference tuning (DPO):  
  https://github.com/rasbt/LLMs-from-scratch/tree/main/ch07/04_preference-tuning-with-dpo

What to port into this repo:
- `src/llm/finetune_instruction.py`
- `src/llm/eval.py` with structured output scoring hooks
- JSONL dataset schema checks and data validators

## 5) Setup and Environment Cliff Notes

Canonical setup reference:
- https://github.com/rasbt/LLMs-from-scratch/blob/main/setup/README.md

Practical notes:
- The upstream repo supports regular pip workflows and `uv`-style workflows.
- Main chapter material is designed to run on ordinary laptops.
- GPU improves runtime in later chapters (pretraining + finetuning), but not required to start.

For this repo:
- Keep our own tooling (`Makefile`, `pyproject.toml`) as the primary execution path.
- Use upstream setup docs as guidance when adopting dependencies from chapter scripts.

## 6) Implementation Order for This Repo (Concrete Plan)

Phase A (now):
1. Add `src/llm/data.py` for sequence packing and batch generation.
2. Expand `src/llm/model.py` with attention + transformer block + GPT model.
3. Add tests for data pipeline and attention masking.

Phase B:
1. Add `src/llm/training.py` with train/val loops.
2. Add `src/llm/generate.py` with greedy + top-k + top-p decoding.
3. Add checkpoint save/load + reproducibility controls.

Phase C:
1. Add classifier finetuning module and metrics.
2. Add instruction finetuning module and dataset schema checks.
3. Add evaluation hooks and report outputs.

Definition of done for each phase:
- command runnable from CLI
- at least one unit test and one smoke test
- docs updated in `README.md` + `AGENTS.md`

## 7) “Read This First” Map (Quick Links by Task)

If you are implementing tokenization/data:
- https://github.com/rasbt/LLMs-from-scratch/blob/main/ch02/01_main-chapter-code/ch02.ipynb
- https://github.com/rasbt/LLMs-from-scratch/blob/main/ch02/01_main-chapter-code/dataloader.ipynb

If you are implementing attention:
- https://github.com/rasbt/LLMs-from-scratch/blob/main/ch03/01_main-chapter-code/ch03.ipynb
- https://github.com/rasbt/LLMs-from-scratch/blob/main/ch03/01_main-chapter-code/multihead-attention.ipynb

If you are implementing GPT architecture:
- https://github.com/rasbt/LLMs-from-scratch/blob/main/ch04/01_main-chapter-code/gpt.py
- https://github.com/rasbt/LLMs-from-scratch/blob/main/ch04/03_kv-cache/gpt_with_kv_cache.py

If you are implementing training:
- https://github.com/rasbt/LLMs-from-scratch/blob/main/ch05/01_main-chapter-code/gpt_train.py
- https://github.com/rasbt/LLMs-from-scratch/blob/main/ch05/01_main-chapter-code/gpt_generate.py

If you are implementing finetuning/eval:
- https://github.com/rasbt/LLMs-from-scratch/blob/main/ch06/01_main-chapter-code/gpt_class_finetune.py
- https://github.com/rasbt/LLMs-from-scratch/blob/main/ch07/01_main-chapter-code/gpt_instruction_finetuning.py
- https://github.com/rasbt/LLMs-from-scratch/blob/main/ch07/01_main-chapter-code/ollama_evaluate.py

## 8) Guardrails for Using These References

When porting ideas from upstream into this repo:
- keep code style consistent with this repository’s tooling
- avoid blind copy-paste; rewrite with minimal dependencies when possible
- preserve attribution in comments/docs when adapting non-trivial logic
- add tests immediately for any imported algorithmic behavior

When adding new reference files to `information/`:
- explain why it matters
- map it to one or more modules in `src/llm/`
- list exact next implementation tasks it informs
