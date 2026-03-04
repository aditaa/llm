# Raschka Reference Notes

Last updated: 2026-03-04

## Sources Used
- Local PDF:
  `information/Build a Large Language Model (From Scratch) - Sebastian Raschka.pdf`
- Blog post:
  https://magazine.sebastianraschka.com/p/coding-llms-from-the-ground-up
- Code repository:
  https://github.com/rasbt/LLMs-from-scratch

## High-Value Takeaways
- The book/course flow is intentionally sequential:
  text prep -> tokenization -> attention -> GPT model -> pretraining -> finetuning -> instruction tuning.
- The educational strategy is "small but complete" models that run on normal hardware while preserving the same core mechanics used in larger LLM systems.
- PyTorch is used directly and avoids high-level LLM frameworks in the core learning path.
- The GitHub repo mirrors the chapter structure (`ch02` to `ch07`) and includes both main chapter code and optional bonus material.

## Practical Mapping to This Repository
- `src/llm/tokenizer.py`
  aligns with early text and tokenization chapters (Chapter 2 style scope).
- `src/llm/model.py`
  should next implement GPT components in this order:
  embeddings -> causal self-attention -> transformer block -> language-model head.
- Future `src/llm/training.py`
  should follow the Chapter 5 progression:
  loss tracking, train/val split, checkpoint save/load, and deterministic seeds.
- Future `src/llm/finetune.py`
  can split into:
  classifier finetuning path first, instruction finetuning path second.

## Immediate Build Plan (Derived from Sources)
1. Add sequence packing and batch generation for fixed context windows.
2. Implement causal multi-head attention and transformer blocks.
3. Add next-token pretraining loop with logging and checkpointing.
4. Add generation helpers (greedy, top-k, top-p) and eval metrics.
5. Add finetuning scripts for classification and instruction following.

## Notes About the Blog Post
- The blog post is a curated "complete course" index around the same pipeline, with chapter-aligned video modules and pointers back to the GitHub repo.
- It is useful as a practical companion for implementation order and environment setup details.

## Repository Usage Rule
When adding new source material to `information/`, include:
- why it matters,
- which module(s) it informs,
- and what concrete implementation step it should influence.
