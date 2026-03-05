# Data Directory

This directory is intentionally ignored in Git except for this file.

Use it for local/intermediate corpus artifacts, for example:
- `data/extracted/*.txt` from ZIM extraction
- `data/tokenized/*.json` tokenizer or token ID outputs

Raw `.zim` files should stay on your server storage (for example `/data/iiab/zim/`) and should not be committed.
For warm/durable storage, prefer `/mnt/ceph/llm/data` (for example `/mnt/ceph/llm/data/shards/`).
When syncing to warm storage, keep raw archives under `/mnt/ceph/llm/data/raw_zim/`.
