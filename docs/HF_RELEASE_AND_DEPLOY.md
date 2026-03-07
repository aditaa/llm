# Hugging Face Release And Single-Server Deploy

This workflow publishes a complete model bundle to Hugging Face and deploys it on a separate server as an OpenAI-compatible API.

## 1) Prepare and publish model bundle
From this repo:

```bash
# optional: set token once
export HF_TOKEN=hf_xxx

# bundle only (local)
.venv/bin/python scripts/hf_prepare_and_publish_model.py \
  --repo-id aditaa/llm-from-scratch-v1 \
  --checkpoint artifacts/checkpoints/fineweb-global-bpe-v1-big-run1/last.pt

# bundle + push to HF model repo
.venv/bin/python scripts/hf_prepare_and_publish_model.py \
  --repo-id aditaa/llm-from-scratch-v1 \
  --checkpoint artifacts/checkpoints/fineweb-global-bpe-v1-big-run1/last.pt \
  --push
```

Bundle output includes:
- `checkpoint.pt`
- `tokenizer.json`
- `release_manifest.json`
- `README.md` (model card)

## 2) On deployment server (restricted network)
Clone repo or copy scripts, then install runtime:

```bash
bash scripts/bootstrap_inference.sh
```

Download the full model snapshot locally:

```bash
export HF_TOKEN=hf_xxx
bash scripts/hf_download_model.sh aditaa/llm-from-scratch-v1 /srv/models/llm-v1
```

This creates a full local copy so inference can continue even if external access is later blocked.

## 3) Run GPT-style API (OpenAI-compatible)

```bash
HOST=0.0.0.0 PORT=8000 MODEL_ID=llm-from-scratch-v1 \
bash scripts/run_openai_server.sh /srv/models/llm-v1
```

Endpoints:
- `GET /healthz`
- `GET /v1/models`
- `POST /v1/completions`
- `POST /v1/chat/completions`

## 4) Quick API tests

```bash
curl http://127.0.0.1:8000/v1/models
```

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model":"llm-from-scratch-v1",
    "messages":[{"role":"user","content":"Hello, who are you?"}],
    "max_tokens":120,
    "temperature":0.8,
    "top_k":40
  }'
```

## Notes
- The server is intentionally minimal and non-streaming.
- API auth/rate-limits/TLS should be added at a reverse proxy layer for production.
- If model files update, rerun `hf_download_model.sh` to refresh local snapshot.
