# Release and Deployment

This page covers publishing a trained model to Hugging Face and deploying it on a single server with an OpenAI-compatible API.

## Publish to Hugging Face

```bash
export HF_TOKEN=hf_xxx
.venv/bin/python scripts/hf_prepare_and_publish_model.py \
  --repo-id aditaa/llm-from-scratch-v1 \
  --checkpoint artifacts/checkpoints/fineweb-global-bpe-v1-big-run1/last.pt \
  --push
```

## Deploy on another server

1. Bootstrap runtime:
```bash
bash scripts/bootstrap_inference.sh
```

2. Download full model snapshot:
```bash
export HF_TOKEN=hf_xxx
bash scripts/hf_download_model.sh aditaa/llm-from-scratch-v1 /srv/models/llm-v1
```

3. Start OpenAI-compatible API:
```bash
HOST=0.0.0.0 PORT=8000 MODEL_ID=llm-from-scratch-v1 \
bash scripts/run_openai_server.sh /srv/models/llm-v1
```

## Endpoints
- `GET /healthz`
- `GET /v1/models`
- `POST /v1/completions`
- `POST /v1/chat/completions`

## Quick test

```bash
curl http://127.0.0.1:8000/v1/models
```
