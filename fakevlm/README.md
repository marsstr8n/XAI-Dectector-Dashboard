# 🧠 FakeVLM Deployment Guide

This document provides step-by-step instructions for deploying **FakeVLM**.

---

## 📥 1. Download FakeVLM Model

First, install **Git LFS** (Large File Storage) and clone the FakeVLM model repository from Hugging Face.

```bash
cd YOUR_WORK_PATH
git lfs install
git clone https://huggingface.co/lingcco/fakeVLM
```

This will download all model files into the `fakeVLM/` directory.

---

## 🐳 2. Run FakeVLM with Docker

You can deploy FakeVLM using the `vllm/vllm-openai` container image.

```bash
docker run --runtime nvidia --gpus all --name fakeVLM --rm -d \
    -v YOUR_WORK_PATH:/models \
    -p 8000:8000 \
    vllm/vllm-openai:v0.10.1.1 \
    --model /models/fakeVLM \
    --served-model-name fakevlm \
    --max-model-len 4096 \
    --chat-template-content-format openai
```

**Explanation of key options:**
- `--runtime nvidia` and `--gpus all`: Enables GPU acceleration.
- `--served-model-name fakevlm`: Registers the model name as `fakevlm` for API calls.
- `-p 8000:8000`: Exposes the API on port 8000.

---

## 🚀 3. Access the API

Once the container is running, FakeVLM will serve an **OpenAI-compatible API** endpoint at:

👉 **http://localhost:8000/v1/chat/completions**

You can interact with it just like OpenAI’s API, for example:

```bash
curl http://localhost:8000/v1/chat/completions   -H "Content-Type: application/json"   -d '{
    "model": "fakevlm",
    "messages": [{"role": "user", "content": "Analyze this image for manipulation"}]
  }'
```

---

## 🧩 4. Verify Deployment

Check that the container is active and serving requests:

```bash
docker ps
```

Logs can be viewed using:

```bash
docker logs -f fakeVLM
```

To stop the container:

```bash
docker stop fakeVLM
```

