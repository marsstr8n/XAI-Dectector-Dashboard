# 🛡️ FakeShield Installation

This folder provides a Docker-based environment for running the **FakeShield** for the Explainable AI (XAI) system. It includes setup instructions for cloning, building, downloading model weights, and launching the FastAPI service.

---

## 📦 1. Clone and Navigate to FakeShield

```bash
git clone https://github.com/marsstr8n/XAI-Dectector-Dashboard.git
cd XAI-Dectector-Dashboard/fakeshield
```

---

## 🧰 2. Build Container Image

Build the container image with CUDA 11.7, Python 3.9, and PyTorch 1.13 support:

```bash
docker build -t cuda117-py39-torch113 .
```

---

## 🎯 3. Download Model Weights

### (1) Download FakeShield Weights from Hugging Face

The FakeShield model consists of three main components:  
- `DTE-FDM` (Deepfake Textual Explanation — Feature Detection Module)  
- `MFLM` (Multi-Fusion Learning Module)  
- `DTG` (Deepfake Text Generator)

All three parts are packaged together and available at the [Hugging Face repository](https://huggingface.co/zhipeixu/fakeshield-v1-22b/tree/main).

To download all weights efficiently:

```bash
pip install huggingface_hub
huggingface-cli download --resume-download zhipeixu/fakeshield-v1-22b --local-dir weight/
```

---

### (2) Download Pretrained SAM Weight

The MFLM module relies on pretrained SAM (Segment Anything Model) weights.  
Download using `wget`:

```bash
wget https://huggingface.co/ybelkada/segment-anything/resolve/main/checkpoints/sam_vit_h_4b8939.pth -P weight/
```

---

### (3) Organize the Weights

Ensure your `weight/` folder structure looks like this:

```
fakeshield/
 ├── weight/
 │   ├── fakeshield-v1-22b/
 │   │   ├── DTE-FDM/
 │   │   ├── MFLM/
 │   │   ├── DTG.pth
 │   ├── sam_vit_h_4b8939.pth
```

---

## 🚀 4. Run the Container

Launch the FakeShield FastAPI service using the container:

```bash
docker run --gpus all --rm -it --name fakeshield \ 
    -v /path/to/fakeshield/weight:/workspace/FakeShield/weight \ 
    -v /path/to/fakeshield/analyze.sh:/workspace/FakeShield/scripts/analyze.sh \
    -v /path/to/fakeshield/fastapi_app.py:/workspace/FakeShield/fastapi_app.py \
    -w /workspace/FakeShield \
    -p 8888:8888 \
    cuda117-py39-torch113:latest \
    bash -lc 'uvicorn fastapi_app:app --host 0.0.0.0 --port 8888'
```

Once the container starts, the FakeShield API will be available at:  
👉 **http://localhost:8888**

---

## 🧪 5. Example Usage

### (1) Using `curl`

Send an image to the FakeShield API for analysis:

```bash
curl -s -X POST "http://localhost:8888/analyze" -F "file=@/path/to/image/input" | jq
```

The response will include a JSON object containing detection results and the mask.

## 📚 References

- [FakeShield on GitHub](https://github.com/zhipeixu/FakeShield)  

