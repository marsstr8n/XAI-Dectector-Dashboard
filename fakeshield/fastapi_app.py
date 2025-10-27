import os
import json
import base64
import pathlib
import traceback
from typing import Any, Dict, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import subprocess
import mimetypes


app = FastAPI(title="FakeShield REST API")

# Resolve env with sensible defaults (override via container env if needed)
IMAGE_PATH = os.getenv("IMAGE_PATH", "/workspace/FakeShield/data/input.jpg")
DTE_FDM_OUTPUT = os.getenv("DTE_FDM_OUTPUT", "/workspace/FakeShield/out/dte_fdm.jsonl")
MFLM_OUTPUT = os.getenv("MFLM_OUTPUT", "/workspace/FakeShield/out/mflm_output/input.jpg")

# Your script paths
CONDA_ROOT = "/root/anaconda3"
CONDA_SH = f"{CONDA_ROOT}/etc/profile.d/conda.sh"
ENV_NAME = os.getenv("CONDA_ENV", "fakeshield")  # default to fakeshield
PIPELINE_SCRIPT = os.getenv("PIPELINE_SCRIPT", "/workspace/FakeShield/scripts/cli_demo.sh")


def _safe_delete(path: str):
    p = pathlib.Path(path)
    if p.exists():
        try:
            p.unlink()
            return {"path": path, "deleted": True}
        except Exception as e:
            return {"path": path, "deleted": False, "error": str(e)}
    else:
        return {"path": path, "deleted": False, "note": "file not found"}


def _read_output(path: str) -> Dict[str, Any]:
    p = pathlib.Path(path)
    if not p.exists():
        return {"path": path, "exists": False, "data": None, "note": "file not found"}

    # Try JSON first
    try:
        with p.open("r", encoding="utf-8") as f:
            return {"path": path, "exists": True, "data": json.load(f), "type": "json"}
    except Exception:
        pass

    # Try text next
    try:
        with p.open("r", encoding="utf-8", errors="ignore") as f:
            return {"path": path, "exists": True, "data": f.read(), "type": "text"}
    except Exception:
        pass

    # Fallback: binary -> base64
    with p.open("rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
        return {"path": path, "exists": True, "data_b64": b64, "type": "binary-base64"}


def _read_json(path: str) -> Dict[str, Any]:
    p = pathlib.Path(path)
    if not p.exists():
        return {"path": path, "exists": False, "data": None, "note": "file not found"}
    try:
        with p.open("r", encoding="utf-8") as f:
            return {"path": path, "exists": True, "data": json.load(f), "type": "json"}
    except Exception as e:
        return {"path": path, "exists": True, "error": f"not valid JSON: {e}"}


def _file_mime(path: str) -> Optional[str]:
    mt, _ = mimetypes.guess_type(path)
    return mt or "application/octet-stream"


def _read_image_b64(path: str) -> Dict[str, Any]:
    p = pathlib.Path(path)
    if not p.exists():
        return {"path": path, "exists": False, "data_b64": None, "note": "file not found"}
    with p.open("rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return {
        "path": path,
        "exists": True,
        "type": "image-base64",
        "mime": _file_mime(path),
        "data_b64": b64,
    }


def _run_pipeline(image_path: str) -> str:
    """
    Run cli_demo.sh inside the 'fakeshield' Conda environment.
    We export IMAGE_PATH/DTE_FDM_OUTPUT/MFLM_OUTPUT into the script's env.
    """
    # Ensure target dirs exist
    pathlib.Path(image_path).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(DTE_FDM_OUTPUT).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(MFLM_OUTPUT).parent.mkdir(parents=True, exist_ok=True)

    # Build env
    env = os.environ.copy()
    env["IMAGE_PATH"] = image_path
    env["DTE_FDM_OUTPUT"] = DTE_FDM_OUTPUT
    env["MFLM_OUTPUT"] = MFLM_OUTPUT

    # Use bash -lc so 'source conda.sh && conda activate' works
    cmd = (
        f"source {CONDA_SH} && conda activate {ENV_NAME} && "
        f"bash {PIPELINE_SCRIPT}"
    )

    # Capture stdout/stderr for debugging
    proc = subprocess.run(
        ["bash", "-lc", cmd],
        env=env,
        capture_output=True,
        text=True,
        timeout=int(os.getenv("PIPELINE_TIMEOUT_SEC", "1800")),  # 30 min default
    )
    logs = f"STDOUT:\n{proc.stdout}\n\nSTDERR:\n{proc.stderr}"
    if proc.returncode != 0:
        raise RuntimeError(f"Pipeline failed (exit={proc.returncode}).\n{logs}")
    return logs

@app.get("/health")
def health():
    return {"status": "ok", "image_path": IMAGE_PATH, "dte_fdm_output": DTE_FDM_OUTPUT, "mflm_output": MFLM_OUTPUT}

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    try:
        # Delete the previous results
        _safe_delete(IMAGE_PATH)
        _safe_delete(DTE_FDM_OUTPUT)
        _safe_delete(MFLM_OUTPUT)

        # Save uploaded image to IMAGE_PATH
        target_path = IMAGE_PATH
        with open(target_path, "wb") as out:
            content = await file.read()
            out.write(content)

        # Run your pipeline
        _ = _run_pipeline(target_path)

        # Read outputs
        dte = _read_json(DTE_FDM_OUTPUT)
        mflm = _read_image_b64(MFLM_OUTPUT)

        return JSONResponse(
            {
                "status": "success",
                "image_saved_to": target_path,
                "results": {
                    "dte_fdm": dte,
                    "mflm": mflm,
                },
            }
        )
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Pipeline timed out")
    except Exception as e:
        tb = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"{e}\n{tb}")