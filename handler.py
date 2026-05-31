import os
import sys
import base64
import tempfile
import traceback

import runpod

sys.path.insert(0, "/app")

MODEL_DIR = os.environ.get("MODEL_DIR", "/models")

# HuggingFace repo IDs and internal task name for each supported task
_HF_REPOS = {
    "t2v": ("Wan-AI/Wan2.2-T2V-A14B", "t2v-A14B"),
    "ti2v": ("Wan-AI/Wan2.2-TI2V-5B", "ti2v-5B"),
}

# Cached pipeline instances (loaded once, reused across warm invocations)
_pipelines: dict = {}


def _checkpoint_dir(task: str) -> str:
    return os.path.join(MODEL_DIR, task)


def _download_model(task: str) -> str:
    from huggingface_hub import snapshot_download

    repo_id, _ = _HF_REPOS[task]
    ckpt_dir = _checkpoint_dir(task)

    if os.path.isdir(ckpt_dir) and os.listdir(ckpt_dir):
        print(f"[handler] Using cached model at {ckpt_dir}")
        return ckpt_dir

    print(f"[handler] Downloading {repo_id} → {ckpt_dir} (this may take a while)...")
    os.makedirs(ckpt_dir, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=ckpt_dir,
        ignore_patterns=["*.git*", "*.md", "*.txt"],
    )
    print(f"[handler] Download complete: {task}")
    return ckpt_dir


def _load_pipeline(task: str):
    if task in _pipelines:
        return _pipelines[task]

    if task not in _HF_REPOS:
        raise ValueError(f"Unknown task '{task}'. Valid options: {list(_HF_REPOS)}")

    _, task_name = _HF_REPOS[task]
    ckpt_dir = _download_model(task)

    from wan.configs import WAN_CONFIGS

    cfg = WAN_CONFIGS[task_name]

    if task == "t2v":
        from wan import WanT2V
        pipeline = WanT2V(config=cfg, checkpoint_dir=ckpt_dir, device_id=0, rank=0)
    elif task == "ti2v":
        from wan import WanTI2V
        pipeline = WanTI2V(config=cfg, checkpoint_dir=ckpt_dir, device_id=0, rank=0)

    _pipelines[task] = pipeline
    return pipeline


def _save_video(frames_tensor, output_path: str, fps: int = 16) -> None:
    """Save a video tensor (C, N, H, W) to an MP4 file."""
    import imageio
    import torch

    t = frames_tensor
    if t.dim() == 4 and t.shape[0] in (1, 3):  # (C, N, H, W) → (N, H, W, C)
        t = t.permute(1, 2, 3, 0)

    # Normalize to [0, 255] — handles both [-1,1] and [0,1] outputs
    if t.min() < 0:
        t = (t + 1.0) / 2.0
    frames_np = (t.clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()

    with imageio.get_writer(output_path, fps=fps, codec="libx264", quality=8) as writer:
        for frame in frames_np:
            writer.append_data(frame)


# ---------------------------------------------------------------------------
# Optional preload: set PRELOAD_TASKS=t2v,ti2v to warm the model at startup
# ---------------------------------------------------------------------------
_preload = os.environ.get("PRELOAD_TASKS", "")
for _task in [t.strip() for t in _preload.split(",") if t.strip()]:
    if _task in _HF_REPOS:
        print(f"[handler] Preloading pipeline: {_task}")
        _load_pipeline(_task)


# ---------------------------------------------------------------------------
# RunPod handler
# ---------------------------------------------------------------------------
def handler(job: dict) -> dict:
    job_input = job.get("input", {})

    task = job_input.get("task", "t2v")
    prompt = job_input.get("prompt", "")
    if not prompt:
        return {"error": "'prompt' is required"}

    # Resolution — default differs per task
    default_size = "832*480" if task == "ti2v" else "1280*720"
    size_str = job_input.get("size", default_size)
    try:
        w, h = map(int, size_str.split("*"))
    except ValueError:
        return {"error": f"Invalid 'size' format '{size_str}'. Use WxH e.g. '832*480'"}

    frame_num    = int(job_input.get("frame_num", 81))
    steps        = int(job_input.get("steps", 50))
    guide_scale  = float(job_input.get("guide_scale", 5.0))
    seed         = int(job_input.get("seed", -1))
    fps          = int(job_input.get("fps", 16))
    offload      = bool(job_input.get("offload_model", True))

    try:
        pipeline = _load_pipeline(task)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "output.mp4")

            if task == "t2v":
                frames = pipeline.generate(
                    input_prompt=prompt,
                    size=(w, h),
                    frame_num=frame_num,
                    sampling_steps=steps,
                    guide_scale=guide_scale,
                    seed=seed,
                    offload_model=offload,
                )

            elif task == "ti2v":
                image_b64 = job_input.get("image")
                if not image_b64:
                    return {"error": "'image' (base64 JPEG/PNG) is required for ti2v"}

                from PIL import Image
                import io

                img_bytes = base64.b64decode(image_b64)
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

                frames = pipeline.generate(
                    input_prompt=prompt,
                    img=img,
                    size=(w, h),
                    frame_num=frame_num,
                    sampling_steps=steps,
                    guide_scale=guide_scale,
                    seed=seed,
                    offload_model=offload,
                )

            _save_video(frames, output_path, fps=fps)

            with open(output_path, "rb") as f:
                video_b64 = base64.b64encode(f.read()).decode("utf-8")

        return {
            "video_base64": video_b64,
            "task": task,
            "size": size_str,
            "frame_num": frame_num,
            "seed": seed,
        }

    except Exception as exc:
        return {"error": str(exc), "traceback": traceback.format_exc()}


runpod.serverless.start({"handler": handler})
