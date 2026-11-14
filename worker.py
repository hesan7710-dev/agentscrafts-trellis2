from __future__ import annotations
import base64, io, os, time, uuid
from pathlib import Path
from typing import Any

import runpod
import torch
from PIL import Image, ImageOps

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

PIPELINE = None
MAX_INPUT_BYTES = int(os.getenv("MAX_INPUT_BYTES", str(20 * 1024 * 1024)))
MAX_IMAGE_PIXELS = int(os.getenv("MAX_IMAGE_PIXELS", "40000000"))
DECIMATION_TARGET = int(os.getenv("DECIMATION_TARGET", "1000000"))
TEXTURE_SIZE = int(os.getenv("TEXTURE_SIZE", "2048"))
MODEL_ID = os.getenv("MODEL_ID", "microsoft/TRELLIS.2-4B")


def load_pipeline():
    global PIPELINE
    if PIPELINE is None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is required for TRELLIS.2 inference.")
        from trellis2.pipelines import Trellis2ImageTo3DPipeline
        PIPELINE = Trellis2ImageTo3DPipeline.from_pretrained(MODEL_ID)
        PIPELINE.cuda()
    return PIPELINE


def decode_image(value: str) -> Image.Image:
    if not isinstance(value, str) or not value:
        raise ValueError("input.image_base64 is required.")
    if value.startswith("data:"):
        value = value.split(",", 1)[1]
    raw = base64.b64decode(value, validate=True)
    if len(raw) > MAX_INPUT_BYTES:
        raise ValueError(f"Input image exceeds {MAX_INPUT_BYTES} bytes.")
    image = Image.open(io.BytesIO(raw))
    image.load()
    image = ImageOps.exif_transpose(image).convert("RGB")
    if image.width * image.height > MAX_IMAGE_PIXELS:
        raise ValueError("Image dimensions exceed the configured pixel limit.")
    return image


def store_result(raw: bytes, job_id: str) -> dict[str, Any]:
    if raw[:4] != b"glTF":
        raise RuntimeError("TRELLIS export did not produce a valid GLB header.")
    bucket = os.getenv("S3_BUCKET")
    if not bucket:
        return {"model_base64": base64.b64encode(raw).decode("ascii"), "content_type": "model/gltf-binary"}
    import boto3
    client = boto3.client(
        "s3", endpoint_url=os.getenv("S3_ENDPOINT_URL"),
        region_name=os.getenv("S3_REGION", "auto"),
        aws_access_key_id=os.getenv("S3_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("S3_SECRET_ACCESS_KEY"),
    )
    key = f"{os.getenv('S3_PREFIX', 'agentscrafts/trellis2')}/{job_id}.glb"
    client.put_object(Bucket=bucket, Key=key, Body=raw, ContentType="model/gltf-binary")
    expires = int(os.getenv("S3_PRESIGNED_URL_SECONDS", "3600"))
    url = client.generate_presigned_url("get_object", Params={"Bucket": bucket, "Key": key}, ExpiresIn=expires)
    return {"model_url": url, "storage_key": key, "content_type": "model/gltf-binary"}


def handler(job):
    data = job.get("input") or {}
    started = time.time()
    job_id = str(job.get("id") or uuid.uuid4().hex)
    try:
        image = decode_image(data.get("image_base64"))
        pipeline = load_pipeline()
        mesh = pipeline.run(image)[0]
        # The official example uses this simplification step before O-Voxel export.
        mesh.simplify(16777216)  # Match the upstream nvdiffrast face-count limit; final GLB decimation is handled below.
        import o_voxel
        glb = o_voxel.postprocess.to_glb(
            vertices=mesh.vertices,
            faces=mesh.faces,
            attr_volume=mesh.attrs,
            coords=mesh.coords,
            attr_layout=mesh.layout,
            voxel_size=mesh.voxel_size,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target=int(data.get("decimation_target", DECIMATION_TARGET)),
            texture_size=int(data.get("texture_size", TEXTURE_SIZE)),
            remesh=True,
            remesh_band=1,
            remesh_project=0,
            verbose=False,
        )
        output_path = Path("/tmp") / f"{job_id}.glb"
        glb.export(str(output_path), extension_webp=True)
        raw = output_path.read_bytes()
        stored = store_result(raw, job_id)
        return {"status": "completed", "job_id": job_id,
                "elapsed_seconds": round(time.time() - started, 2),
                "file_size_bytes": len(raw), **stored}
    except Exception as exc:
        return {"status": "failed", "job_id": job_id, "error": str(exc),
                "error_type": type(exc).__name__}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
