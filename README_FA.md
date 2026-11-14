# AgentsCrafts — TRELLIS.2 Serverless Worker

RunPod Queue Serverless worker for image-to-3D generation and GLB export using Microsoft's official TRELLIS.2 implementation.

## Status

Implementation draft based on the official upstream inference example. Docker build, CUDA extension compilation, model loading, and GPU inference have **not** yet been run in this environment. Do not deploy to production until the acceptance tests pass.

## Upstream requirements

The upstream project currently documents Linux-only testing, an NVIDIA GPU with at least 24 GB VRAM, CUDA Toolkit 12.4 (recommended), and its PyTorch 2.6.0/CUDA 12.4 environment. The Dockerfile follows that baseline and compiles the upstream CUDA extensions. Use a GPU with at least 24 GB VRAM; A100 40/80GB or H100 class is a reasonable initial test target, subject to availability and actual compatibility.

Upstream: https://github.com/microsoft/TRELLIS.2
Model: https://huggingface.co/microsoft/TRELLIS.2-4B

## Input

RunPod job input:

- `image_base64` (required; raw base64 or data URL)
- `decimation_target` (optional; default 1,000,000)
- `texture_size` (optional; default 2048)

## Output

The worker runs the official `Trellis2ImageTo3DPipeline`, simplifies the mesh, exports through `o_voxel.postprocess.to_glb`, and verifies the GLB magic header before returning it. When S3 settings are configured, it returns a presigned URL; otherwise it returns base64 for development only.

## Persistent storage

Configure `S3_BUCKET`, `S3_ENDPOINT_URL`, `S3_REGION`, `S3_ACCESS_KEY_ID`, and `S3_SECRET_ACCESS_KEY` as RunPod endpoint secrets. Optional `S3_PREFIX` and `S3_PRESIGNED_URL_SECONDS`. Do not commit credentials.

The Dockerfile sets `HF_HOME=/runpod-volume/huggingface`; mount a persistent RunPod Network Volume at `/runpod-volume` to preserve the Hugging Face cache between worker lifetimes. Confirm the mount path in endpoint settings. The local container disk is scratch only.

## Acceptance tests before deployment

1. Build the image successfully; ensure all upstream CUDA extensions compile.
2. Start on Linux with a supported NVIDIA GPU of at least 24 GB VRAM.
3. Load `microsoft/TRELLIS.2-4B` successfully (provide HF token as a secret if model access requires it).
4. Submit a known test image; verify output starts with GLB magic bytes `glTF` and opens in Blender/Three.js.
5. Verify S3 upload, content type, object key, presigned download, and expiry.
6. Verify oversized, invalid, and malformed images fail safely.
7. Record cold-start time, model-load time, inference time, peak VRAM, and actual RunPod billed duration.

## Important

The upstream README's generation times are measured on H100 and are not a promise for other GPUs. This worker exports textured GLB by default; it does not promise print-ready geometry or STL output.
