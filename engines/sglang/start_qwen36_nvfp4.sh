#!/usr/bin/env bash
# Name: Qwen3.6-35B-A3B-NVFP4
# Description: Qwen 3.6 35B with NVFP4 quantization via SGLang
# VRAM: 24

docker run -d --rm --gpus all --ipc=host \
  --name sglang-qwen36 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -p 30000:30000 \
  lmsysorg/sglang:latest \
  python3 -m sglang.launch_server \
    --model-path /root/.cache/huggingface/hub/models--lyf--Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-NVFP4/snapshots/main \
    --host 0.0.0.0 \
    --port 30000 \
    --tp 1