#!/usr/bin/env bash
# Name: Qwen3.6-35B-A3B-NVFP4
# Description: Qwen 3.6 35B with NVFP4 quantization via vLLM
# VRAM: 24

docker run -d --rm --gpus all --ipc=host \
  --name vllm-qwen36 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -p 8000:8000 \
  vllm/vllm-openai:latest \
  --model /root/.cache/huggingface/hub/models--lyf--Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-NVFP4/snapshots/main \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.8