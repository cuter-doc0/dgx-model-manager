#!/usr/bin/env bash
# Name: Default GGUF Server
# Description: Serve GGUF models from the models directory
# VRAM: 8

docker run -d --rm --gpus all \
  --name llamacpp-server \
  -v ./models:/models \
  -p 8080:8080 \
  ghcr.io/ggerganov/llama.cpp:server \
  --model /models/YOUR_MODEL.gguf \
  --host 0.0.0.0 \
  --port 8080