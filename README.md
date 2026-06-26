# DGX Spark Model Manager

A Docker-based multi-engine inference management platform for NVIDIA DGX Spark. Manage Ollama models, route through LiteLLM, control multiple inference engines, and maintain a unified model inventory — all from one web interface.

## Features

| Feature | Description |
|---------|-------------|
| **Ollama Management** | Pull, list, and delete models with live progress |
| **LiteLLM Routing** | One-click wildcard routing so every Ollama model is auto-exposed |
| **4 Inference Engines** | Start/stop SGLang, vLLM, llama.cpp, Ollama via Docker |
| **Unified Inventory** | View all models — Ollama, HuggingFace cache, local directories |
| **HuggingFace Browser** | Search HF Hub, discover quantized variants, one-click download |
| **Live Status Bar** | Real-time health indicators for all services |
| **System Monitoring** | GPU, RAM, disk usage with restart capabilities |
| **Docker Stack** | All services containerized with GPU passthrough |

## Architecture

```
Your Apps (Open WebUI, agents, scripts, any OpenAI-compatible client)
         |
         v
   LiteLLM :4601  ──────────────────┬────────────┬──────────────┐
         |                          |            |              |
         v                          v            v              v
  SGLang :4620                Ollama :4610   vLLM :4630    llama.cpp :4640
  (large models)              (small/medium,  (alternative   (GGUF models)
                               hot-swap)       engine)

  Model Manager :4600  <-- this app (sits alongside, never in request path)
```

## Ports

| Port | Service |
|------|---------|
| 4600 | Model Manager Web UI & API |
| 4601 | LiteLLM (unified API router) |
| 4610 | Ollama |
| 4620 | SGLang |
| 4630 | vLLM |
| 4640 | llama.cpp |
| 4650 | LocalAI (optional) |
| 4660 | ComfyUI (optional) |

## Prerequisites

- NVIDIA DGX Spark or system with NVIDIA GPU
- Docker Engine with Compose
- NVIDIA Container Toolkit

## Quick Start

```bash
cd dgx-model-manager

# Build and start the stack
docker-compose up -d --build

# Access the web UI
open http://localhost:4600
```

> **Note:** `.env` and `config/config.json` come pre-configured with defaults. No manual setup needed. If you're using dockhand's git sync, set the additional env file to `.env` and override any variables via dockhand's env declarations.

The `config/litellm_config.yaml` is the default LiteLLM routing config — it ships pre-configured with Ollama wildcard routing and doesn't need any changes for initial setup. You can manage routes via the web UI (LiteLLM tab) or edit the file directly for advanced configurations.

## Configuration

### Environment Variables

`.env` comes pre-configured with defaults. Override any variable via dockhand env declarations or edit `.env` directly:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODELS_PATH` | `./models` | Where to store downloaded models |
| `HF_CACHE` | `~/.cache/huggingface` | HuggingFace cache directory |
| `HF_TOKEN` | - | HuggingFace token for private/gated models |
| `MANAGER_PORT` | `4600` | Web UI port |
| `LITELLM_PORT` | `4601` | LiteLLM port |
| `OLLAMA_PORT` | `4610` | Ollama port |
| `SGLANG_PORT` | `4620` | SGLang port |
| `VLLM_PORT` | `4630` | vLLM port |
| `LLAMACPP_PORT` | `4640` | llama.cpp port |

### Application Config

`config/config.json` comes pre-configured with defaults. Edit directly for customizations:

**Important:** The service URLs in `config.json` use Docker internal DNS names (e.g., `http://dgx-ollama:11434`). These resolve automatically within the Docker network — **do not change them** unless you're running services outside Docker.

```json
{
  "app": { "port": 4600 },
  "services": {
    "ollama_base": "http://dgx-ollama:11434",
    "litellm_base": "http://dgx-litellm:4000",
    "sglang_base": "http://dgx-sglang:30000",
    "vllm_base": "http://dgx-vllm:8000",
    "llamacpp_base": "http://dgx-llamacpp:8080"
  },
  "engines": {
    "ollama": { "enabled": true, "auto_start": true },
    "sglang": { "enabled": true, "auto_start": false },
    "vllm": { "enabled": true, "auto_start": false },
    "llamacpp": { "enabled": true, "auto_start": false }
  }
}
```

## Usage

### Web Interface

The status bar shows real-time health of all services. Tabs provide access to:

- **Overview** — System stats and engine status cards
- **Ollama** — Pull, list, delete Ollama models
- **LiteLLM** — View routes, apply wildcard routing
- **Engines** — Start/stop/restart inference engines
- **Inventory** — Unified view of all local models
- **HuggingFace** — Search and download from HF Hub
- **Logs** — View engine logs
- **Settings** — Configure service URLs

### API Endpoints

```bash
# Get system status
curl http://localhost:4600/api/status

# List Ollama models
curl http://localhost:4600/api/ollama/models

# Pull an Ollama model
curl -X POST http://localhost:4600/api/ollama/pull \
  -H "Content-Type: application/json" \
  -d '{"model": "llama3.2", "tag": "latest"}'

# List all engines
curl http://localhost:4600/api/engines

# Start an engine
curl -X POST http://localhost:4600/api/engines/vllm/control \
  -H "Content-Type: application/json" \
  -d '{"action": "start", "model": "meta-llama/Llama-3.1-8B"}'

# Get unified inventory
curl http://localhost:4600/api/inventory

# Search HuggingFace
curl "http://localhost:4600/api/hf/search?query=Qwen3.6-35B&limit=10"

# Apply Ollama wildcard to LiteLLM
curl -X POST "http://localhost:4600/api/litellm/wildcard?engine=ollama"
```

## LiteLLM Routing

After applying the wildcard, every Ollama model is automatically available through LiteLLM at port 4601:

```bash
# Query any Ollama model through LiteLLM
curl http://localhost:4601/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ollama/llama3.2",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## Engine Comparison

| Engine | Best For | Port | Docker Image |
|--------|----------|------|--------------|
| **Ollama** | Small/medium models, quick testing | 4610 | `ollama/ollama:latest` |
| **SGLang** | Large models, prefix caching | 4620 | `lmsysorg/sglang:latest` |
| **vLLM** | Broad model support, PagedAttention | 4630 | `vllm/vllm-openai:latest` |
| **llama.cpp** | GGUF quantized models | 4640 | `ghcr.io/ggerganov/llama.cpp:server` |

## Project Structure

```
dgx-model-manager/
├── docker-compose.yml          # Docker stack definition
├── Dockerfile                  # Model Manager image
├── requirements.txt            # Python dependencies
├── config/
│   ├── config.json             # App configuration
│   └── litellm_config.yaml     # LiteLLM routing config
├── engines/                    # Engine profile scripts
│   ├── sglang/
│   ├── vllm/
│   ├── llamacpp/
│   └── ollama/
├── models/                     # Local model storage
└── app/
    ├── main.py                 # FastAPI application
    ├── models.py               # Pydantic schemas
    ├── config.py               # Configuration management
    └── services/
        ├── docker_manager.py   # Docker container management
        ├── engine_manager.py   # Inference engine control
        ├── ollama_service.py   # Ollama integration
        ├── litellm_service.py  # LiteLLM integration
        ├── inventory.py        # Unified model inventory
        ├── hf_service.py       # HuggingFace search/download
        └── system_monitor.py   # System status monitoring
```

## Stopping Services

```bash
# Stop everything
docker-compose down

# Stop and remove volumes
docker-compose down -v

# Stop specific engine (via API)
curl -X POST http://localhost:4600/api/engines/vllm/control \
  -H "Content-Type: application/json" \
  -d '{"action": "stop"}'
```

## Troubleshooting

### GPU not detected
```bash
# Verify nvidia-smi works
nvidia-smi

# Test Docker GPU access
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

### Engine won't start
Check engine logs through the web UI (Logs tab) or API:
```bash
curl http://localhost:4600/api/engines/vllm/logs
```

### Port conflicts
Edit `docker-compose.yml` to change host ports if 4600-4660 are in use.

## License

MIT