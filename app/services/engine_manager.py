"""
Engine manager service - Manage inference engines via Docker Compose

All engines are defined in docker-compose.yml with individual profiles.
This service uses `docker compose run` to start engines with a model,
and `docker compose stop` to stop them.
"""

import os
import json
import logging
import httpx
import subprocess
from pathlib import Path
from typing import Optional, List, Dict
from datetime import datetime

from app.models import (
    EngineType, EngineStatus, EngineState, EngineProfile, EngineControl
)
from app.services.docker_manager import docker_manager
from app.config import get_config, EngineConfig

logger = logging.getLogger(__name__)

# Compose service name mapping
COMPOSE_SERVICES = {
    EngineType.OLLAMA: "ollama",
    EngineType.VLLM: "vllm",
    EngineType.SGLANG: "sglang",
    EngineType.LLAMACPP: "llamacpp",
    EngineType.LITELLM: "litellm",
}

# Container name mapping (for status checks)
CONTAINER_NAMES = {
    EngineType.OLLAMA: "dgx-ollama",
    EngineType.VLLM: "dgx-vllm",
    EngineType.SGLANG: "dgx-sglang",
    EngineType.LLAMACPP: "dgx-llamacpp",
    EngineType.LITELLM: "dgx-litellm",
}

# External port mapping
DEFAULT_PORTS = {
    EngineType.OLLAMA: 11434,
    EngineType.VLLM: 8000,
    EngineType.SGLANG: 30000,
    EngineType.LLAMACPP: 8080,
    EngineType.LITELLM: 4000,
}

# Host port env vars
PORT_ENV_VARS = {
    EngineType.OLLAMA: "OLLAMA_PORT",
    EngineType.VLLM: "VLLM_PORT",
    EngineType.SGLANG: "SGLANG_PORT",
    EngineType.LLAMACPP: "LLAMACPP_PORT",
    EngineType.LITELLM: "LITELLM_PORT",
}


class EngineManager:
    """Manage inference engines via Docker Compose"""

    def __init__(self):
        self.config = get_config()
        self._compose_file = self._find_compose_file()

    def _find_compose_file(self) -> Optional[str]:
        """Find docker-compose.yml relative to this project"""
        # Try /app/project (mounted in manager container)
        candidates = [
            "/app/project/docker-compose.yml",
            "/app/docker-compose.yml",
            str(Path(__file__).parent.parent.parent / "docker-compose.yml"),
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        return None

    def _compose_cmd(self, *args) -> subprocess.CompletedProcess:
        """Run a docker compose command"""
        if not self._compose_file:
            logger.error("docker-compose.yml not found")
            return subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="Compose file not found")

        cmd = ["docker", "compose", "-f", self._compose_file] + list(args)
        logger.info(f"Running compose command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            logger.error(f"Compose command failed: returncode={result.returncode}, stderr={result.stderr.strip()}, stdout={result.stdout.strip()}")
        else:
            logger.info(f"Compose command succeeded: stdout={result.stdout.strip()}")
        return result

    def get_engine_state(self, engine: EngineType) -> EngineState:
        """Get current state of an engine"""
        container_name = CONTAINER_NAMES.get(engine)
        port = DEFAULT_PORTS.get(engine, 0)
        api_base = f"http://localhost:{port}"
        external_port = self.config.ports.get(engine.value, port)

        if not docker_manager.is_available():
            return EngineState(
                engine=engine, status=EngineStatus.UNKNOWN,
                port=external_port, api_base=api_base, error="Docker not available"
            )

        status_info = docker_manager.get_container_status(container_name)

        if not status_info.get("exists"):
            return EngineState(
                engine=engine, status=EngineStatus.STOPPED,
                port=external_port, api_base=api_base
            )

        status_map = {
            "running": EngineStatus.RUNNING,
            "exited": EngineStatus.STOPPED,
            "restarting": EngineStatus.STARTING,
            "paused": EngineStatus.STOPPED,
            "dead": EngineStatus.ERROR,
        }
        status = status_map.get(status_info.get("status", ""), EngineStatus.UNKNOWN)

        running_model = None
        if status == EngineStatus.RUNNING:
            running_model = self._detect_running_model(engine, api_base)

        return EngineState(
            engine=engine, status=status, port=external_port,
            container_id=status_info.get("id"), container_name=container_name,
            running_model=running_model, api_base=api_base
        )

    def _detect_running_model(self, engine: EngineType, api_base: str) -> Optional[str]:
        """Detect which model is running on an engine"""
        try:
            if engine == EngineType.OLLAMA:
                return None
            with httpx.Client(timeout=5) as client:
                response = client.get(f"{api_base}/v1/models")
                if response.status_code == 200:
                    models = response.json().get("data", [])
                    if models:
                        return models[0].get("id")
        except Exception as e:
            logger.debug(f"Could not detect running model for {engine}: {e}")
        return None

    def get_all_engines(self) -> List[EngineState]:
        """Get state of all engines"""
        states = []
        for engine in EngineType:
            cfg = self.config.engines.get(engine.value, EngineConfig())
            if cfg.enabled:
                states.append(self.get_engine_state(engine))
        return states

    def _validate_model_path(self, model_path: str, engine: EngineType) -> tuple[bool, str, str]:
        """Validate model path before starting engine"""
        if not model_path:
            return True, "", model_path

        if os.path.exists(model_path):
            if os.path.isdir(model_path):
                return self._validate_local_directory(model_path)
            return True, "", model_path

        model_id = self._extract_hf_model_id(model_path)
        if model_id:
            return True, "", model_id

        return False, f"Model path does not exist: {model_path}. Download it from the HuggingFace tab first.", model_path

    def _extract_hf_model_id(self, model_path: str) -> Optional[str]:
        """Extract a HuggingFace model ID from a local path or direct ID"""
        models_dir = self.config.paths.get("models", "/models")

        if model_path.startswith(models_dir + "/"):
            rel_path = model_path[len(models_dir) + 1:]
            first_underscore = rel_path.find("_")
            if first_underscore > 0:
                namespace = rel_path[:first_underscore]
                name = rel_path[first_underscore + 1:]
                if namespace and name:
                    return f"{namespace}/{name}"

        if "/" in model_path and not model_path.startswith("/"):
            parts = model_path.split("/")
            if len(parts) == 2 and all(part.strip() for part in parts):
                return model_path

        return None

    def _validate_local_directory(self, model_path: str) -> tuple[bool, str, str]:
        """Validate a local model directory has required files"""
        config_path = os.path.join(model_path, "config.json")
        if not os.path.exists(config_path):
            if "models--" in model_path:
                snapshots_dir = os.path.join(model_path, "snapshots")
                if os.path.exists(snapshots_dir):
                    snapshots = [d for d in os.listdir(snapshots_dir) if os.path.isdir(os.path.join(snapshots_dir, d))]
                    if snapshots:
                        snapshot_path = os.path.join(snapshots_dir, snapshots[0])
                        if os.path.exists(os.path.join(snapshot_path, "config.json")):
                            return self._validate_local_directory(snapshot_path)
            return False, f"Model directory does not contain config.json: {model_path}", model_path

        has_weights = any(
            f.endswith(('.safetensors', '.bin', '.gguf'))
            for f in os.listdir(model_path)
        )
        if not has_weights:
            return False, f"Model directory does not contain weight files: {model_path}", model_path

        return True, "", model_path

    def start_engine(self, engine: EngineType, model: Optional[str] = None,
                     port: Optional[int] = None, **kwargs) -> bool:
        """Start an engine with a model using docker compose run"""
        service = COMPOSE_SERVICES.get(engine)
        if not service:
            logger.error(f"Unknown engine: {engine}")
            return False

        if not docker_manager.is_available():
            logger.error(f"Docker not available: {docker_manager.get_error()}")
            return False

        # Validate model
        resolved_model = model
        if model:
            is_valid, error_msg, resolved_model = self._validate_model_path(model, engine)
            logger.info(f"Model validation: valid={is_valid}, model={resolved_model}, error={error_msg}")
            if not is_valid:
                logger.error(f"Model validation failed: {error_msg}")
                return False

        # Build command for the engine
        command = self._build_engine_command(engine, resolved_model, **kwargs)
        logger.info(f"Engine command: {command}")

        if not command:
            logger.error(f"No command built for {engine.value}")
            return False

        # Stop existing container if running
        self.stop_engine(engine)

        # Start with docker compose run
        # --rm: remove container when stopped
        # -d: detach
        compose_args = [
            "run", "--rm", "-d",
            "--name", CONTAINER_NAMES.get(engine, f"dgx-{engine.value}"),
            service,
        ] + command.split()

        logger.info(f"Starting engine: docker compose {' '.join(compose_args)}")
        result = self._compose_cmd(*compose_args)

        if result.returncode == 0:
            logger.info(f"Started engine {engine.value} (service={service})")
            return True

        logger.error(f"Failed to start engine {engine.value}: returncode={result.returncode}, stderr={result.stderr.strip()}, stdout={result.stdout.strip()}")
        return False

    def _build_engine_command(self, engine: EngineType, model: Optional[str] = None, **kwargs) -> Optional[str]:
        """Build startup command for engine"""
        if not model:
            return None

        # Only convert local /models/org_name path to HF model ID if the
        # model does NOT exist on disk. For engines that mount /models/
        # (vLLM, SGLang, llama.cpp), keep using the local path when available.
        if model.startswith("/models/") and "_" in model.split("/models/")[1] and not os.path.exists(model):
            hf_id = self._extract_hf_model_id(model)
            if hf_id:
                logger.info(f"Converted model path to HF model ID: {model} -> {hf_id}")
                model = hf_id

        if engine == EngineType.VLLM:
            return self._build_vllm_command(model, **kwargs)
        elif engine == EngineType.SGLANG:
            return self._build_sglang_command(model, **kwargs)
        elif engine == EngineType.LLAMACPP:
            return self._build_llamacpp_command(model, **kwargs)
        elif engine == EngineType.OLLAMA:
            return f"serve"  # Ollama just needs to run, model loaded via API
        elif engine == EngineType.LITELLM:
            return None  # LiteLLM uses its own config

        return None

    def _build_vllm_command(self, model: str, **kwargs) -> str:
        """Build vLLM serve command"""
        tp_size = kwargs.get("tensor_parallel_size", 1)
        gpu_mem = kwargs.get("gpu_memory_utilization", 0.8)

        cmd_parts = [f"serve {model}"]

        # Detect quantization
        quantization = self._detect_quantization(model)
        if quantization:
            cmd_parts.append(f"--quantization {quantization}")

        cmd_parts.extend([
            "--trust-remote-code",
            "--host 0.0.0.0",
            "--port 8000",
            f"--tensor-parallel-size {tp_size}",
            f"--gpu-memory-utilization {gpu_mem}",
        ])

        return " ".join(cmd_parts)

    def _build_sglang_command(self, model: str, **kwargs) -> str:
        """Build SGLang launch command"""
        tp_size = kwargs.get("tensor_parallel_size", 1)
        return (
            f"python3 -m sglang.launch_server "
            f"--model-path {model} --host 0.0.0.0 --port 30000 "
            f"--tensor-parallel-size {tp_size} --trust-remote-code"
        )

    def _build_llamacpp_command(self, model: str, **kwargs) -> str:
        """Build llama.cpp server command"""
        return f"--model {model} --host 0.0.0.0 --port 8080"

    def _detect_quantization(self, model: str) -> Optional[str]:
        """Detect quantization from model config or name"""
        # Try config.json first
        model_exists = os.path.isdir(model) or os.path.isfile(model)
        if model_exists:
            config_path = os.path.join(model, "config.json")
            if os.path.exists(config_path):
                try:
                    with open(config_path) as f:
                        config = json.load(f)
                    qc = config.get("quantization_config", {})
                    quant_method = qc.get("quant_method", "")
                    if quant_method:
                        return quant_method
                except Exception:
                    pass

        # Fallback: detect from model name
        model_lower = model.lower()
        if "nvfp4" in model_lower or "fp4" in model_lower:
            return "compressed-tensors"
        elif "fp8" in model_lower:
            return "fp8"
        elif "awq" in model_lower:
            return "awq"
        elif "gptq" in model_lower:
            return "gptq"
        return None

    def stop_engine(self, engine: EngineType) -> bool:
        """Stop an engine"""
        service = COMPOSE_SERVICES.get(engine)
        if not service:
            return False

        container_name = CONTAINER_NAMES.get(engine)
        if container_name:
            # Try compose stop first
            result = self._compose_cmd("stop", service)
            # Also force-remove the container
            docker_manager.remove_container(container_name, force=True)

        return True

    def restart_engine(self, engine: EngineType) -> bool:
        """Restart an engine"""
        self.stop_engine(engine)
        return True

    def get_engine_logs(self, engine: EngineType, tail: int = 100) -> str:
        """Get engine logs"""
        container_name = CONTAINER_NAMES.get(engine)
        if not container_name:
            return ""
        return docker_manager.get_container_logs(container_name, tail=tail)

    def list_profiles(self, engine: EngineType) -> List[EngineProfile]:
        """List available profiles for an engine"""
        return []

    def start_with_profile(self, engine: EngineType, profile_name: str) -> bool:
        """Start engine using a profile (deprecated, use start_engine instead)"""
        return False


# Singleton instance
engine_manager = EngineManager()
