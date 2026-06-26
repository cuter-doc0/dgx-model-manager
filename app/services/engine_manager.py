"""
Engine manager service - Manage inference engines (SGLang, vLLM, llama.cpp, etc.)
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

# Container name mapping
CONTAINER_NAMES = {
    EngineType.OLLAMA: "dgx-ollama",
    EngineType.SGLANG: "dgx-sglang",
    EngineType.VLLM: "dgx-vllm",
    EngineType.LLAMACPP: "dgx-llamacpp",
    EngineType.LOCALAI: "dgx-localai",
    EngineType.COMFYUI: "dgx-comfyui",
}

# Port mapping
DEFAULT_PORTS = {
    EngineType.OLLAMA: 11434,
    EngineType.SGLANG: 30000,
    EngineType.VLLM: 8000,
    EngineType.LLAMACPP: 8080,
    EngineType.LOCALAI: 9090,
    EngineType.COMFYUI: 8188,
}

# Image mapping
ENGINE_IMAGES = {
    EngineType.OLLAMA: "ollama/ollama:latest",
    EngineType.SGLANG: "lmsysorg/sglang:latest",
    EngineType.VLLM: "vllm/vllm-openai:latest",
    EngineType.LLAMACPP: "ghcr.io/ggerganov/llama.cpp:server",
    EngineType.LOCALAI: "localai/localai:latest",
    EngineType.COMFYUI: "yanwk/comfyui-boot:latest",
}

# Script directories
ENGINE_DIRS = {
    EngineType.SGLANG: Path("/app/engines/sglang"),
    EngineType.VLLM: Path("/app/engines/vllm"),
    EngineType.LLAMACPP: Path("/app/engines/llamacpp"),
    EngineType.LOCALAI: Path("/app/engines/localai"),
    EngineType.COMFYUI: Path("/app/engines/comfyui"),
}


class EngineManager:
    """Manage inference engines"""
    
    def __init__(self):
        self.config = get_config()
    
    def get_engine_state(self, engine: EngineType) -> EngineState:
        """Get current state of an engine"""
        container_name = CONTAINER_NAMES.get(engine)
        port = DEFAULT_PORTS.get(engine, 0)
        api_base = getattr(self.config.services, f"{engine.value}_base", f"http://localhost:{port}")
        
        # Get external port from config (for display to users)
        external_port = self.config.ports.get(engine.value, port)
        
        if not docker_manager.is_available():
            return EngineState(
                engine=engine,
                status=EngineStatus.UNKNOWN,
                port=external_port,
                api_base=api_base,
                error="Docker not available"
            )
        
        status_info = docker_manager.get_container_status(container_name)
        
        if not status_info.get("exists"):
            return EngineState(
                engine=engine,
                status=EngineStatus.STOPPED,
                port=external_port,
                api_base=api_base
            )
        
        status_map = {
            "running": EngineStatus.RUNNING,
            "exited": EngineStatus.STOPPED,
            "restarting": EngineStatus.STARTING,
            "paused": EngineStatus.STOPPED,
            "dead": EngineStatus.ERROR,
        }
        
        status = status_map.get(status_info.get("status", ""), EngineStatus.UNKNOWN)
        
        # Try to get running model
        running_model = None
        if status == EngineStatus.RUNNING:
            running_model = self._detect_running_model(engine, api_base)
        
        return EngineState(
            engine=engine,
            status=status,
            port=external_port,
            container_id=status_info.get("id"),
            container_name=container_name,
            running_model=running_model,
            api_base=api_base
        )
    
    def _detect_running_model(self, engine: EngineType, api_base: str) -> Optional[str]:
        """Detect which model is running on an engine"""
        try:
            if engine == EngineType.OLLAMA:
                # Ollama doesn't have a direct "running" endpoint
                # We'd need to check process or use ps
                return None
            
            elif engine in [EngineType.SGLANG, EngineType.VLLM, EngineType.LLAMACPP]:
                # These use OpenAI-compatible API
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
            if self.config.engines.get(engine.value, EngineConfig()).enabled:
                states.append(self.get_engine_state(engine))
        return states
    
    def _validate_model_path(self, model_path: str, engine: EngineType) -> tuple[bool, str, str]:
        """Validate model path before starting engine
        
        Returns:
            Tuple of (is_valid, error_message, resolved_path)
        """
        if not model_path:
            return True, "", model_path
        
        # Path exists locally - validate it
        if os.path.exists(model_path):
            if os.path.isdir(model_path):
                return self._validate_local_directory(model_path)
            return True, "", model_path  # Single file (e.g., GGUF)
        
        # Path doesn't exist locally - try to resolve it as a HuggingFace model ID
        model_id = self._extract_hf_model_id(model_path)
        if model_id:
            # Let vLLM auto-download from HuggingFace
            return True, "", model_id
        
        return False, f"Model path does not exist: {model_path}. Download it from the HuggingFace tab first.", model_path
    
    def _extract_hf_model_id(self, model_path: str) -> Optional[str]:
        """Extract a HuggingFace model ID from a local path or direct ID
        
        Converts:
          /models/lyf_Qwen3.6-35B-A3B -> lyf/Qwen3.6-35B-A3B  (path under /models)
          lyf/Qwen3.6-35B-A3B        -> lyf/Qwen3.6-35B-A3B  (already an HF ID)
          /root/.cache/.../models--lyf--Qwen3.6-35B-A3B -> None (HF cache, handled by snapshot resolution)
        """
        models_dir = self.config.paths.get("models", "/models")
        
        if model_path.startswith(models_dir + "/"):
            rel_path = model_path[len(models_dir) + 1:]
            first_underscore = rel_path.find("_")
            if first_underscore > 0:
                namespace = rel_path[:first_underscore]
                name = rel_path[first_underscore + 1:]
                # Verify it looks like a valid HF model ID
                if namespace and name:
                    return f"{namespace}/{name}"
        
        # Direct HuggingFace model ID (contains / but doesn't start with /)
        if "/" in model_path and not model_path.startswith("/"):
            parts = model_path.split("/")
            if len(parts) == 2 and all(part.strip() for part in parts):
                return model_path
        
        return None
    
    def _validate_local_directory(self, model_path: str) -> tuple[bool, str, str]:
        """Validate a local model directory has required files"""
        # Check for config.json (required by vLLM/SGLang)
        config_path = os.path.join(model_path, "config.json")
        if not os.path.exists(config_path):
            # Check if this is a HuggingFace cache directory (models-- prefix)
            if "models--" in model_path:
                snapshots_dir = os.path.join(model_path, "snapshots")
                if os.path.exists(snapshots_dir):
                    snapshots = [d for d in os.listdir(snapshots_dir) if os.path.isdir(os.path.join(snapshots_dir, d))]
                    if snapshots:
                        snapshot_path = os.path.join(snapshots_dir, snapshots[0])
                        if os.path.exists(os.path.join(snapshot_path, "config.json")):
                            logger.info(f"Found HuggingFace cache directory, using snapshot: {snapshot_path}")
                            return self._validate_local_directory(snapshot_path)
            
            return False, f"Model directory does not contain config.json: {model_path}", model_path
        
        # Check for model weight files
        has_weights = any(
            f.endswith(('.safetensors', '.bin', '.gguf'))
            for f in os.listdir(model_path)
        )
        if not has_weights:
            return False, f"Model directory does not contain weight files: {model_path}", model_path
        
        return True, "", model_path
    
    def start_engine(self, engine: EngineType, model: Optional[str] = None, 
                     port: Optional[int] = None, **kwargs) -> bool:
        """Start an engine"""
        container_name = CONTAINER_NAMES.get(engine)
        image = ENGINE_IMAGES.get(engine)
        default_port = DEFAULT_PORTS.get(engine)
        
        if not container_name or not image:
            logger.error(f"Unknown engine: {engine}")
            return False
        
        if not docker_manager.is_available():
            logger.error("Docker not available")
            return False
        
        # Validate model path if provided
        resolved_model = model
        if model:
            is_valid, error_msg, resolved_model = self._validate_model_path(model, engine)
            if not is_valid:
                logger.error(f"Model validation failed: {error_msg}")
                return False
        
        # Check if already running
        state = self.get_engine_state(engine)
        if state.status == EngineStatus.RUNNING:
            logger.info(f"Engine {engine.value} already running")
            return True
        
        # Build port mapping
        actual_port = port or default_port
        ports = {f"{default_port}/tcp": actual_port}
        
        # Build volume mappings
        hf_cache = self.config.paths.get("hf_cache", "~/.cache/huggingface")
        models_path = self.config.paths.get("models", "/models")
        
        volumes = {
            hf_cache: {"bind": "/root/.cache/huggingface", "mode": "rw"},
        }
        
        # Add /models mount for engines that load local models
        if engine in [EngineType.VLLM, EngineType.SGLANG, EngineType.LLAMACPP, EngineType.LOCALAI, EngineType.OLLAMA]:
            volumes[models_path] = {"bind": "/models", "mode": "rw"}
        
        # Build environment
        environment = {
            "NVIDIA_VISIBLE_DEVICES": "all"
        }
        
        # Build command based on engine and model
        command = self._build_engine_command(engine, resolved_model, **kwargs)
        
        # Detect Docker network (look for compose project network)
        network = self._detect_docker_network()
        
        # Run container
        container = docker_manager.run_container(
            image=image,
            name=container_name,
            ports=ports,
            volumes=volumes,
            environment=environment,
            command=command,
            gpu=True,
            network=network
        )
        
        if container:
            logger.info(f"Started engine {engine.value} with container {container_name}")
            return True
        
        return False
    
    def _build_engine_command(self, engine: EngineType, model: Optional[str] = None, **kwargs) -> Optional[str]:
        """Build startup command for engine"""
        if not model:
            return None
        
        # Check if model path exists locally
        model_exists_locally = os.path.isdir(model) or os.path.isfile(model)
        
        if engine == EngineType.SGLANG:
            tp_size = kwargs.get("tensor_parallel_size", 1)
            return f"python3 -m sglang.launch_server --model-path {model} --host 0.0.0.0 --port 30000 --tensor-parallel-size {tp_size}"
        
        elif engine == EngineType.VLLM:
            tp_size = kwargs.get("tensor_parallel_size", 1)
            gpu_mem = kwargs.get("gpu_memory_utilization", 0.9)
            
            # Build vLLM command with optional flags
            cmd_parts = [model]
            
            # Detect quantization from config.json if available
            quantization = None
            if model_exists_locally:
                config_path = os.path.join(model, "config.json")
                if os.path.exists(config_path):
                    try:
                        with open(config_path) as f:
                            config = json.load(f)
                        qc = config.get("quantization_config", {})
                        quant_method = qc.get("quant_method", "")
                        if quant_method:
                            quantization = quant_method
                        elif "NVFP4" in str(config).upper():
                            quantization = "compressed-tensors"
                    except Exception:
                        pass
            
            # Fallback: detect from model name
            if not quantization:
                model_lower = model.lower()
                if "nvfp4" in model_lower or "fp4" in model_lower:
                    quantization = "compressed-tensors"
                elif "fp8" in model_lower:
                    quantization = "fp8"
                elif "awq" in model_lower:
                    quantization = "awq"
                elif "gptq" in model_lower:
                    quantization = "gptq"
                elif "int4" in model_lower:
                    quantization = "bitsandbytes"
                elif "int8" in model_lower:
                    quantization = "bitsandbytes"
            
            if quantization:
                cmd_parts.append(f"--quantization {quantization}")
            
            # Trust remote code for custom architectures
            if model_exists_locally:
                config_path = os.path.join(model, "config.json")
                if os.path.exists(config_path):
                    try:
                        with open(config_path) as f:
                            config = json.load(f)
                        if config.get("auto_map") or config.get("architectures", [])[0:1] not in [["LlamaForCausalLM"], ["MistralForCausalLM"]]:
                            cmd_parts.append("--trust-remote-code")
                    except Exception:
                        cmd_parts.append("--trust-remote-code")
                else:
                    cmd_parts.append("--trust-remote-code")
            else:
                cmd_parts.append("--trust-remote-code")
            
            # Add common options
            cmd_parts.extend([
                "--host 0.0.0.0",
                "--port 8000",
                f"--tensor-parallel-size {tp_size}",
                f"--gpu-memory-utilization {gpu_mem}"
            ])
            
            # vLLM Docker image entrypoint is already "vllm serve"
            # Just pass the model as positional argument with options
            return " ".join(cmd_parts)
        
        elif engine == EngineType.LLAMACPP:
            return f"--model {model} --host 0.0.0.0 --port 8080"
        
        return None
    
    def _detect_docker_network(self) -> Optional[str]:
        """Detect the Docker Compose network for this project"""
        try:
            if not docker_manager.is_available():
                return None
            
            # Use Docker CLI to list networks
            result = docker_manager._run([
                "network", "ls", "--format", "{{.Name}}"
            ])
            
            if result.returncode == 0:
                for net_name in result.stdout.strip().split("\n"):
                    if 'model-network' in net_name:
                        return net_name
            
            return None
        except Exception as e:
            logger.debug(f"Could not detect Docker network: {e}")
            return None
    
    def stop_engine(self, engine: EngineType) -> bool:
        """Stop an engine"""
        container_name = CONTAINER_NAMES.get(engine)
        if not container_name:
            return False
        
        return docker_manager.stop_container(container_name)
    
    def restart_engine(self, engine: EngineType) -> bool:
        """Restart an engine"""
        container_name = CONTAINER_NAMES.get(engine)
        if not container_name:
            return False
        
        return docker_manager.restart_container(container_name)
    
    def get_engine_logs(self, engine: EngineType, tail: int = 100) -> str:
        """Get engine logs"""
        container_name = CONTAINER_NAMES.get(engine)
        if not container_name:
            return ""
        
        return docker_manager.get_container_logs(container_name, tail=tail)
    
    def list_profiles(self, engine: EngineType) -> List[EngineProfile]:
        """List available profiles for an engine"""
        profiles = []
        engine_dir = ENGINE_DIRS.get(engine)
        
        if not engine_dir or not engine_dir.exists():
            return profiles
        
        for script_file in engine_dir.glob("start_*.sh"):
            profile = self._parse_profile(script_file, engine)
            if profile:
                profiles.append(profile)
        
        return profiles
    
    def _parse_profile(self, script_path: Path, engine: EngineType) -> Optional[EngineProfile]:
        """Parse profile from script file"""
        try:
            with open(script_path, "r") as f:
                content = f.read()
            
            lines = content.split("\n")
            name = None
            description = None
            vram = None
            
            for line in lines[:20]:  # Only check first 20 lines
                if line.startswith("# Name:"):
                    name = line.split(":", 1)[1].strip()
                elif line.startswith("# Description:"):
                    description = line.split(":", 1)[1].strip()
                elif line.startswith("# VRAM:"):
                    try:
                        vram = int(line.split(":", 1)[1].strip())
                    except ValueError:
                        pass
            
            if not name:
                name = script_path.stem.replace("start_", "").replace("_", " ").title()
            
            return EngineProfile(
                name=name,
                filename=script_path.name,
                description=description,
                vram_required=vram,
                port=DEFAULT_PORTS.get(engine)
            )
            
        except Exception as e:
            logger.error(f"Error parsing profile {script_path}: {e}")
            return None
    
    def start_with_profile(self, engine: EngineType, profile_name: str) -> bool:
        """Start engine using a profile script"""
        engine_dir = ENGINE_DIRS.get(engine)
        if not engine_dir:
            return False
        
        script_path = engine_dir / profile_name
        if not script_path.exists():
            logger.error(f"Profile not found: {profile_name}")
            return False
        
        try:
            # Make script executable
            os.chmod(script_path, 0o755)
            
            # Execute script
            result = subprocess.run(
                ["bash", str(script_path)],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode != 0:
                logger.error(f"Profile script failed: {result.stderr}")
                return False
            
            logger.info(f"Started engine {engine.value} with profile {profile_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error starting with profile: {e}")
            return False


# Singleton instance
engine_manager = EngineManager()