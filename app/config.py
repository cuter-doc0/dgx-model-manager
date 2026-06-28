"""
Configuration management for DGX Spark Model Manager
"""

import os
import json
import logging
from pathlib import Path
from typing import Any, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

CONFIG_PATH = Path("/app/config/config.json")
LITELLM_CONFIG_PATH = Path("/app/config/litellm_config.yaml")


class AppConfig(BaseModel):
    """Application configuration"""
    host: str = "0.0.0.0"
    port: int = 4600
    name: str = "DGX Spark Model Manager"


class ServiceConfig(BaseModel):
    """Service URLs"""
    ollama_base: str = "http://dgx-ollama:11434"
    litellm_base: str = "http://dgx-litellm:4000"
    sglang_base: str = "http://dgx-sglang:30000"
    vllm_base: str = "http://dgx-vllm:8000"
    llamacpp_base: str = "http://dgx-llamacpp:8080"


class EngineConfig(BaseModel):
    """Engine configuration"""
    enabled: bool = True
    auto_start: bool = False


class InventoryConfig(BaseModel):
    """Inventory configuration"""
    scan_dirs: list[str] = Field(default_factory=lambda: ["/models", "/root/.cache/huggingface/hub"])
    auto_refresh: bool = True
    refresh_interval: int = 300


class SecurityConfig(BaseModel):
    """Security configuration"""
    api_key_hash: Optional[str] = None
    allow_unauth: bool = True


class Settings(BaseModel):
    """Complete application settings"""
    app: AppConfig = Field(default_factory=AppConfig)
    services: ServiceConfig = Field(default_factory=ServiceConfig)
    ports: dict[str, int] = Field(default_factory=lambda: {
        "manager": 4600,
        "ollama": 4610,
        "litellm": 4601,
        "sglang": 4620,
        "vllm": 4630,
        "llamacpp": 4640,
    })
    paths: dict[str, str] = Field(default_factory=lambda: {
        "models": "/models",
        "hf_cache": "/root/.cache/huggingface",
        "litellm_config": "/app/config/litellm_config.yaml"
    })
    engines: dict[str, EngineConfig] = Field(default_factory=lambda: {
        "ollama": EngineConfig(enabled=True, auto_start=False),
        "vllm": EngineConfig(enabled=True, auto_start=False),
        "sglang": EngineConfig(enabled=True, auto_start=False),
        "llamacpp": EngineConfig(enabled=True, auto_start=False),
        "litellm": EngineConfig(enabled=True, auto_start=False),
    })
    inventory: InventoryConfig = Field(default_factory=InventoryConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)


def load_config() -> Settings:
    """Load configuration from file"""
    try:
        if CONFIG_PATH.exists():
            with open(CONFIG_PATH, "r") as f:
                data = json.load(f)
                return Settings(**data)
    except Exception as e:
        logger.error(f"Error loading config: {e}")
    
    return Settings()


def save_config(settings: Settings) -> bool:
    """Save configuration to file"""
    try:
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CONFIG_PATH, "w") as f:
            json.dump(settings.model_dump(), f, indent=2)
        return True
    except Exception as e:
        logger.error(f"Error saving config: {e}")
        return False


def get_config() -> Settings:
    """Get current configuration (cached)"""
    if not hasattr(get_config, "_cache"):
        get_config._cache = load_config()
    return get_config._cache


def reload_config() -> Settings:
    """Force reload configuration"""
    get_config._cache = load_config()
    return get_config._cache
