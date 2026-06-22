"""
Pydantic models for API request/response schemas
"""

from typing import Optional, List
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field


# ============================================
# Enums
# ============================================

class EngineType(str, Enum):
    """Supported inference engine types"""
    OLLAMA = "ollama"
    SGLANG = "sglang"
    VLLM = "vllm"
    LLAMACPP = "llamacpp"
    LOCALAI = "localai"
    COMFYUI = "comfyui"


class EngineStatus(str, Enum):
    """Engine status"""
    RUNNING = "running"
    STOPPED = "stopped"
    STARTING = "starting"
    ERROR = "error"
    UNKNOWN = "unknown"


class ModelSource(str, Enum):
    """Model source"""
    OLLAMA = "ollama"
    HUGGINGFACE = "huggingface"
    LOCAL = "local"
    CUSTOM = "custom"


# ============================================
# System Models
# ============================================

class GPUInfo(BaseModel):
    """GPU information"""
    index: int
    name: str
    memory_total: int  # MB
    memory_used: int
    memory_free: int
    utilization: float = 0.0


class SystemStatus(BaseModel):
    """System status information"""
    hostname: str
    ip_address: str
    architecture: str
    python_version: str
    uptime: str
    cpu_count: int
    cpu_percent: float
    ram_total: int
    ram_used: int
    ram_free: int
    disk_total: int
    disk_used: int
    disk_free: int
    gpus: List[GPUInfo]
    docker_version: Optional[str] = None


# ============================================
# Engine Models
# ============================================

class EngineProfile(BaseModel):
    """Engine profile (startup script)"""
    name: str
    filename: str
    description: Optional[str] = None
    vram_required: Optional[int] = None
    model_path: Optional[str] = None
    port: Optional[int] = None


class EngineState(BaseModel):
    """Engine state information"""
    engine: EngineType
    status: EngineStatus
    port: int
    container_id: Optional[str] = None
    container_name: Optional[str] = None
    running_model: Optional[str] = None
    uptime: Optional[str] = None
    error: Optional[str] = None
    health_check_url: Optional[str] = None
    api_base: str


class EngineControl(BaseModel):
    """Engine control request"""
    action: str = Field(..., pattern="^(start|stop|restart)$")
    profile: Optional[str] = None
    model: Optional[str] = None
    port: Optional[int] = None
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    extra_args: Optional[dict] = None


# ============================================
# Ollama Models
# ============================================

class OllamaModel(BaseModel):
    """Ollama model information"""
    name: str
    model_id: str
    size: int
    size_human: str
    digest: str
    modified_at: datetime
    details: dict = {}


class OllamaPullRequest(BaseModel):
    """Ollama pull request"""
    model: str
    tag: str = "latest"


class OllamaPullStatus(BaseModel):
    """Ollama pull status"""
    model: str
    status: str
    completed: Optional[int] = None
    total: Optional[int] = None


# ============================================
# HuggingFace Models
# ============================================

class HFModel(BaseModel):
    """HuggingFace model info"""
    id: str
    author: Optional[str] = None
    model_id: str
    pipeline_tag: Optional[str] = None
    tags: List[str] = []
    downloads: int = 0
    likes: int = 0
    last_modified: Optional[datetime] = None
    private: bool = False
    gated: Optional[str] = None


class HFSearchRequest(BaseModel):
    """Search request for HuggingFace"""
    query: str = Field(..., min_length=1)
    limit: int = Field(20, ge=1, le=100)
    sort: Optional[str] = None
    pipeline_tag: Optional[str] = None
    author: Optional[str] = None


class HFSearchResponse(BaseModel):
    """Search response from HuggingFace"""
    models: List[HFModel]
    total: int
    query: str


class HFDownloadRequest(BaseModel):
    """HuggingFace download request"""
    model_id: str
    revision: Optional[str] = None
    local_dir: Optional[str] = None


class HFDownloadStatus(BaseModel):
    """HuggingFace download status"""
    model_id: str
    status: str  # pending, downloading, completed, failed
    progress: Optional[float] = None
    speed: Optional[str] = None
    eta: Optional[str] = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


# ============================================
# Inventory Models
# ============================================

class InventoryItem(BaseModel):
    """Unified inventory item"""
    id: str
    name: str
    source: ModelSource
    path: Optional[str] = None
    size_bytes: int = 0
    size_human: str = ""
    format: Optional[str] = None
    quantization: Optional[str] = None
    engine: Optional[EngineType] = None
    tags: List[str] = []
    metadata: dict = {}


class InventoryResponse(BaseModel):
    """Inventory response"""
    items: List[InventoryItem]
    total: int
    total_size_bytes: int
    total_size_human: str
    sources: dict[str, int]  # source -> count


# ============================================
# LiteLLM Models
# ============================================

class LiteLLMRoute(BaseModel):
    """LiteLLM route"""
    model_name: str
    litellm_params: dict
    status: str = "active"


class LiteLLMConfig(BaseModel):
    """LiteLLM configuration"""
    model_list: List[LiteLLMRoute]
    litellm_settings: dict = {}
    general_settings: dict = {}


# ============================================
# Status Models
# ============================================

class ServiceHealth(BaseModel):
    """Service health information"""
    name: str
    status: str
    url: str
    latency_ms: Optional[float] = None
    error: Optional[str] = None


class AggregatedStatus(BaseModel):
    """Aggregated status for all services"""
    manager: ServiceHealth
    ollama: ServiceHealth
    litellm: ServiceHealth
    sglang: Optional[ServiceHealth] = None
    vllm: Optional[ServiceHealth] = None
    llamacpp: Optional[ServiceHealth] = None
    localai: Optional[ServiceHealth] = None
    comfyui: Optional[ServiceHealth] = None
    system: SystemStatus


# ============================================
# API Response Models
# ============================================

class APIResponse(BaseModel):
    """Generic API response"""
    success: bool
    message: str
    data: Optional[dict] = None


class APIError(BaseModel):
    """API error response"""
    detail: str
    error_code: Optional[str] = None