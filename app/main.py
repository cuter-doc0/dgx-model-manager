"""
DGX Spark Model Manager - Main Application
Multi-engine inference management with LiteLLM routing
"""

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from typing import Optional

from app.config import load_config, save_config, reload_config, get_config
from app.models import (
    EngineType, EngineControl, SystemStatus,
    OllamaPullRequest, HFSearchRequest, HFDownloadRequest,
    APIResponse, LiteLLMRoute
)
from app.services import (
    docker_manager, engine_manager, ollama_service,
    litellm_service, inventory_service, hf_service, system_monitor
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    print("DGX Spark Model Manager starting...")
    yield
    print("DGX Spark Model Manager shutting down...")


app = FastAPI(
    title="DGX Spark Model Manager",
    description="Multi-engine inference management with LiteLLM routing",
    version="2.0.0",
    lifespan=lifespan
)

# Mount static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")


# ============================================
# Root & Health Endpoints
# ============================================

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the web UI"""
    with open("app/static/index.html", "r") as f:
        return HTMLResponse(content=f.read())


@app.get("/api/status")
async def get_status():
    """Get aggregated status for all services"""
    try:
        # Get system status
        sys_status = system_monitor.get_status()
        
        # Get engine states
        engines = engine_manager.get_all_engines()
        
        # Get Ollama status
        ollama_available = ollama_service.is_available()
        
        # Get LiteLLM status
        litellm_available = litellm_service.is_available()
        
        return {
            "system": sys_status,
            "engines": engines,
            "ollama_available": ollama_available,
            "litellm_available": litellm_available
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return system_monitor.get_health_check()


# ============================================
# System Endpoints
# ============================================

@app.get("/api/system")
async def get_system_status():
    """Get system status"""
    return system_monitor.get_status()


@app.get("/api/system/gpus")
async def get_gpu_info():
    """Get GPU information"""
    status = system_monitor.get_status()
    return {"gpus": status.gpus}


# ============================================
# Engine Endpoints
# ============================================

@app.get("/api/engines")
async def list_engines():
    """List all engines and their states"""
    return {"engines": engine_manager.get_all_engines()}


@app.get("/api/engines/{engine}")
async def get_engine_state(engine: EngineType):
    """Get state of a specific engine"""
    return engine_manager.get_engine_state(engine)


@app.post("/api/engines/{engine}/control")
async def control_engine(engine: EngineType, control: EngineControl):
    """Control an engine (start/stop/restart)"""
    if control.action == "start":
        success = engine_manager.start_engine(
            engine,
            model=control.model,
            port=control.port,
            tensor_parallel_size=control.tensor_parallel_size,
            gpu_memory_utilization=control.gpu_memory_utilization,
            **(control.extra_args or {})
        )
    elif control.action == "stop":
        success = engine_manager.stop_engine(engine)
    elif control.action == "restart":
        success = engine_manager.restart_engine(engine)
    else:
        raise HTTPException(status_code=400, detail=f"Unknown action: {control.action}")
    
    if success:
        return APIResponse(success=True, message=f"Engine {engine.value} {control.action} successful")
    else:
        raise HTTPException(status_code=500, detail=f"Failed to {control.action} engine")


@app.get("/api/engines/{engine}/logs")
async def get_engine_logs(engine: EngineType, tail: int = Query(100, ge=1, le=1000)):
    """Get engine logs"""
    logs = engine_manager.get_engine_logs(engine, tail=tail)
    return {"engine": engine.value, "logs": logs}


# ============================================
# Ollama Endpoints
# ============================================

@app.get("/api/ollama/models")
async def list_ollama_models():
    """List Ollama models"""
    models = ollama_service.list_models()
    return {"models": models, "total": len(models)}


@app.get("/api/ollama/models/{name}")
async def get_ollama_model(name: str):
    """Get Ollama model info"""
    model = ollama_service.get_model(name)
    if model:
        return model
    raise HTTPException(status_code=404, detail="Model not found")


@app.post("/api/ollama/pull")
async def pull_ollama_model(request: OllamaPullRequest, background_tasks: BackgroundTasks):
    """Pull an Ollama model"""
    background_tasks.add_task(ollama_service.pull_model, request.model, request.tag)
    return APIResponse(success=True, message=f"Started pulling {request.model}:{request.tag}")


@app.delete("/api/ollama/models/{name}")
async def delete_ollama_model(name: str):
    """Delete an Ollama model"""
    success = ollama_service.delete_model(name)
    if success:
        return APIResponse(success=True, message=f"Deleted {name}")
    raise HTTPException(status_code=500, detail="Failed to delete model")


@app.get("/api/ollama/status")
async def get_ollama_status():
    """Get Ollama status"""
    return {"available": ollama_service.is_available()}


# ============================================
# LiteLLM Endpoints
# ============================================

@app.get("/api/litellm/models")
async def list_litellm_models():
    """List LiteLLM available models"""
    models = litellm_service.get_models()
    return {"models": models, "total": len(models)}


@app.get("/api/litellm/config")
async def get_litellm_config():
    """Get LiteLLM configuration"""
    config = litellm_service.get_config()
    if config:
        return config
    raise HTTPException(status_code=404, detail="LiteLLM config not found")


@app.post("/api/litellm/wildcard")
async def apply_wildcard(engine: str = "ollama", api_base: Optional[str] = None):
    """Apply wildcard routing for an engine"""
    success = litellm_service.apply_wildcard(engine, api_base)
    if success:
        return APIResponse(success=True, message=f"Applied wildcard for {engine}")
    raise HTTPException(status_code=500, detail="Failed to apply wildcard")


@app.post("/api/litellm/routes")
async def add_litellm_route(model_name: str, litellm_params: dict):
    """Add a LiteLLM route"""
    success = litellm_service.add_route(model_name, litellm_params)
    if success:
        return APIResponse(success=True, message=f"Added route {model_name}")
    raise HTTPException(status_code=500, detail="Failed to add route")


@app.delete("/api/litellm/routes/{model_name}")
async def remove_litellm_route(model_name: str):
    """Remove a LiteLLM route"""
    success = litellm_service.remove_route(model_name)
    if success:
        return APIResponse(success=True, message=f"Removed route {model_name}")
    raise HTTPException(status_code=500, detail="Failed to remove route")


@app.get("/api/litellm/status")
async def get_litellm_status():
    """Get LiteLLM status"""
    return {
        "available": litellm_service.is_available(),
        "health": litellm_service.get_health()
    }


# ============================================
# Inventory Endpoints
# ============================================

@app.get("/api/inventory")
async def get_inventory(
    source: Optional[str] = Query(None, description="Filter by source: ollama, huggingface, local"),
    search: Optional[str] = Query(None, description="Search query")
):
    """Get unified model inventory"""
    from app.models import ModelSource
    
    source_enum = None
    if source:
        try:
            source_enum = ModelSource(source)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid source: {source}")
    
    return inventory_service.get_inventory(source=source_enum, search=search)


@app.post("/api/inventory/scan")
async def scan_inventory_directory(directory: str):
    """Add a directory to scan list"""
    success = inventory_service.scan_directory(directory)
    if success:
        return APIResponse(success=True, message=f"Added directory {directory}")
    raise HTTPException(status_code=500, detail="Failed to add directory")


@app.delete("/api/inventory/scan")
async def remove_inventory_directory(directory: str):
    """Remove a directory from scan list"""
    success = inventory_service.remove_directory(directory)
    if success:
        return APIResponse(success=True, message=f"Removed directory {directory}")
    raise HTTPException(status_code=500, detail="Failed to remove directory")


@app.delete("/api/inventory/{item_id}")
async def delete_inventory_item(item_id: str):
    """Delete an inventory item"""
    success = inventory_service.delete_item(item_id)
    if success:
        return APIResponse(success=True, message=f"Deleted {item_id}")
    raise HTTPException(status_code=500, detail="Failed to delete item")


# ============================================
# HuggingFace Endpoints
# ============================================

@app.get("/api/hf/search")
async def search_hf_models(
    query: str = Query(..., min_length=1),
    limit: int = Query(20, ge=1, le=100),
    sort: Optional[str] = Query(None),
    pipeline_tag: Optional[str] = Query(None),
    author: Optional[str] = Query(None)
):
    """Search HuggingFace Hub"""
    return hf_service.search_models(
        query=query,
        limit=limit,
        sort=sort,
        pipeline_tag=pipeline_tag,
        author=author
    )


@app.get("/api/hf/model/{model_id:path}")
async def get_hf_model_info(model_id: str):
    """Get HuggingFace model info"""
    model = hf_service.get_model_info(model_id)
    if model:
        return model
    raise HTTPException(status_code=404, detail="Model not found")


@app.get("/api/hf/model/{model_id:path}/files")
async def get_hf_model_files(model_id: str):
    """Get HuggingFace model files"""
    files = hf_service.get_model_files(model_id)
    return {"model_id": model_id, "files": files}


@app.post("/api/hf/download")
async def download_hf_model(request: HFDownloadRequest):
    """Start a background download"""
    task = hf_service.start_download(
        request.model_id,
        request.revision,
        request.local_dir
    )
    return {"message": "Download started", "model_id": request.model_id}


@app.get("/api/hf/downloads")
async def list_downloads():
    """List all downloads"""
    downloads = hf_service.get_all_downloads()
    return {"downloads": downloads, "total": len(downloads)}


@app.get("/api/hf/downloads/active")
async def list_active_downloads():
    """List active downloads"""
    downloads = hf_service.get_active_downloads()
    return {"downloads": downloads, "total": len(downloads)}


@app.get("/api/hf/downloads/{model_id:path}/status")
async def get_hf_download_status(model_id: str):
    """Get HuggingFace download status"""
    status = hf_service.get_download_status(model_id)
    if status:
        return status
    raise HTTPException(status_code=404, detail="Download not found")


@app.post("/api/hf/downloads/{model_id:path}/cancel")
async def cancel_hf_download(model_id: str):
    """Cancel an active download"""
    success = hf_service.cancel_download(model_id)
    if success:
        return APIResponse(success=True, message=f"Cancelled download {model_id}")
    raise HTTPException(status_code=404, detail="Download not found or not cancellable")


@app.post("/api/hf/downloads/{model_id:path}/pause")
async def pause_hf_download(model_id: str):
    """Pause an active download"""
    success = hf_service.pause_download(model_id)
    if success:
        return APIResponse(success=True, message=f"Paused download {model_id}")
    raise HTTPException(status_code=404, detail="Download not found or not pausable")


@app.post("/api/hf/downloads/{model_id:path}/resume")
async def resume_hf_download(model_id: str):
    """Resume a paused download"""
    success = hf_service.resume_download(model_id)
    if success:
        return APIResponse(success=True, message=f"Resumed download {model_id}")
    raise HTTPException(status_code=404, detail="Download not found or not resumable")


@app.delete("/api/hf/downloads")
async def clear_downloads():
    """Clear completed/failed/cancelled downloads"""
    count = hf_service.clear_completed()
    return APIResponse(success=True, message=f"Cleared {count} downloads")


@app.get("/api/hf/trending")
async def get_hf_trending(limit: int = Query(10, ge=1, le=50)):
    """Get trending HuggingFace models"""
    models = hf_service.get_trending(limit=limit)
    return {"models": models, "total": len(models)}


# ============================================
# Configuration Endpoints
# ============================================

@app.get("/api/config")
async def get_app_config():
    """Get application configuration"""
    return get_config()


@app.put("/api/config")
async def update_config(settings: dict):
    """Update application configuration"""
    try:
        from app.config import Settings
        new_settings = Settings(**settings)
        if save_config(new_settings):
            reload_config()
            return APIResponse(success=True, message="Configuration updated")
        raise HTTPException(status_code=500, detail="Failed to save configuration")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ============================================
# Docker Endpoints
# ============================================

@app.get("/api/docker/status")
async def get_docker_status():
    """Get Docker status"""
    if not docker_manager.is_available():
        error_msg = docker_manager.get_error() or "Docker not available"
        return {"available": False, "error": error_msg}
    
    return {
        "available": True,
        "info": docker_manager.get_docker_info(),
        "containers": [
            {
                "name": c.name,
                "status": c.status,
                "image": c.image.tags[0] if c.image.tags else "unknown"
            }
            for c in docker_manager.list_containers()
        ]
    }


if __name__ == "__main__":
    import uvicorn
    config = get_config()
    uvicorn.run(app, host=config.app.host, port=config.app.port)