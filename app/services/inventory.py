"""
Inventory service - Unified model inventory across all sources
"""

import os
import logging
from pathlib import Path
from typing import Optional, List, Dict
from datetime import datetime

from app.models import (
    InventoryItem, InventoryResponse, ModelSource, EngineType
)
from app.services.ollama_service import ollama_service
from app.config import get_config

logger = logging.getLogger(__name__)


class InventoryService:
    """Unified model inventory"""
    
    def __init__(self):
        self.config = get_config()
    
    def get_inventory(self, source: Optional[ModelSource] = None,
                      search: Optional[str] = None) -> InventoryResponse:
        """Get unified inventory from all sources"""
        items = []
        
        # Get Ollama models
        if source is None or source == ModelSource.OLLAMA:
            items.extend(self._get_ollama_models())
        
        # Get HuggingFace cache models
        if source is None or source == ModelSource.HUGGINGFACE:
            items.extend(self._get_hf_cache_models())
        
        # Get local models
        if source is None or source == ModelSource.LOCAL:
            items.extend(self._get_local_models())
        
        # Apply search filter
        if search:
            search_lower = search.lower()
            items = [item for item in items if search_lower in item.name.lower()
                    or search_lower in item.id.lower()
                    or any(search_lower in tag.lower() for tag in item.tags)]
        
        # Calculate totals
        total_size = sum(item.size_bytes for item in items)
        sources_count = {}
        for item in items:
            source_key = item.source.value
            sources_count[source_key] = sources_count.get(source_key, 0) + 1
        
        return InventoryResponse(
            items=items,
            total=len(items),
            total_size_bytes=total_size,
            total_size_human=self._format_size(total_size),
            sources=sources_count
        )
    
    def _get_ollama_models(self) -> List[InventoryItem]:
        """Get models from Ollama"""
        items = []
        
        try:
            models = ollama_service.list_models()
            for model in models:
                items.append(InventoryItem(
                    id=f"ollama:{model.name}",
                    name=model.name,
                    source=ModelSource.OLLAMA,
                    size_bytes=model.size,
                    size_human=model.size_human,
                    format="gguf",
                    engine=EngineType.OLLAMA,
                    tags=["ollama"],
                    metadata=model.details
                ))
        except Exception as e:
            logger.error(f"Error getting Ollama models: {e}")
        
        return items
    
    def _get_hf_cache_models(self) -> List[InventoryItem]:
        """Get models from HuggingFace cache"""
        items = []
        hf_cache = Path(self.config.paths.get("hf_cache", "/root/.cache/huggingface/hub"))
        
        if not hf_cache.exists():
            return items
        
        try:
            # HF cache structure: models--{org}--{model}
            for item in hf_cache.iterdir():
                if item.is_dir() and item.name.startswith("models--"):
                    # Parse org and model name
                    parts = item.name.replace("models--", "").split("--")
                    if len(parts) >= 2:
                        org = parts[0]
                        model_name = "--".join(parts[1:])
                        full_name = f"{org}/{model_name}"
                    else:
                        full_name = model_name = parts[0]
                    
                    # Calculate size
                    size = self._get_dir_size(item)
                    
                    # Check for snapshots
                    snapshots_dir = item / "snapshots"
                    format_info = None
                    if snapshots_dir.exists():
                        for snapshot in snapshots_dir.iterdir():
                            if snapshot.is_dir():
                                format_info = self._detect_format(snapshot)
                                break
                    
                    items.append(InventoryItem(
                        id=f"hf:{full_name}",
                        name=full_name,
                        source=ModelSource.HUGGINGFACE,
                        path=str(item),
                        size_bytes=size,
                        size_human=self._format_size(size),
                        format=format_info,
                        tags=["huggingface", "cached"]
                    ))
                    
        except Exception as e:
            logger.error(f"Error getting HF cache models: {e}")
        
        return items
    
    def _get_local_models(self) -> List[InventoryItem]:
        """Get models from local directories"""
        items = []
        
        for scan_dir in self.config.inventory.scan_dirs:
            dir_path = Path(scan_dir)
            if not dir_path.exists():
                continue
            
            try:
                for item in dir_path.iterdir():
                    if item.is_dir():
                        # Check if it looks like a model directory
                        if self._is_model_dir(item):
                            size = self._get_dir_size(item)
                            format_info = self._detect_format(item)
                            quantization = self._detect_quantization(item)
                            
                            items.append(InventoryItem(
                                id=f"local:{item.name}",
                                name=item.name,
                                source=ModelSource.LOCAL,
                                path=str(item),
                                size_bytes=size,
                                size_human=self._format_size(size),
                                format=format_info,
                                quantization=quantization,
                                tags=["local"]
                            ))
                            
            except Exception as e:
                logger.error(f"Error scanning directory {scan_dir}: {e}")
        
        return items
    
    def _is_model_dir(self, path: Path) -> bool:
        """Check if directory looks like a model"""
        model_extensions = {".safetensors", ".bin", ".onnx", ".gguf", ".pt", ".pth"}
        
        try:
            for item in path.iterdir():
                if item.is_file() and item.suffix.lower() in model_extensions:
                    return True
        except Exception:
            pass
        
        return False
    
    def _detect_format(self, path: Path) -> Optional[str]:
        """Detect model format from files"""
        try:
            for item in path.rglob("*"):
                if item.is_file():
                    suffix = item.suffix.lower()
                    if suffix == ".safetensors":
                        return "safetensors"
                    elif suffix == ".gguf":
                        return "gguf"
                    elif suffix == ".onnx":
                        return "onnx"
                    elif suffix in [".pt", ".pth"]:
                        return "pytorch"
                    elif suffix == ".bin":
                        return "pytorch"
        except Exception:
            pass
        
        return None
    
    def _detect_quantization(self, path: Path) -> Optional[str]:
        """Detect quantization from path"""
        path_str = str(path).lower()
        
        quantizations = [
            ("nvfp4", "NVFP4"),
            ("fp4", "FP4"),
            ("int4", "INT4"),
            ("int8", "INT8"),
            ("fp8", "FP8"),
            ("awq", "AWQ"),
            ("gptq", "GPTQ"),
            ("gguf", "GGUF"),
            ("exl2", "EXL2"),
            ("fp16", "FP16"),
            ("bf16", "BF16"),
        ]
        
        for pattern, name in quantizations:
            if pattern in path_str:
                return name
        
        return None
    
    def _get_dir_size(self, path: Path) -> int:
        """Calculate directory size"""
        total = 0
        try:
            for item in path.rglob("*"):
                if item.is_file():
                    total += item.stat().st_size
        except Exception:
            pass
        return total
    
    def _format_size(self, size_bytes: int) -> str:
        """Format bytes to human readable string"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} PB"
    
    def delete_item(self, item_id: str) -> bool:
        """Delete an inventory item"""
        try:
            if item_id.startswith("ollama:"):
                model_name = item_id.replace("ollama:", "")
                return ollama_service.delete_model(model_name)
            
            elif item_id.startswith("local:"):
                # Find and delete local model
                items = self.get_inventory(source=ModelSource.LOCAL)
                for item in items.items:
                    if item.id == item_id and item.path:
                        import shutil
                        shutil.rmtree(item.path)
                        return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error deleting item {item_id}: {e}")
            return False
    
    def scan_directory(self, directory: str) -> bool:
        """Add a directory to scan list"""
        try:
            if directory not in self.config.inventory.scan_dirs:
                self.config.inventory.scan_dirs.append(directory)
                from app.config import save_config
                return save_config(self.config)
            return True
        except Exception as e:
            logger.error(f"Error adding directory: {e}")
            return False
    
    def remove_directory(self, directory: str) -> bool:
        """Remove a directory from scan list"""
        try:
            if directory in self.config.inventory.scan_dirs:
                self.config.inventory.scan_dirs.remove(directory)
                from app.config import save_config
                return save_config(self.config)
            return True
        except Exception as e:
            logger.error(f"Error removing directory: {e}")
            return False


# Singleton instance
inventory_service = InventoryService()