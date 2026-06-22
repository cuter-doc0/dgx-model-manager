"""
Ollama service - Manage Ollama models and pull progress
"""

import logging
import httpx
from typing import Optional, List, Dict
from datetime import datetime

from app.models import OllamaModel, OllamaPullRequest, OllamaPullStatus
from app.config import get_config

logger = logging.getLogger(__name__)


class OllamaService:
    """Manage Ollama models"""
    
    def __init__(self):
        self.config = get_config()
        self.base_url = self.config.services.ollama_base
    
    def _get_client(self) -> httpx.Client:
        """Get HTTP client"""
        return httpx.Client(timeout=30)
    
    def is_available(self) -> bool:
        """Check if Ollama is available"""
        try:
            with self._get_client() as client:
                response = client.get(f"{self.base_url}/api/tags")
                return response.status_code == 200
        except Exception:
            return False
    
    def list_models(self) -> List[OllamaModel]:
        """List all Ollama models"""
        try:
            with self._get_client() as client:
                response = client.get(f"{self.base_url}/api/tags")
                response.raise_for_status()
                
                data = response.json()
                models = []
                
                for model in data.get("models", []):
                    models.append(OllamaModel(
                        name=model.get("name", ""),
                        model_id=model.get("name", ""),
                        size=model.get("size", 0),
                        size_human=self._format_size(model.get("size", 0)),
                        digest=model.get("digest", ""),
                        modified_at=datetime.fromisoformat(model.get("modified_at", datetime.now().isoformat()).replace("Z", "+00:00")),
                        details=model.get("details", {})
                    ))
                
                return models
                
        except Exception as e:
            logger.error(f"Error listing Ollama models: {e}")
            return []
    
    def get_model(self, name: str) -> Optional[OllamaModel]:
        """Get specific model info"""
        try:
            with self._get_client() as client:
                response = client.post(
                    f"{self.base_url}/api/show",
                    json={"name": name}
                )
                response.raise_for_status()
                
                data = response.json()
                return OllamaModel(
                    name=name,
                    model_id=name,
                    size=data.get("size", 0),
                    size_human=self._format_size(data.get("size", 0)),
                    digest=data.get("digest", ""),
                    modified_at=datetime.now(),
                    details=data.get("details", {})
                )
                
        except Exception as e:
            logger.error(f"Error getting Ollama model {name}: {e}")
            return None
    
    def pull_model(self, model: str, tag: str = "latest") -> bool:
        """Pull a model from Ollama registry"""
        try:
            with self._get_client() as client:
                response = client.post(
                    f"{self.base_url}/api/pull",
                    json={"name": model, "tag": tag}
                )
                response.raise_for_status()
                logger.info(f"Started pulling {model}:{tag}")
                return True
        except Exception as e:
            logger.error(f"Error pulling model {model}: {e}")
            return False
    
    def pull_model_stream(self, model: str, tag: str = "latest"):
        """Pull a model with streaming progress"""
        try:
            with httpx.Client(timeout=None) as client:
                with client.stream(
                    "POST",
                    f"{self.base_url}/api/pull",
                    json={"name": model, "tag": tag}
                ) as response:
                    for line in response.iter_lines():
                        if line:
                            yield line
        except Exception as e:
            logger.error(f"Error streaming pull for {model}: {e}")
    
    def delete_model(self, name: str) -> bool:
        """Delete a model"""
        try:
            with self._get_client() as client:
                response = client.delete(
                    f"{self.base_url}/api/delete",
                    json={"name": name}
                )
                response.raise_for_status()
                logger.info(f"Deleted model {name}")
                return True
        except Exception as e:
            logger.error(f"Error deleting model {name}: {e}")
            return False
    
    def copy_model(self, source: str, destination: str) -> bool:
        """Copy a model"""
        try:
            with self._get_client() as client:
                response = client.post(
                    f"{self.base_url}/api/copy",
                    json={"source": source, "destination": destination}
                )
                response.raise_for_status()
                return True
        except Exception as e:
            logger.error(f"Error copying model: {e}")
            return False
    
    def generate(self, model: str, prompt: str, **kwargs) -> Optional[str]:
        """Generate text with a model"""
        try:
            with self._get_client() as client:
                response = client.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": model,
                        "prompt": prompt,
                        "stream": False,
                        **kwargs
                    }
                )
                response.raise_for_status()
                return response.json().get("response")
        except Exception as e:
            logger.error(f"Error generating with model {model}: {e}")
            return None
    
    def chat(self, model: str, messages: List[Dict], **kwargs) -> Optional[Dict]:
        """Chat with a model"""
        try:
            with self._get_client() as client:
                response = client.post(
                    f"{self.base_url}/api/chat",
                    json={
                        "model": model,
                        "messages": messages,
                        "stream": False,
                        **kwargs
                    }
                )
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.error(f"Error chatting with model {model}: {e}")
            return None
    
    def _format_size(self, size_bytes: int) -> str:
        """Format bytes to human readable string"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} PB"


# Singleton instance
ollama_service = OllamaService()