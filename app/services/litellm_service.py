"""
LiteLLM service - Manage LiteLLM routing and configuration
"""

import logging
import httpx
import yaml
from pathlib import Path
from typing import Optional, List, Dict

from app.models import LiteLLMRoute, LiteLLMConfig
from app.config import get_config

logger = logging.getLogger(__name__)


class LiteLLMService:
    """Manage LiteLLM routing"""
    
    def __init__(self):
        self.config = get_config()
        self.base_url = self.config.services.litellm_base
        self.config_path = Path(self.config.paths.get("litellm_config", "/app/config/litellm_config.yaml"))
    
    def _get_client(self) -> httpx.Client:
        """Get HTTP client"""
        return httpx.Client(timeout=10)
    
    def is_available(self) -> bool:
        """Check if LiteLLM is available"""
        try:
            with self._get_client() as client:
                response = client.get(f"{self.base_url}/health")
                return response.status_code == 200
        except Exception:
            return False
    
    def get_models(self) -> List[Dict]:
        """Get list of available models"""
        try:
            with self._get_client() as client:
                response = client.get(f"{self.base_url}/v1/models")
                response.raise_for_status()
                return response.json().get("data", [])
        except Exception as e:
            logger.error(f"Error getting LiteLLM models: {e}")
            return []
    
    def get_config(self) -> Optional[LiteLLMConfig]:
        """Get LiteLLM configuration"""
        try:
            if not self.config_path.exists():
                return None
            
            with open(self.config_path, "r") as f:
                data = yaml.safe_load(f)
            
            model_list = []
            for model in data.get("model_list", []):
                model_list.append(LiteLLMRoute(
                    model_name=model.get("model_name", ""),
                    litellm_params=model.get("litellm_params", {}),
                    status="active"
                ))
            
            return LiteLLMConfig(
                model_list=model_list,
                litellm_settings=data.get("litellm_settings", {}),
                general_settings=data.get("general_settings", {})
            )
            
        except Exception as e:
            logger.error(f"Error getting LiteLLM config: {e}")
            return None
    
    def save_config(self, config: LiteLLMConfig) -> bool:
        """Save LiteLLM configuration"""
        try:
            data = {
                "model_list": [
                    {
                        "model_name": route.model_name,
                        "litellm_params": route.litellm_params
                    }
                    for route in config.model_list
                ],
                "litellm_settings": config.litellm_settings,
                "general_settings": config.general_settings
            }
            
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, "w") as f:
                yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving LiteLLM config: {e}")
            return False
    
    def apply_wildcard(self, engine: str = "ollama", api_base: Optional[str] = None) -> bool:
        """Apply wildcard routing for an engine"""
        try:
            config = self.get_config()
            if not config:
                config = LiteLLMConfig(model_list=[], litellm_settings={}, general_settings={})
            
            # Check if wildcard already exists
            wildcard_name = f"{engine}/*"
            for route in config.model_list:
                if route.model_name == wildcard_name:
                    logger.info(f"Wildcard {wildcard_name} already exists")
                    return True
            
            # Determine API base
            if not api_base:
                api_base = getattr(self.config.services, f"{engine}_base", f"http://localhost:11434")
            
            # Add wildcard route
            new_route = LiteLLMRoute(
                model_name=wildcard_name,
                litellm_params={
                    "model": f"{engine}/*",
                    "api_base": api_base
                },
                status="active"
            )
            
            config.model_list.append(new_route)
            
            if self.save_config(config):
                logger.info(f"Applied wildcard routing for {engine}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error applying wildcard: {e}")
            return False
    
    def add_route(self, model_name: str, litellm_params: Dict) -> bool:
        """Add a new route"""
        try:
            config = self.get_config()
            if not config:
                config = LiteLLMConfig(model_list=[], litellm_settings={}, general_settings={})
            
            new_route = LiteLLMRoute(
                model_name=model_name,
                litellm_params=litellm_params,
                status="active"
            )
            
            config.model_list.append(new_route)
            return self.save_config(config)
            
        except Exception as e:
            logger.error(f"Error adding route: {e}")
            return False
    
    def remove_route(self, model_name: str) -> bool:
        """Remove a route"""
        try:
            config = self.get_config()
            if not config:
                return False
            
            config.model_list = [r for r in config.model_list if r.model_name != model_name]
            return self.save_config(config)
            
        except Exception as e:
            logger.error(f"Error removing route: {e}")
            return False
    
    def test_connection(self) -> Dict:
        """Test connection to LiteLLM"""
        try:
            with self._get_client() as client:
                response = client.get(f"{self.base_url}/health")
                
                if response.status_code == 200:
                    return {
                        "status": "connected",
                        "latency_ms": response.elapsed.total_seconds() * 1000,
                        "data": response.json()
                    }
                else:
                    return {
                        "status": "error",
                        "error": f"HTTP {response.status_code}"
                    }
                    
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def get_health(self) -> Dict:
        """Get LiteLLM health status"""
        try:
            with self._get_client() as client:
                response = client.get(f"{self.base_url}/health")
                response.raise_for_status()
                return response.json()
        except Exception as e:
            return {"status": "error", "error": str(e)}


# Singleton instance
litellm_service = LiteLLMService()