"""
HuggingFace service - Search and download models from HF Hub
"""

import logging
from datetime import datetime
from typing import Optional, List
from fastapi import HTTPException
from huggingface_hub import HfApi, snapshot_download
from app.models import HFModel, HFSearchRequest, HFSearchResponse, HFDownloadStatus
from app.config import get_config

logger = logging.getLogger(__name__)


class HFService:
    """HuggingFace Hub integration"""
    
    def __init__(self):
        self.config = get_config()
        self.api = HfApi()
        self._download_tasks: dict[str, HFDownloadStatus] = {}
    
    def search_models(self, query: str, limit: int = 20,
                      sort: Optional[str] = None,
                      pipeline_tag: Optional[str] = None,
                      author: Optional[str] = None) -> HFSearchResponse:
        """Search for models on HuggingFace Hub"""
        try:
            filter_kwargs = {}
            if author:
                filter_kwargs["author"] = author
            if pipeline_tag:
                filter_kwargs["pipeline_tag"] = pipeline_tag
            
            models = self.api.list_models(
                search=query,
                limit=limit,
                sort=sort,
                direction=-1 if sort else None,
                **filter_kwargs
            )
            
            hf_models = []
            for model in models:
                hf_models.append(HFModel(
                    id=model.id,
                    author=model.author,
                    model_id=model.id,
                    pipeline_tag=model.pipeline_tag,
                    tags=model.tags if model.tags else [],
                    downloads=model.downloads or 0,
                    likes=model.likes or 0,
                    last_modified=model.last_modified,
                    private=model.private,
                    gated=model.gated
                ))
            
            return HFSearchResponse(
                models=hf_models,
                total=len(hf_models),
                query=query
            )
            
        except Exception as e:
            logger.error(f"Error searching HF models: {e}")
            raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")
    
    def get_model_info(self, model_id: str) -> Optional[HFModel]:
        """Get detailed model information"""
        try:
            model_info = self.api.model_info(model_id)
            
            return HFModel(
                id=model_info.id,
                author=model_info.author,
                model_id=model_info.id,
                pipeline_tag=model_info.pipeline_tag,
                tags=model_info.tags if model_info.tags else [],
                downloads=model_info.downloads or 0,
                likes=model_info.likes or 0,
                last_modified=model_info.last_modified,
                private=model_info.private,
                gated=model_info.gated
            )
            
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            return None
    
    def get_model_files(self, model_id: str) -> List[str]:
        """Get list of files in a model repository"""
        try:
            files = self.api.list_repo_files(model_id)
            return files
        except Exception as e:
            logger.error(f"Error getting model files: {e}")
            return []
    
    def download_model(self, model_id: str, revision: Optional[str] = None,
                       local_dir: Optional[str] = None) -> bool:
        """Download a model from HuggingFace Hub"""
        try:
            # Initialize download status
            self._download_tasks[model_id] = HFDownloadStatus(
                model_id=model_id,
                status="downloading",
                progress=0,
                started_at=datetime.now()
            )
            
            # Build download kwargs
            download_kwargs = {
                "repo_id": model_id,
                "resume_download": True
            }
            
            if revision:
                download_kwargs["revision"] = revision
            
            if local_dir:
                download_kwargs["local_dir"] = local_dir
            else:
                # Download to models directory
                models_dir = self.config.paths.get("models", "/models")
                download_kwargs["local_dir"] = f"{models_dir}/{model_id.replace('/', '_')}"
            
            # Start download
            snapshot_download(**download_kwargs)
            
            # Update status
            self._download_tasks[model_id].status = "completed"
            self._download_tasks[model_id].progress = 100
            self._download_tasks[model_id].completed_at = datetime.now()
            
            logger.info(f"Downloaded model {model_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error downloading model {model_id}: {e}")
            if model_id in self._download_tasks:
                self._download_tasks[model_id].status = "failed"
                self._download_tasks[model_id].error = str(e)
            return False
    
    def get_download_status(self, model_id: str) -> Optional[HFDownloadStatus]:
        """Get download status"""
        return self._download_tasks.get(model_id)
    
    def get_trending(self, limit: int = 10) -> List[HFModel]:
        """Get trending models"""
        try:
            models = self.api.list_models(
                sort="downloads",
                direction=-1,
                limit=limit
            )
            
            return [
                HFModel(
                    id=m.id,
                    author=m.author,
                    model_id=m.id,
                    pipeline_tag=m.pipeline_tag,
                    tags=m.tags if m.tags else [],
                    downloads=m.downloads or 0,
                    likes=m.likes or 0,
                    last_modified=m.last_modified,
                    private=m.private,
                    gated=m.gated
                )
                for m in models
            ]
            
        except Exception as e:
            logger.error(f"Error getting trending models: {e}")
            return []


# Singleton instance
hf_service = HFService()