"""
HuggingFace service - Search and download models from HF Hub
"""

import os
import logging
import threading
from datetime import datetime
from typing import Optional, List, Dict, Callable
from huggingface_hub import HfApi, snapshot_download, hf_hub_download
from huggingface_hub.utils import HfHubHTTPError
from app.models import HFModel, HFSearchRequest, HFSearchResponse, HFDownloadStatus
from app.config import get_config

logger = logging.getLogger(__name__)


class DownloadTask:
    """Track a single download task"""
    
    def __init__(self, model_id: str, local_dir: str, revision: Optional[str] = None):
        self.model_id = model_id
        self.local_dir = local_dir
        self.revision = revision
        self.status = "pending"  # pending, downloading, completed, failed, cancelled
        self.progress = 0.0
        self.files_total = 0
        self.files_done = 0
        self.current_file = ""
        self.bytes_total = 0
        self.bytes_done = 0
        self.speed = ""
        self.error = None
        self.started_at = datetime.now()
        self.completed_at = None
        self.cancelled = False
        self._thread = None
        self._lock = threading.Lock()
    
    def to_status(self) -> HFDownloadStatus:
        """Convert to HFDownloadStatus model"""
        with self._lock:
            return HFDownloadStatus(
                model_id=self.model_id,
                status=self.status,
                progress=self.progress,
                speed=self.speed,
                error=self.error,
                started_at=self.started_at,
                completed_at=self.completed_at,
                files_total=self.files_total,
                files_done=self.files_done,
                current_file=self.current_file,
                bytes_total=self.bytes_total,
                bytes_done=self.bytes_done
            )
    
    def update_progress(self, progress: float, files_total: int = 0, files_done: int = 0,
                       current_file: str = "", bytes_total: int = 0, bytes_done: int = 0,
                       speed: str = ""):
        """Update download progress"""
        with self._lock:
            self.progress = progress
            if files_total > 0:
                self.files_total = files_total
            if files_done > 0:
                self.files_done = files_done
            if current_file:
                self.current_file = current_file
            if bytes_total > 0:
                self.bytes_total = bytes_total
            if bytes_done > 0:
                self.bytes_done = bytes_done
            if speed:
                self.speed = speed
    
    def mark_completed(self):
        """Mark download as completed"""
        with self._lock:
            self.status = "completed"
            self.progress = 100.0
            self.completed_at = datetime.now()
    
    def mark_failed(self, error: str):
        """Mark download as failed"""
        with self._lock:
            self.status = "failed"
            self.error = error
            self.completed_at = datetime.now()
    
    def mark_cancelled(self):
        """Mark download as cancelled"""
        with self._lock:
            self.status = "cancelled"
            self.cancelled = True
            self.completed_at = datetime.now()


class ProgressCallback:
    """HuggingFace download progress callback"""
    
    def __init__(self, task: DownloadTask):
        self.task = task
        self._file_progress: Dict[str, dict] = {}
        self._start_time = datetime.now()
    
    def __call__(self, filename: str, size: int, loaded: int) -> None:
        """Called during file download"""
        try:
            with self.task._lock:
                # Track per-file progress
                if filename not in self._file_progress:
                    self._file_progress[filename] = {"size": size, "loaded": loaded}
                else:
                    self._file_progress[filename]["loaded"] = loaded
                    if size > 0:
                        self._file_progress[filename]["size"] = size
                
                # Calculate overall progress
                total_bytes = sum(p["size"] for p in self._file_progress.values())
                done_bytes = sum(p["loaded"] for p in self._file_progress.values())
                
                if total_bytes > 0:
                    self.task.progress = (done_bytes / total_bytes) * 100
                    self.task.bytes_total = total_bytes
                    self.task.bytes_done = done_bytes
                
                # Update file counts
                self.task.files_total = len(self._file_progress)
                self.task.files_done = sum(1 for p in self._file_progress.values() if p["loaded"] >= p["size"] and p["size"] > 0)
                self.task.current_file = filename
                
                # Calculate speed
                elapsed = (datetime.now() - self._start_time).total_seconds()
                if elapsed > 0 and done_bytes > 0:
                    speed_bytes = done_bytes / elapsed
                    if speed_bytes > 1024 * 1024:
                        self.task.speed = f"{speed_bytes / (1024 * 1024):.1f} MB/s"
                    elif speed_bytes > 1024:
                        self.task.speed = f"{speed_bytes / 1024:.1f} KB/s"
                    else:
                        self.task.speed = f"{speed_bytes:.0f} B/s"
                
                # ETA
                if self.task.speed and speed_bytes > 0:
                    remaining = total_bytes - done_bytes
                    eta_seconds = remaining / speed_bytes
                    if eta_seconds > 3600:
                        self.task.eta = f"{int(eta_seconds // 3600)}h {int((eta_seconds % 3600) // 60)}m"
                    elif eta_seconds > 60:
                        self.task.eta = f"{int(eta_seconds // 60)}m {int(eta_seconds % 60)}s"
                    else:
                        self.task.eta = f"{int(eta_seconds)}s"
        except Exception as e:
            logger.debug(f"Progress callback error: {e}")


class HFService:
    """HuggingFace Hub integration"""
    
    def __init__(self):
        self.config = get_config()
        self.api = HfApi()
        self._download_tasks: Dict[str, DownloadTask] = {}
        self._download_lock = threading.Lock()
    
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
            raise Exception(f"Search failed: {str(e)}")
    
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
    
    def start_download(self, model_id: str, revision: Optional[str] = None,
                       local_dir: Optional[str] = None) -> DownloadTask:
        """Start a background download"""
        with self._download_lock:
            # Check if already downloading
            if model_id in self._download_tasks:
                existing = self._download_tasks[model_id]
                if existing.status in ("pending", "downloading"):
                    return existing
            
            # Determine local directory
            if not local_dir:
                models_dir = self.config.paths.get("models", "/models")
                local_dir = f"{models_dir}/{model_id.replace('/', '_')}"
            
            # Create task
            task = DownloadTask(model_id, local_dir, revision)
            task.status = "downloading"
            self._download_tasks[model_id] = task
            
            # Start background thread
            thread = threading.Thread(
                target=self._download_worker,
                args=(task,),
                daemon=True
            )
            task._thread = thread
            thread.start()
            
            return task
    
    def _download_worker(self, task: DownloadTask):
        """Background download worker"""
        try:
            logger.info(f"Starting download of {task.model_id} to {task.local_dir}")
            
            # Create directory
            os.makedirs(task.local_dir, exist_ok=True)
            
            # Progress callback
            progress_callback = ProgressCallback(task)
            
            # Download
            download_kwargs = {
                "repo_id": task.model_id,
                "local_dir": task.local_dir,
                "resume_download": True,
            }
            
            if task.revision:
                download_kwargs["revision"] = task.revision
            
            # Use tqdm_class for progress tracking
            snapshot_download(**download_kwargs)
            
            if not task.cancelled:
                task.mark_completed()
                logger.info(f"Download completed: {task.model_id}")
            
        except Exception as e:
            if not task.cancelled:
                error_msg = str(e)
                if "cancelled" in error_msg.lower():
                    task.mark_cancelled()
                else:
                    task.mark_failed(error_msg)
                    logger.error(f"Download failed: {task.model_id}: {e}")
    
    def cancel_download(self, model_id: str) -> bool:
        """Cancel an active download"""
        with self._download_lock:
            task = self._download_tasks.get(model_id)
            if not task:
                return False
            
            if task.status not in ("pending", "downloading"):
                return False
            
            task.mark_cancelled()
            
            # Try to remove partial download
            try:
                if task.local_dir and os.path.exists(task.local_dir):
                    import shutil
                    shutil.rmtree(task.local_dir, ignore_errors=True)
                    logger.info(f"Removed partial download: {task.local_dir}")
            except Exception as e:
                logger.warning(f"Failed to remove partial download: {e}")
            
            return True
    
    def pause_download(self, model_id: str) -> bool:
        """Pause a download (not fully supported by HF, mark as paused)"""
        with self._download_lock:
            task = self._download_tasks.get(model_id)
            if not task or task.status != "downloading":
                return False
            task.status = "paused"
            return True
    
    def resume_download(self, model_id: str) -> bool:
        """Resume a paused download"""
        with self._download_lock:
            task = self._download_tasks.get(model_id)
            if not task or task.status != "paused":
                return False
            task.status = "downloading"
            # Restart the worker thread
            thread = threading.Thread(
                target=self._download_worker,
                args=(task,),
                daemon=True
            )
            task._thread = thread
            thread.start()
            return True
    
    def get_download_status(self, model_id: str) -> Optional[HFDownloadStatus]:
        """Get download status"""
        task = self._download_tasks.get(model_id)
        if task:
            return task.to_status()
        return None
    
    def get_all_downloads(self) -> List[HFDownloadStatus]:
        """Get all download statuses"""
        with self._download_lock:
            return [task.to_status() for task in self._download_tasks.values()]
    
    def get_active_downloads(self) -> List[HFDownloadStatus]:
        """Get active (pending/downloading/paused) downloads"""
        with self._download_lock:
            return [
                task.to_status() for task in self._download_tasks.values()
                if task.status in ("pending", "downloading", "paused")
            ]
    
    def clear_completed(self) -> int:
        """Clear completed/failed/cancelled downloads from tracking"""
        with self._download_lock:
            to_remove = [
                model_id for model_id, task in self._download_tasks.items()
                if task.status in ("completed", "failed", "cancelled")
            ]
            for model_id in to_remove:
                del self._download_tasks[model_id]
            return len(to_remove)
    
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