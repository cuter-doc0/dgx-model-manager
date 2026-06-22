"""
Docker container management service
"""

import os
import logging
import docker
from typing import Optional, List, Dict
from docker.models.containers import Container

logger = logging.getLogger(__name__)


class DockerManager:
    """Manage Docker containers for inference engines"""
    
    def __init__(self):
        self.client = None
        self._error = None
        self._try_connect()
    
    def _try_connect(self):
        """Try to connect to Docker daemon"""
        try:
            # Check if socket exists
            socket_path = os.environ.get("DOCKER_HOST", "unix:///var/run/docker.sock")
            logger.info(f"Attempting Docker connection via: {socket_path}")
            
            self.client = docker.from_env()
            
            # Test the connection
            self.client.ping()
            logger.info("Docker client connected successfully")
            
        except FileNotFoundError as e:
            self._error = f"Docker socket not found: {e}"
            logger.warning(self._error)
            self.client = None
        except PermissionError as e:
            self._error = f"Permission denied accessing Docker: {e}"
            logger.warning(self._error)
            self.client = None
        except docker.errors.DockerException as e:
            self._error = f"Docker daemon not accessible: {e}"
            logger.warning(self._error)
            self.client = None
        except Exception as e:
            self._error = f"Unexpected Docker error: {type(e).__name__}: {e}"
            logger.warning(self._error)
            self.client = None
    
    def is_available(self) -> bool:
        """Check if Docker is available"""
        return self.client is not None
    
    def get_error(self) -> Optional[str]:
        """Get the last connection error"""
        return self._error
    
    def get_container(self, name: str) -> Optional[Container]:
        """Get container by name"""
        try:
            return self.client.containers.get(name)
        except docker.errors.NotFound:
            return None
        except Exception as e:
            logger.error(f"Error getting container {name}: {e}")
            return None
    
    def list_containers(self, all: bool = True) -> List[Container]:
        """List all containers"""
        try:
            return self.client.containers.list(all=all)
        except Exception as e:
            logger.error(f"Error listing containers: {e}")
            return []
    
    def get_container_status(self, name: str) -> dict:
        """Get container status"""
        container = self.get_container(name)
        if not container:
            return {"exists": False, "status": "not_found"}
        
        return {
            "exists": True,
            "status": container.status,
            "id": container.short_id,
            "name": container.name,
            "image": container.image.tags[0] if container.image.tags else "unknown",
            "ports": container.ports,
            "created": container.attrs.get("Created"),
            "started": container.attrs.get("State", {}).get("StartedAt"),
        }
    
    def start_container(self, name: str) -> bool:
        """Start a container"""
        try:
            container = self.get_container(name)
            if container:
                if container.status == "running":
                    logger.info(f"Container {name} already running")
                    return True
                container.start()
                logger.info(f"Started container {name}")
                return True
            logger.warning(f"Container {name} not found")
            return False
        except Exception as e:
            logger.error(f"Error starting container {name}: {e}")
            return False
    
    def stop_container(self, name: str, timeout: int = 10) -> bool:
        """Stop a container"""
        try:
            container = self.get_container(name)
            if container:
                if container.status != "running":
                    logger.info(f"Container {name} not running")
                    return True
                container.stop(timeout=timeout)
                logger.info(f"Stopped container {name}")
                return True
            logger.warning(f"Container {name} not found")
            return False
        except Exception as e:
            logger.error(f"Error stopping container {name}: {e}")
            return False
    
    def restart_container(self, name: str, timeout: int = 10) -> bool:
        """Restart a container"""
        try:
            container = self.get_container(name)
            if container:
                container.restart(timeout=timeout)
                logger.info(f"Restarted container {name}")
                return True
            logger.warning(f"Container {name} not found")
            return False
        except Exception as e:
            logger.error(f"Error restarting container {name}: {e}")
            return False
    
    def remove_container(self, name: str, force: bool = False) -> bool:
        """Remove a container"""
        try:
            container = self.get_container(name)
            if container:
                container.remove(force=force)
                logger.info(f"Removed container {name}")
                return True
            logger.warning(f"Container {name} not found")
            return False
        except Exception as e:
            logger.error(f"Error removing container {name}: {e}")
            return False
    
    def get_container_logs(self, name: str, tail: int = 100) -> str:
        """Get container logs"""
        try:
            container = self.get_container(name)
            if container:
                logs = container.logs(tail=tail, timestamps=True)
                return logs.decode("utf-8", errors="replace")
            return ""
        except Exception as e:
            logger.error(f"Error getting logs for {name}: {e}")
            return ""
    
    def get_docker_info(self) -> dict:
        """Get Docker system information"""
        try:
            info = self.client.info()
            return {
                "version": info.get("ServerVersion", "unknown"),
                "containers_running": info.get("ContainersRunning", 0),
                "containers_stopped": info.get("ContainersStopped", 0),
                "containers_paused": info.get("ContainersPaused", 0),
                "images": info.get("Images", 0),
                "driver": info.get("Driver", "unknown"),
                "kernel_version": info.get("KernelVersion", "unknown"),
                "os": info.get("OperatingSystem", "unknown"),
                "architecture": info.get("Architecture", "unknown"),
            }
        except Exception as e:
            logger.error(f"Error getting Docker info: {e}")
            return {"error": str(e)}
    
    def run_container(
        self,
        image: str,
        name: str,
        ports: Optional[Dict] = None,
        volumes: Optional[Dict] = None,
        environment: Optional[Dict] = None,
        command: Optional[str] = None,
        gpu: bool = True,
        network: Optional[str] = None,
        detach: bool = True
    ) -> Optional[Container]:
        """Run a new container"""
        try:
            # Remove existing container if present
            existing = self.get_container(name)
            if existing:
                existing.remove(force=True)
            
            kwargs = {
                "image": image,
                "name": name,
                "detach": detach,
            }
            
            if network:
                kwargs["network"] = network
            
            if ports:
                kwargs["ports"] = ports
            if volumes:
                kwargs["volumes"] = volumes
            if environment:
                kwargs["environment"] = environment
            if command:
                kwargs["command"] = command
            
            if gpu:
                kwargs["device_requests"] = [
                    docker.types.DeviceRequest(
                        capabilities=[["gpu"]],
                        count=-1  # All GPUs
                    )
                ]
            
            container = self.client.containers.run(**kwargs)
            logger.info(f"Started container {name} from image {image}")
            return container
            
        except Exception as e:
            logger.error(f"Error running container {name}: {e}")
            return None


# Singleton instance
docker_manager = DockerManager()