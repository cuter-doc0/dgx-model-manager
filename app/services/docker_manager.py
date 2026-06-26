"""
Docker container management service
Uses Docker CLI via subprocess for maximum compatibility
"""

import os
import json
import logging
import subprocess
from typing import Optional, List, Dict

logger = logging.getLogger(__name__)


class Container:
    """Lightweight container representation"""
    def __init__(self, data: dict):
        self._data = data
        self.short_id = data.get("Id", "")[:12]
        self.name = data.get("Names", [""])[0].lstrip("/") if data.get("Names") else ""
        self.status = data.get("Status", "unknown")
        self.image = self._parse_image(data)
        self.ports = self._parse_ports(data)
        self.attrs = data
    
    def _parse_image(self, data: dict) -> str:
        """Parse image from container data"""
        tags = data.get("Image", "unknown")
        return tags.split(":")[0] if ":" in tags else tags
    
    def _parse_ports(self, data: dict) -> dict:
        """Parse ports from container data"""
        ports = {}
        ports_data = data.get("Ports", [])
        for port in ports_data:
            if port.get("PublicPort"):
                key = f"{port.get('PrivatePort')}/{port.get('Type', 'tcp')}"
                ports[key] = port.get("PublicPort")
        return ports


class DockerManager:
    """Manage Docker containers using CLI"""
    
    def __init__(self):
        self.client = None
        self._error = None
        self._docker_cmd = self._find_docker_cmd()
        self._try_connect()
    
    def _find_docker_cmd(self) -> Optional[str]:
        """Find Docker CLI command"""
        # Try common locations
        candidates = [
            "/usr/bin/docker",
            "/usr/local/bin/docker",
            "docker"
        ]
        for cmd in candidates:
            try:
                result = subprocess.run(
                    [cmd, "--version"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    logger.info(f"Found Docker CLI at: {cmd}")
                    return cmd
            except (FileNotFoundError, subprocess.TimeoutExpired):
                continue
        return None
    
    def _try_connect(self):
        """Try to connect to Docker daemon"""
        if not self._docker_cmd:
            self._error = "Docker CLI not found in container"
            logger.warning(self._error)
            self.client = None
            return
        
        # Check socket exists
        socket_path = "/var/run/docker.sock"
        if not os.path.exists(socket_path):
            self._error = f"Docker socket not found at {socket_path}"
            logger.warning(self._error)
            self.client = None
            return
        
        # Test connection
        try:
            result = subprocess.run(
                [self._docker_cmd, "version", "--format", "{{.Server.Version}}"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                version = result.stdout.strip()
                logger.info(f"Docker daemon connected (version {version})")
                self.client = self  # Self is the client
            else:
                self._error = f"Docker daemon not accessible: {result.stderr.strip()}"
                logger.warning(self._error)
                self.client = None
        except subprocess.TimeoutExpired:
            self._error = "Docker daemon connection timed out"
            logger.warning(self._error)
            self.client = None
        except Exception as e:
            self._error = f"Docker connection error: {type(e).__name__}: {e}"
            logger.warning(self._error)
            self.client = None
    
    def is_available(self) -> bool:
        """Check if Docker is available"""
        return self.client is not None
    
    def get_error(self) -> Optional[str]:
        """Get the last connection error"""
        return self._error
    
    def _run(self, args: List[str], input_data: Optional[str] = None, timeout: int = 60) -> subprocess.CompletedProcess:
        """Run Docker CLI command"""
        return subprocess.run(
            [self._docker_cmd] + args,
            capture_output=True,
            text=True,
            input=input_data,
            timeout=timeout
        )
    
    def get_container(self, name: str) -> Optional[Container]:
        """Get container by name"""
        try:
            result = self._run([
                "inspect", name,
                "--format", "{{json .}}"
            ])
            if result.returncode == 0 and result.stdout.strip():
                data = json.loads(result.stdout.strip())
                return Container(data)
            return None
        except Exception as e:
            logger.error(f"Error getting container {name}: {e}")
            return None
    
    def list_containers(self, all: bool = True) -> List[Container]:
        """List all containers"""
        try:
            args = ["ps", "--format", "{{json .}}"]
            if all:
                args.insert(1, "-a")
            
            result = self._run(args)
            containers = []
            
            for line in result.stdout.strip().split("\n"):
                if line.strip():
                    try:
                        data = json.loads(line)
                        containers.append(Container(data))
                    except json.JSONDecodeError:
                        continue
            
            return containers
        except Exception as e:
            logger.error(f"Error listing containers: {e}")
            return []
    
    def get_container_status(self, name: str) -> dict:
        """Get container status"""
        container = self.get_container(name)
        if not container:
            return {"exists": False, "status": "not_found"}
        
        # Check if running
        status_text = container.status.lower()
        if "up" in status_text:
            status = "running"
        elif "exited" in status_text:
            status = "exited"
        elif "restarting" in status_text:
            status = "restarting"
        elif "paused" in status_text:
            status = "paused"
        elif "dead" in status_text:
            status = "dead"
        elif "created" in status_text:
            status = "created"
        else:
            status = "unknown"
        
        return {
            "exists": True,
            "status": status,
            "id": container.short_id,
            "name": container.name,
            "image": container.image,
            "ports": container.ports,
        }
    
    def start_container(self, name: str) -> bool:
        """Start a container"""
        try:
            result = self._run(["start", name])
            if result.returncode == 0:
                logger.info(f"Started container {name}")
                return True
            logger.warning(f"Failed to start {name}: {result.stderr.strip()}")
            return False
        except Exception as e:
            logger.error(f"Error starting container {name}: {e}")
            return False
    
    def stop_container(self, name: str, timeout: int = 10) -> bool:
        """Stop a container"""
        try:
            result = self._run(["stop", "-t", str(timeout), name])
            if result.returncode == 0:
                logger.info(f"Stopped container {name}")
                return True
            logger.warning(f"Failed to stop {name}: {result.stderr.strip()}")
            return False
        except Exception as e:
            logger.error(f"Error stopping container {name}: {e}")
            return False
    
    def restart_container(self, name: str, timeout: int = 10) -> bool:
        """Restart a container"""
        try:
            result = self._run(["restart", "-t", str(timeout), name])
            if result.returncode == 0:
                logger.info(f"Restarted container {name}")
                return True
            logger.warning(f"Failed to restart {name}: {result.stderr.strip()}")
            return False
        except Exception as e:
            logger.error(f"Error restarting container {name}: {e}")
            return False
    
    def remove_container(self, name: str, force: bool = False) -> bool:
        """Remove a container"""
        try:
            args = ["rm"]
            if force:
                args.append("-f")
            args.append(name)
            result = self._run(args)
            if result.returncode == 0:
                logger.info(f"Removed container {name}")
                return True
            logger.warning(f"Failed to remove {name}: {result.stderr.strip()}")
            return False
        except Exception as e:
            logger.error(f"Error removing container {name}: {e}")
            return False
    
    def get_container_logs(self, name: str, tail: int = 100) -> str:
        """Get container logs"""
        try:
            result = self._run(["logs", "--tail", str(tail), name])
            if result.returncode == 0:
                return result.stdout + result.stderr
            return ""
        except Exception as e:
            logger.error(f"Error getting logs for {name}: {e}")
            return ""
    
    def get_docker_info(self) -> dict:
        """Get Docker system information"""
        try:
            result = self._run(["version", "--format", "{{json .}}"])
            if result.returncode == 0:
                version_data = json.loads(result.stdout)
                return {
                    "version": version_data.get("Server", {}).get("Version", "unknown"),
                    "api_version": version_data.get("Server", {}).get("ApiVersion", "unknown"),
                }
            return {"error": "Failed to get Docker info"}
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
        detach: bool = True,
        timeout: int = 600
    ) -> bool:
        """Run a new container using Docker CLI"""
        try:
            args = ["run"]

            if detach:
                args.append("-d")

            args.extend(["--name", name])

            # Remove existing container if present
            self.remove_container(name, force=True)

            # Add restart policy
            args.extend(["--restart", "unless-stopped"])

            # GPU support
            if gpu:
                args.extend(["--gpus", "all"])

            # Network
            if network:
                args.extend(["--network", network])

            # Ports
            if ports:
                for container_port, host_port in ports.items():
                    args.extend(["-p", f"{host_port}:{container_port}"])

            # Volumes
            if volumes:
                for host_path, mount_info in volumes.items():
                    if isinstance(mount_info, dict):
                        container_path = mount_info.get("bind", host_path)
                        mode = mount_info.get("mode", "rw")
                    else:
                        container_path = mount_info
                        mode = "rw"
                    args.extend(["-v", f"{host_path}:{container_path}:{mode}"])

            # Environment
            if environment:
                for key, value in environment.items():
                    args.extend(["-e", f"{key}={value}"])

            args.append(image)

            if command:
                args.extend(command.split())

            result = self._run(args, timeout=timeout)
            if result.returncode == 0:
                logger.info(f"Started container {name}")
                return True
            logger.warning(f"Failed to start {name}: returncode={result.returncode}, stderr={result.stderr.strip()}, stdout={result.stdout.strip()}")
            return False

        except Exception as e:
            logger.error(f"Error running container {name}: {e}")
            return False


# Singleton instance
docker_manager = DockerManager()