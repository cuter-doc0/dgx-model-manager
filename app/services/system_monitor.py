"""
System monitor service - Get system status and GPU information
"""

import os
import subprocess
import logging
from typing import List, Optional
import psutil

from app.models import SystemStatus, GPUInfo
from app.config import get_config
from app.services.docker_manager import docker_manager

logger = logging.getLogger(__name__)


class SystemMonitor:
    """Monitor system status"""
    
    def __init__(self):
        self.config = get_config()
    
    def get_status(self) -> SystemStatus:
        """Get complete system status"""
        try:
            # Get hostname and IP
            hostname = os.uname().nodename
            ip_address = self._get_ip_address()
            
            # Get system info
            ram = psutil.virtual_memory()
            disk = psutil.disk_usage("/")
            
            # Get GPU info
            gpus = self._get_gpu_info()
            
            # Get Docker version
            docker_version = None
            if docker_manager.is_available():
                docker_info = docker_manager.get_docker_info()
                docker_version = docker_info.get("version")
            
            # Get uptime
            uptime = self._get_uptime()
            
            return SystemStatus(
                hostname=hostname,
                ip_address=ip_address,
                architecture=os.uname().machine,
                python_version=f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
                uptime=uptime,
                cpu_count=psutil.cpu_count(),
                cpu_percent=psutil.cpu_percent(interval=0.1),
                ram_total=ram.total,
                ram_used=ram.used,
                ram_free=ram.available,
                disk_total=disk.total,
                disk_used=disk.used,
                disk_free=disk.free,
                gpus=gpus,
                docker_version=docker_version
            )
            
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            raise
    
    def _get_ip_address(self) -> str:
        """Get primary IP address"""
        try:
            # Get IP address
            import socket
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                s.connect(("8.8.8.8", 80))
                ip = s.getsockname()[0]
            finally:
                s.close()
            return ip
        except Exception:
            return "127.0.0.1"
    
    def _get_uptime(self) -> str:
        """Get system uptime"""
        try:
            boot_time = psutil.boot_time()
            uptime_seconds = psutil.time.time() - boot_time
            
            days = int(uptime_seconds // 86400)
            hours = int((uptime_seconds % 86400) // 3600)
            minutes = int((uptime_seconds % 3600) // 60)
            
            parts = []
            if days > 0:
                parts.append(f"{days}d")
            if hours > 0:
                parts.append(f"{hours}h")
            parts.append(f"{minutes}m")
            
            return " ".join(parts)
            
        except Exception as e:
            logger.error(f"Error getting uptime: {e}")
            return "unknown"
    
    def _get_gpu_info(self) -> List[GPUInfo]:
        """Get GPU information using nvidia-smi"""
        try:
            # Try full path first, then just nvidia-smi
            nvidia_smi_paths = ["/usr/bin/nvidia-smi", "nvidia-smi"]
            
            for nvidia_smi in nvidia_smi_paths:
                result = subprocess.run(
                    [nvidia_smi, "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
                     "--format=csv,noheader,nounits"],
                    capture_output=True,
                    text=True
                )
                
                if result.returncode == 0:
                    break
            else:
                # Try using pynvml as fallback
                return self._get_gpu_info_pynvml()
            
            gpus = []
            for line in result.stdout.strip().split("\n"):
                if not line.strip():
                    continue
                
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 6:
                    gpus.append(GPUInfo(
                        index=int(parts[0]),
                        name=parts[1],
                        memory_total=int(parts[2]),
                        memory_used=int(parts[3]),
                        memory_free=int(parts[4]),
                        utilization=float(parts[5])
                    ))
            
            return gpus
            
        except FileNotFoundError:
            logger.warning("nvidia-smi not found, trying pynvml")
            return self._get_gpu_info_pynvml()
        except Exception as e:
            logger.error(f"Error getting GPU info: {e}")
            return []
    
    def _get_gpu_info_pynvml(self) -> List[GPUInfo]:
        """Fallback GPU detection using pynvml"""
        try:
            import pynvml
            pynvml.nvmlInit()
            
            device_count = pynvml.nvmlDeviceGetCount()
            gpus = []
            
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode('utf-8')
                
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                
                gpus.append(GPUInfo(
                    index=i,
                    name=name,
                    memory_total=mem_info.total // (1024 * 1024),  # Convert to MB
                    memory_used=mem_info.used // (1024 * 1024),
                    memory_free=mem_info.free // (1024 * 1024),
                    utilization=util.gpu
                ))
            
            pynvml.nvmlShutdown()
            return gpus
            
        except Exception as e:
            logger.warning(f"pynvml fallback failed: {e}")
            return []
    
    def get_health_check(self) -> dict:
        """Quick health check"""
        try:
            ram = psutil.virtual_memory()
            disk = psutil.disk_usage("/")
            
            return {
                "status": "healthy",
                "cpu_percent": psutil.cpu_percent(interval=0.1),
                "ram_percent": ram.percent,
                "disk_percent": (disk.used / disk.total) * 100,
                "gpus": len(self._get_gpu_info())
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}


# Singleton instance
system_monitor = SystemMonitor()