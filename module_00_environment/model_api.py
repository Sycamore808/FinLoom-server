"""
FIN-R1模型管理API端点
提供模型状态查询、配置和管理功能
"""

from pathlib import Path
from typing import Dict
import logging

logger = logging.getLogger("model_api")


def register_model_routes(app):
    """注册模型管理相关的API路由"""
    
    @app.get("/api/v1/model/status")
    async def get_model_status():
        """获取FIN-R1模型状态"""
        try:
            import yaml
            
            # 读取配置
            config_path = Path("module_10_ai_interaction/config/fin_r1_config.yaml")
            configured = False
            model_path = None
            exists = False
            size_mb = 0
            
            if config_path.exists():
                with open(config_path, "r", encoding="utf-8") as f:
                    config = yaml.safe_load(f)
                    model_config = config.get("model", {})
                    model_path = model_config.get("model_path", "models/fin_r1")
                    configured = True
                    
                    # 检查模型是否存在
                    full_path = Path(model_path)
                    if full_path.exists():
                        exists = True
                        # 计算模型大小
                        if full_path.is_dir():
                            size_mb = sum(f.stat().st_size for f in full_path.rglob('*') if f.is_file()) / (1024 * 1024)
                        else:
                            size_mb = full_path.stat().st_size / (1024 * 1024)
            
            return {
                "configured": configured,
                "path": str(model_path) if model_path else None,
                "exists": exists,
                "size_mb": size_mb
            }
        except Exception as e:
            logger.error(f"获取模型状态失败: {e}")
            return {
                "configured": False,
                "path": None,
                "exists": False,
                "size_mb": 0,
                "error": str(e)
            }

    @app.get("/api/v1/model/system-requirements")
    async def get_system_requirements():
        """获取系统要求"""
        import psutil
        import platform
        
        try:
            cpu_count = psutil.cpu_count()
            ram_gb = psutil.virtual_memory().total / (1024 ** 3)
            
            # 根据操作系统获取磁盘空间
            if platform.system() == "Windows":
                disk_free_gb = psutil.disk_usage('C:\\').free / (1024 ** 3)
            else:
                disk_free_gb = psutil.disk_usage('/').free / (1024 ** 3)
            
            return {
                "os": platform.system(),
                "cpu_cores": cpu_count,
                "ram_gb": round(ram_gb, 2),
                "disk_free_gb": round(disk_free_gb, 2),
                "python_version": platform.python_version(),
                "requirements_met": cpu_count >= 4 and ram_gb >= 8 and disk_free_gb >= 10
            }
        except Exception as e:
            logger.error(f"获取系统要求失败: {e}")
            return {"error": str(e)}

    @app.get("/api/v1/model/available-disks")
    async def get_available_disks():
        """获取可用磁盘列表"""
        import psutil
        import platform
        
        try:
            disks = []
            
            if platform.system() == "Windows":
                # Windows系统
                for partition in psutil.disk_partitions():
                    try:
                        usage = psutil.disk_usage(partition.mountpoint)
                        disks.append({
                            "letter": partition.device.replace("\\", ""),
                            "label": partition.device,
                            "free_gb": round(usage.free / (1024 ** 3), 2),
                            "total_gb": round(usage.total / (1024 ** 3), 2),
                            "used_percent": usage.percent
                        })
                    except:
                        continue
            else:
                # Linux/Mac系统
                for partition in psutil.disk_partitions():
                    try:
                        usage = psutil.disk_usage(partition.mountpoint)
                        disks.append({
                            "letter": partition.mountpoint,
                            "label": partition.mountpoint,
                            "free_gb": round(usage.free / (1024 ** 3), 2),
                            "total_gb": round(usage.total / (1024 ** 3), 2),
                            "used_percent": usage.percent
                        })
                    except:
                        continue
            
            return {"disks": disks}
        except Exception as e:
            logger.error(f"获取可用磁盘失败: {e}")
            return {"disks": [], "error": str(e)}

    @app.get("/api/v1/model/search-local")
    async def search_local_model(paths: str = None):
        """搜索本地FIN-R1模型"""
        import json
        
        try:
            search_paths = []
            if paths:
                search_paths = json.loads(paths)
            else:
                # 默认搜索路径
                search_paths = [
                    "models/fin_r1",
                    "Fin-R1",
                    ".Fin-R1",
                    str(Path.home() / "models" / "fin_r1"),
                    str(Path.home() / "Fin-R1"),
                ]
            
            found_models = []
            for path_str in search_paths:
                path = Path(path_str)
                if path.exists() and path.is_dir():
                    # 检查是否包含模型文件
                    has_config = (path / "config.json").exists()
                    has_model = any(path.glob("*.bin")) or any(path.glob("*.safetensors"))
                    
                    if has_config or has_model:
                        size_mb = sum(f.stat().st_size for f in path.rglob('*') if f.is_file()) / (1024 * 1024)
                        found_models.append({
                            "path": str(path.absolute()),
                            "size_mb": round(size_mb, 2),
                            "has_config": has_config,
                            "has_model": has_model
                        })
            
            return {"found_models": found_models}
        except Exception as e:
            logger.error(f"搜索本地模型失败: {e}")
            return {"found_models": [], "error": str(e)}

    @app.post("/api/v1/model/use-existing")
    async def use_existing_model(request: Dict):
        """使用现有模型"""
        import yaml
        
        try:
            model_path = request.get("path")
            if not model_path:
                return {"success": False, "error": "未提供模型路径"}
            
            # 验证路径
            path = Path(model_path)
            if not path.exists():
                return {"success": False, "error": "模型路径不存在"}
            
            # 更新配置文件
            config_path = Path("module_10_ai_interaction/config/fin_r1_config.yaml")
            if config_path.exists():
                with open(config_path, "r", encoding="utf-8") as f:
                    config = yaml.safe_load(f)
            else:
                config = {"model": {}, "prompts": {}, "performance": {}}
            
            # 更新模型路径
            if "model" not in config:
                config["model"] = {}
            config["model"]["model_path"] = str(path.absolute())
            
            # 保存配置
            config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(config, f, allow_unicode=True)
            
            logger.info(f"已配置FIN-R1模型路径: {path.absolute()}")
            return {"success": True, "message": "模型配置已更新"}
            
        except Exception as e:
            logger.error(f"配置模型失败: {e}")
            return {"success": False, "error": str(e)}

    @app.post("/api/v1/model/download")
    async def download_model(request: Dict):
        """下载FIN-R1模型（占位符）"""
        return {
            "success": False,
            "error": "自动下载功能暂未实现，请手动下载模型并使用'使用现有模型'功能"
        }
    
    logger.info("FIN-R1模型管理API路由已注册")


