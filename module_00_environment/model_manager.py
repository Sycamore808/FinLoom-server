# -*- coding: utf-8 -*-
"""
FIN-R1模型智能管理系统
支持：
- 搜索本地已有模型
- 检测系统配置
- 自定义安装位置
- 模型下载和更新
"""

import json
import logging
import os
import platform
import psutil
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ModelManager:
    """FIN-R1模型管理器"""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.default_model_dir = self.project_root / ".Fin-R1"
        self.config_file = self.project_root / "config" / "model_config.yaml"

    def search_local_models(self, search_paths: Optional[List[str]] = None) -> List[Dict]:
        """
        搜索本地已有的FIN-R1模型
        
        Args:
            search_paths: 搜索路径列表，None时搜索常见位置
            
        Returns:
            找到的模型列表，每个元素包含路径、版本、大小等信息
        """
        logger.info("正在搜索本地FIN-R1模型...")
        
        models_found = []
        
        # 默认搜索路径
        if search_paths is None:
            search_paths = self._get_default_search_paths()
        
        # 遍历搜索路径
        for search_path in search_paths:
            try:
                path = Path(search_path)
                if not path.exists():
                    continue
                
                # 搜索包含FIN-R1模型标识的目录
                for item in path.rglob("*"):
                    if item.is_dir() and self._is_fin_r1_model(item):
                        model_info = self._get_model_info(item)
                        models_found.append(model_info)
                        logger.info(f"找到模型: {item}")
                        
            except PermissionError:
                logger.warning(f"无权限访问: {search_path}")
            except Exception as e:
                logger.error(f"搜索 {search_path} 时出错: {e}")
        
        logger.info(f"共找到 {len(models_found)} 个FIN-R1模型")
        return models_found

    def _get_default_search_paths(self) -> List[str]:
        """获取默认搜索路径"""
        paths = []
        
        # 当前项目目录
        paths.append(str(self.project_root))
        
        # 用户主目录
        home = Path.home()
        paths.append(str(home))
        
        # 常见AI模型存放位置
        if platform.system() == "Windows":
            # Windows常见路径
            paths.extend([
                str(home / "Documents"),
                str(home / "Downloads"),
                "C:\\AI_Models",
                "D:\\AI_Models",
                "C:\\Users\\Public\\AI_Models",
            ])
        else:
            # Linux/Mac常见路径
            paths.extend([
                str(home / "Documents"),
                str(home / "Downloads"),
                str(home / "AI_Models"),
                "/opt/ai_models",
                "/usr/local/ai_models",
            ])
        
        # Hugging Face缓存目录
        hf_cache = home / ".cache" / "huggingface"
        if hf_cache.exists():
            paths.append(str(hf_cache))
        
        # ModelScope缓存目录
        ms_cache = home / ".cache" / "modelscope"
        if ms_cache.exists():
            paths.append(str(ms_cache))
        
        return paths

    def _is_fin_r1_model(self, path: Path) -> bool:
        """判断目录是否为FIN-R1模型"""
        # 检查必需的文件
        required_files = ["config.json"]
        optional_files = ["pytorch_model.bin", "model.safetensors", "tokenizer_config.json"]
        
        # 至少要有config.json
        if not (path / "config.json").exists():
            return False
        
        # 检查配置文件内容
        try:
            with open(path / "config.json", 'r', encoding='utf-8') as f:
                config = json.load(f)
                # 检查是否包含FIN-R1相关标识
                model_type = config.get("model_type", "").lower()
                if "fin" in model_type or "r1" in model_type:
                    return True
                # 检查目录名
                if "fin" in path.name.lower() and "r1" in path.name.lower():
                    return True
        except:
            pass
        
        return False

    def _get_model_info(self, path: Path) -> Dict:
        """获取模型详细信息"""
        info = {
            "path": str(path),
            "name": path.name,
            "size_mb": 0,
            "version": "unknown",
            "last_modified": "",
            "config": {}
        }
        
        try:
            # 计算目录大小
            total_size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
            info["size_mb"] = round(total_size / (1024 * 1024), 2)
            
            # 获取最后修改时间
            info["last_modified"] = path.stat().st_mtime
            
            # 读取配置信息
            config_file = path / "config.json"
            if config_file.exists():
                with open(config_file, 'r', encoding='utf-8') as f:
                    info["config"] = json.load(f)
                    info["version"] = info["config"].get("version", "unknown")
        except Exception as e:
            logger.error(f"获取模型信息失败: {e}")
        
        return info

    def check_system_requirements(self) -> Dict:
        """
        检测系统配置是否满足FIN-R1部署要求
        
        Returns:
            系统配置信息和是否满足要求
        """
        logger.info("检测系统配置...")
        
        requirements = {
            "meets_requirements": True,
            "issues": [],
            "system_info": {}
        }
        
        # CPU信息
        cpu_count = psutil.cpu_count(logical=True)
        cpu_freq = psutil.cpu_freq()
        requirements["system_info"]["cpu_count"] = cpu_count
        requirements["system_info"]["cpu_freq_mhz"] = cpu_freq.current if cpu_freq else 0
        
        if cpu_count < 4:
            requirements["meets_requirements"] = False
            requirements["issues"].append("CPU核心数不足（建议至少4核）")
        
        # 内存信息
        memory = psutil.virtual_memory()
        memory_gb = memory.total / (1024 ** 3)
        requirements["system_info"]["memory_gb"] = round(memory_gb, 2)
        requirements["system_info"]["memory_available_gb"] = round(memory.available / (1024 ** 3), 2)
        
        if memory_gb < 8:
            requirements["meets_requirements"] = False
            requirements["issues"].append("内存不足（建议至少8GB，推荐16GB以上）")
        elif memory_gb < 16:
            requirements["issues"].append("内存较小（推荐16GB以上以获得更好性能）")
        
        # 磁盘空间
        disk = psutil.disk_usage('/')
        disk_free_gb = disk.free / (1024 ** 3)
        requirements["system_info"]["disk_free_gb"] = round(disk_free_gb, 2)
        
        if disk_free_gb < 10:
            requirements["meets_requirements"] = False
            requirements["issues"].append("磁盘空间不足（模型需要至少10GB空间）")
        
        # GPU检测
        gpu_available = self._check_gpu()
        requirements["system_info"]["gpu_available"] = gpu_available
        requirements["system_info"]["cuda_available"] = self._check_cuda()
        
        if not gpu_available:
            requirements["issues"].append("未检测到GPU（将使用CPU运行，速度较慢）")
        
        # Python版本
        python_version = platform.python_version()
        requirements["system_info"]["python_version"] = python_version
        
        major, minor = python_version.split('.')[:2]
        if int(major) < 3 or (int(major) == 3 and int(minor) < 8):
            requirements["meets_requirements"] = False
            requirements["issues"].append("Python版本过低（需要Python 3.8+）")
        
        logger.info(f"系统配置检测完成: {'满足要求' if requirements['meets_requirements'] else '不满足要求'}")
        return requirements

    def _check_gpu(self) -> bool:
        """检测是否有GPU"""
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False

    def _check_cuda(self) -> bool:
        """检测CUDA是否可用"""
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False

    def get_available_disks(self) -> List[Dict]:
        """
        获取所有可用磁盘及其空间信息
        
        Returns:
            磁盘列表，包含路径、空间等信息
        """
        disks = []
        
        if platform.system() == "Windows":
            # Windows: 检查所有驱动器
            import string
            for letter in string.ascii_uppercase:
                drive = f"{letter}:\\"
                if os.path.exists(drive):
                    try:
                        usage = psutil.disk_usage(drive)
                        disks.append({
                            "path": drive,
                            "label": f"{letter}盘",
                            "total_gb": round(usage.total / (1024 ** 3), 2),
                            "free_gb": round(usage.free / (1024 ** 3), 2),
                            "used_gb": round(usage.used / (1024 ** 3), 2),
                            "percent_used": usage.percent
                        })
                    except:
                        pass
        else:
            # Linux/Mac: 检查挂载点
            partitions = psutil.disk_partitions()
            for partition in partitions:
                try:
                    usage = psutil.disk_usage(partition.mountpoint)
                    disks.append({
                        "path": partition.mountpoint,
                        "label": partition.device,
                        "total_gb": round(usage.total / (1024 ** 3), 2),
                        "free_gb": round(usage.free / (1024 ** 3), 2),
                        "used_gb": round(usage.used / (1024 ** 3), 2),
                        "percent_used": usage.percent
                    })
                except:
                    pass
        
        return disks

    def download_model(
        self,
        target_path: Optional[str] = None,
        source: str = "modelscope"
    ) -> Tuple[bool, str]:
        """
        下载FIN-R1模型
        
        Args:
            target_path: 目标安装路径，None时使用默认路径
            source: 模型来源 ("modelscope" or "huggingface")
            
        Returns:
            (是否成功, 消息)
        """
        if target_path is None:
            target_path = str(self.default_model_dir)
        
        model_dir = Path(target_path)
        
        # 检查路径是否已存在
        if model_dir.exists():
            return False, f"目录已存在: {target_path}"
        
        logger.info(f"开始下载FIN-R1模型到: {target_path}")
        
        # 检查Git LFS
        if not self._check_git_lfs():
            return False, "Git LFS未安装，请先安装Git LFS"
        
        # 下载模型
        try:
            if source == "modelscope":
                repo_url = "https://www.modelscope.cn/AI-ModelScope/Fin-R1.git"
            else:
                repo_url = "https://huggingface.co/AI-ModelScope/Fin-R1"
            
            logger.info(f"从 {source} 下载模型...")
            
            cmd = ["git", "clone", repo_url, str(model_dir)]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1800  # 30分钟
            )
            
            if result.returncode == 0:
                logger.info("模型下载成功")
                # 更新配置文件
                self._update_model_config(target_path)
                return True, "模型下载成功"
            else:
                logger.error(f"模型下载失败: {result.stderr}")
                return False, f"下载失败: {result.stderr}"
                
        except subprocess.TimeoutExpired:
            return False, "下载超时（30分钟）"
        except Exception as e:
            logger.error(f"下载过程出错: {e}")
            return False, f"下载出错: {str(e)}"

    def _check_git_lfs(self) -> bool:
        """检查Git LFS是否已安装"""
        try:
            result = subprocess.run(
                ["git", "lfs", "version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.returncode == 0
        except:
            return False

    def _update_model_config(self, model_path: str):
        """更新模型配置文件"""
        try:
            import yaml
            
            if self.config_file.exists():
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f) or {}
            else:
                config = {}
            
            # 更新模型路径
            if "fin_r1" not in config:
                config["fin_r1"] = {}
            
            config["fin_r1"]["model_path"] = model_path
            config["fin_r1"]["last_updated"] = str(Path(model_path).stat().st_mtime)
            
            # 保存配置
            with open(self.config_file, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, allow_unicode=True)
                
            logger.info("配置文件已更新")
        except Exception as e:
            logger.error(f"更新配置文件失败: {e}")

    def use_existing_model(self, model_path: str) -> Tuple[bool, str]:
        """
        使用已有的本地模型
        
        Args:
            model_path: 本地模型路径
            
        Returns:
            (是否成功, 消息)
        """
        path = Path(model_path)
        
        if not path.exists():
            return False, "模型路径不存在"
        
        if not self._is_fin_r1_model(path):
            return False, "不是有效的FIN-R1模型"
        
        try:
            # 更新配置文件指向这个模型
            self._update_model_config(model_path)
            logger.info(f"已配置使用模型: {model_path}")
            return True, "模型配置成功"
        except Exception as e:
            logger.error(f"配置模型失败: {e}")
            return False, f"配置失败: {str(e)}"

    def get_model_status(self) -> Dict:
        """
        获取当前模型状态
        
        Returns:
            模型状态信息
        """
        status = {
            "configured": False,
            "path": None,
            "exists": False,
            "size_mb": 0,
            "info": {}
        }
        
        # 检查配置文件
        try:
            import yaml
            if self.config_file.exists():
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f) or {}
                    
                if "fin_r1" in config and "model_path" in config["fin_r1"]:
                    model_path = Path(config["fin_r1"]["model_path"])
                    status["configured"] = True
                    status["path"] = str(model_path)
                    
                    if model_path.exists():
                        status["exists"] = True
                        status["info"] = self._get_model_info(model_path)
                        status["size_mb"] = status["info"]["size_mb"]
        except Exception as e:
            logger.error(f"获取模型状态失败: {e}")
        
        return status


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    manager = ModelManager()
    
    print("\n=== 系统配置检测 ===")
    requirements = manager.check_system_requirements()
    print(f"满足要求: {requirements['meets_requirements']}")
    print(f"系统信息: {json.dumps(requirements['system_info'], indent=2, ensure_ascii=False)}")
    if requirements['issues']:
        print(f"问题: {requirements['issues']}")
    
    print("\n=== 可用磁盘 ===")
    disks = manager.get_available_disks()
    for disk in disks:
        print(f"{disk['label']}: {disk['free_gb']}GB 可用 / {disk['total_gb']}GB 总计")
    
    print("\n=== 搜索本地模型 ===")
    models = manager.search_local_models()
    print(f"找到 {len(models)} 个模型")
    for model in models:
        print(f"- {model['path']} ({model['size_mb']} MB)")













