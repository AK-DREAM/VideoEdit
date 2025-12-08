from pathlib import Path
from datetime import datetime

def get_root_dir() -> Path:
    """获取项目根目录的路径。

    返回：
        Path: 项目根目录的完整路径。
    """
    return Path(__file__).parent.parent.parent

def get_config_path(name) -> Path:
    """获取配置文件的路径。

    参数：
        name (str): 配置文件名。
    返回：
        Path: 配置文件的完整路径。
    """
    config_dir = get_root_dir() / "configs"
    return config_dir / name

def get_output_dir() -> Path:
    """获取输出目录的路径。

    返回：
        Path: 输出目录的完整路径。
    """
    output_dir = get_root_dir() / "output"
    return output_dir

def get_log_path() -> Path:
    """获取日志文件的路径。

    返回：
        Path: 日志文件的完整路径，使用时间戳作为文件名。
    """
    log_dir = get_output_dir() / "logs"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return log_dir / f"log_{timestamp}.txt"