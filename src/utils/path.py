from pathlib import Path
from datetime import datetime
import csv
import yaml
from typing import Dict, List, Tuple

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

def get_config(name: str) -> Dict:
    """加载并返回配置文件内容。

    参数：
        name (str): 配置文件名。
    返回：
        dict: 配置文件内容的字典表示。
    """
    config_path = get_config_path(name)
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config

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

def load_video_list(csv_path: Path) -> List[Tuple[str, str]]:
    videos = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            video_id, filepath = row["video_id"], row["filepath"]
            videos.append((video_id, filepath))
    return videos