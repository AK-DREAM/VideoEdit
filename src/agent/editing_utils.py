from dataclasses import dataclass
import numpy as np
import cv2
from pathlib import Path
import logging
from typing import List, Dict
import pickle
import os

from ..features.features import VideoFeatures, ShotFeatures
from ..eval.metrics import get_saliency_score, get_semantic_score, get_motion_score, get_energy_score
from ..utils.path import load_video_list

logger = logging.getLogger('logging_utils')

@dataclass
class Score:
    """单项评分"""
    prompt: float
    semantic: float
    saliency: float
    motion: float
    energy: float
    combined: float

@dataclass
class ScoreConfig:
    """评分权重"""
    prompt_embed: np.ndarray
    prompt_weight: float
    semantic_weight: float
    saliency_weight: float
    motion_weight: float
    energy_weight: float
    energy_value: float

    def get_score(self, last_shot: ShotFeatures | None, new_shot: ShotFeatures) -> Score:
        """计算加权总分"""
        prompt_score = get_semantic_score(self.prompt_embed, new_shot.clip_embed)
        energy_score = get_energy_score(self.energy_value, new_shot.energy_value)
            
        if last_shot is None:
            semantic_score = 0.0
            saliency_score = 0.0
            motion_score = 0.0
            emotion_score = 0.0 
        else:
            semantic_score = get_semantic_score(last_shot.clip_embed, new_shot.clip_embed)
            saliency_score = get_saliency_score(last_shot.start_saliency, new_shot.start_saliency)
            motion_score = get_motion_score(last_shot.start_flow, new_shot.start_flow)
        
        combined_score = (
            self.prompt_weight * prompt_score +
            self.semantic_weight * semantic_score +
            self.saliency_weight * saliency_score +
            self.motion_weight * motion_score +
            self.energy_weight * energy_score
        )
        return Score(
            prompt=prompt_score,
            semantic=semantic_score,
            saliency=saliency_score,
            motion=motion_score,
            energy=energy_score,
            combined=combined_score
        )

@dataclass
class ShotCandidate:
    """候选片段"""
    video_id: str
    shot_idx: int
    start_frame: int
    end_frame: int
    score: Score

@dataclass
class EditResult:
    """生成结果"""
    video_candidates: List[ShotCandidate]
    total_frames: int
    total_score: float

    def append_shot(self, shot: ShotCandidate) -> None:
        """添加片段"""
        self.video_candidates.append(shot)
        self.total_frames += (shot.end_frame - shot.start_frame + 1)
        self.total_score += shot.score.combined
    
    def extend(self, other: "EditResult") -> None:
        """合并另一个结果"""
        self.video_candidates.extend(other.video_candidates)
        self.total_frames += other.total_frames
        self.total_score += other.total_score

    def generate_video(self, video_paths: Dict[str, Path], output_path: Path, fps: int = 24) -> None:
        """生成最终视频
        
        参数：
            output_path: 输出视频路径
            fps: 帧率
        """
        if not self.video_candidates:
            logger.warning("没有候选片段，无法生成视频")
            return
        
        # 获取视频分辨率（第一个非黑屏候选片段所属的视频）
        # TODO: 需要改一下
        first_candidate = self.video_candidates[1]
        first_video_path = video_paths[first_candidate.video_id]
        
        cap = cv2.VideoCapture(str(first_video_path))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        # 初始化视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'h264') # type: ignore
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_width, frame_height))
        
        logger.info(f"开始生成视频，输出路径：{output_path}")
        
        # 遍历所有候选片段，依次提取和写入帧
        for candidate in self.video_candidates:
            if candidate.video_id == "BLACK_SCREEN":
                # 生成黑屏片段
                num_frames = candidate.end_frame - candidate.start_frame + 1
                black_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
                for _ in range(num_frames):
                    out.write(black_frame)
                logger.debug(f"添加黑屏片段，共 {num_frames} 帧")
                continue

            video_path = video_paths[candidate.video_id]
            start_frame = candidate.start_frame
            end_frame = candidate.end_frame
            
            logger.debug(f"处理片段：{candidate.video_id} [{start_frame}:{end_frame}]")
            
            cap = cv2.VideoCapture(str(video_path))
            
            # 跳到起始帧
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            # 逐帧读取并写入
            for frame_idx in range(start_frame, end_frame + 1):
                ret, frame = cap.read()
                if not ret:
                    logger.warning(f"读取帧失败：{video_path} frame {frame_idx}")
                    break
                
                # 调整帧大小以匹配视频写入器的尺寸
                if frame.shape[:2] != (frame_height, frame_width):
                    frame = cv2.resize(frame, (frame_width, frame_height))
                
                out.write(frame)
            
            cap.release()
        
        # 释放视频写入器
        out.release()
        logger.info(f"视频生成完成：{output_path}")


def load_video_footages(csv_path: Path, data_root: Path, output_root: Path):
    """从CSV文件加载视频特征到全局变量
    
    参数：
        csv_path: CSV文件路径
        output_root: 特征文件根目录
    
    返回：
        加载的视频特征字典
    
    说明：
        CSV文件应该包含以下列：
        - video_id: 视频ID
        - filepath: 视频文件路径
        
        video_path = data_root / filepath

        特征文件路径构造方式：output_root / filepath.parent / "features.pkl"
    """
    if not os.path.exists(csv_path):
        logger.error(f"CSV文件不存在：{csv_path}")
        return {}

    video_paths: Dict[str, Path] = {}
    video_features: Dict[str, VideoFeatures] = {}
    
    for video_id, filepath in load_video_list(csv_path):
        feature_path = output_root / Path(filepath).parent / "features.pkl"
        
        video_feat = pickle.load(open(feature_path, "rb"))
        video_feat.pre_calc()

        video_path = data_root / filepath
        video_paths[video_id] = video_path
        video_features[video_id] = video_feat

    logger.info(f"成功加载 {len(video_paths)} 个视频")

    return video_paths, video_features
