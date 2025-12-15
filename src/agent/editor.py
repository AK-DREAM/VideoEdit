from ..features.features import VideoFeatures
from .planner import get_plan_with_llm, SegmentPlan, Plan
from .retriever import ShotCandidate, Score, ScoreConfig
from .retriever import add_video_features, get_candidates_pool, retrieve
from ..utils.path import load_video_list

from typing import List
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict
import pickle
import os
import cv2
import numpy as np
from copy import deepcopy

logger = logging.getLogger('editor')

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

    def generate_video(self, output_path: Path, fps: int = 24) -> None:
        """生成最终视频
        
        参数：
            output_path: 输出视频路径
            fps: 帧率
        """
        global video_paths
        
        if not self.video_candidates:
            logger.warning("没有候选片段，无法生成视频")
            return
        
        # 获取视频分辨率（从第一个候选片段所属的视频）
        first_candidate = self.video_candidates[0]
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

video_paths: Dict[str, Path] = {}

def load_video_features(csv_path: Path, data_root: Path, output_root: Path):
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
    
    for video_id, filepath in load_video_list(csv_path):
        feature_path = output_root / Path(filepath).parent / "features.pkl"
        
        video_feat = pickle.load(open(feature_path, "rb"))
        video_feat.pre_calc()
        
        add_video_features(video_id, video_feat)

        video_path = data_root / filepath
        video_paths[video_id] = video_path

    logger.info(f"成功加载 {len(video_paths)} 个视频")


def generate_segment_video(prompt_embed: np.ndarray, cut_points: List[float], config: ScoreConfig, done_frames: int, beam_size: int = 5, exploration: int = 3, pool_size: int = 100, fps: int=24) -> List[EditResult]:
    """根据段落规划生成视频片段。

    参数：
        prompt_embed: 段落提示词嵌入
        cutpoints: 剪辑点（秒）
        weight: 评分权重
        done_frames: 已完成的帧数
        beam_size: Beam搜索大小
        exploration: 探索数量
        pool_size: 候选池大小
        fps: 帧率
    """
    logger.info(f"为段落生成视频，剪辑点：{cut_points}，已完成帧数：{done_frames}")

    beam_state = [
        EditResult(video_candidates=[], total_frames=0, total_score=0.0)
    ]
    
    candidate_pool = get_candidates_pool(
        prompt_embed,
        pool_size
    )

    for cut_point in cut_points:
        shot_len = round(cut_point * fps) - done_frames
        if shot_len <= 0:
            logger.warning(f"剪辑点 {cut_point}s 对应的帧数小于已完成帧数，跳过该剪辑点。")
            continue
        new_beam_state = []
        for segment in beam_state:
            retrieved_shots = retrieve(
                candidate_pool,
                segment.video_candidates,
                config,
                shot_len,
                top_k=exploration
            )

            for shot in retrieved_shots:
                new_segment = deepcopy(segment)
                new_segment.append_shot(shot)
                new_beam_state.append(new_segment)

        # 按总分排序，保留前beam_size个
        new_beam_state.sort(key=lambda x: x.total_score, reverse=True)
        beam_state = new_beam_state[:beam_size]
        done_frames += shot_len
    
    logger.info(f"段落视频生成完成，生成了 {len(beam_state)} 个候选结果。")
    return beam_state

