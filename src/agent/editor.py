"""
实现视频编辑器的核心功能，根据段落规划生成视频片段。
"""

import logging
from copy import deepcopy
from typing import List, Dict
import numpy as np

from .retriever import ShotCandidate
from .editing_utils import ScoreConfig, EditResult
from .retriever import get_candidates_pool, retrieve

logger = logging.getLogger('editor')

def generate_segment_video(
    video_features, 
    prompt_embed: np.ndarray, 
    cut_points: List[float], 
    config: ScoreConfig, 
    prev_shots: EditResult,
    beam_size: int = 5, 
    exploration: int = 3, 
    pool_size: int = 300, 
    fps: int = 24
) -> List[EditResult]:
    """根据段落规划生成视频片段。

    参数：
        video_features: 视频特征字典
        prompt_embed: 段落提示词嵌入
        cutpoints: 剪辑点（秒）
        config: 评分权重
        done_frames: 已完成的帧数
        banned_shots: 使用过的片段列表
        beam_size: Beam搜索大小
        exploration: 探索数量
        pool_size: 候选池大小
        fps: 帧率
    """
    done_frames = prev_shots.total_frames
    logger.info(f"为段落生成视频，剪辑点：{cut_points}，已完成帧数：{done_frames}。")

    beam_state = [
        EditResult(video_candidates=[], total_frames=0, total_score=0.0)
    ]
    
    candidate_pool = get_candidates_pool(
        video_features,
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
                video_features,
                candidate_pool,
                prev_shots.video_candidates + segment.video_candidates,
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

