"""
实现检索算法
三阶段检索：快速索引检索 → 精密区间评分 → 排序返回
"""

import numpy as np
from typing import List, Dict
from copy import deepcopy
import logging
import os
from pathlib import Path
import pickle

from ..features.features import VideoFeatures, ShotFeatures
from .editing_utils import Score, ScoreConfig, ShotCandidate
from ..eval.metrics import get_semantic_score

logger = logging.getLogger('retriever')

# ==================== 阶段1：快速检索 ====================

def get_candidates_pool(
    video_features: Dict[str, VideoFeatures],
    query_embed: np.ndarray,
    pool_size: int = 100
) -> List[ShotCandidate]:
    """快速检索候选片段池（基于CLIP相似度）- 内部实现
    
    参数：
        video_features: 视频特征字典
        query_embed: 查询嵌入向量
        pool_size: 候选池大小
    
    返回：
        候选片段列表（按CLIP分数排序）
    """
    candidates = []
    
    for video_id, features in video_features.items():
        for shot_idx, shot in enumerate(features.shots):

            shot_embed = features.get_clip_embed(shot["start"], shot["end"])
            prompt_score = get_semantic_score(query_embed, shot_embed)
            
            candidates.append(ShotCandidate(
                video_id=video_id,
                shot_idx=shot_idx,
                start_frame=shot["start"],
                end_frame=shot["end"],
                score=Score(prompt=prompt_score, semantic=0.0, saliency=0.0, motion=0.0, energy=0.0, combined=prompt_score)
            ))
    
    candidates.sort(key=lambda x: x.score.prompt, reverse=True)
    return candidates[:pool_size]


# ==================== 阶段2：精密评分 ====================

def stage2_precise_scoring(video_features: Dict[str, VideoFeatures],
                           candidates: List[ShotCandidate],
                           query_features: ShotFeatures | None,
                           score_config: ScoreConfig,
                           next_shot_len: int,
                           frame_step: int = 4) -> List[ShotCandidate]:
    """精密评分：计算加权的saliency/semantic/motion分数
    
    参数：
        video_features: 视频特征字典
        candidates: 候选片段列表
        query_features: 查询片段特征
        score_weights: 评分权重对象
        next_shot_len: 滑动窗口长度
    
    返回：
        评分后的候选片段列表
    """
    result_candidates = []
    
    for candidate in candidates:
        candidate = deepcopy(candidate)
        
        features = video_features[candidate.video_id]
        
        candidate_len = candidate.end_frame - candidate.start_frame + 1
        if candidate_len > next_shot_len:
            # 滑动窗口搜索最优位置
            best_score = -1.0
            best_start = candidate.start_frame
            
            for start in range(candidate.start_frame, candidate.end_frame - next_shot_len + 2, frame_step):
                end = start + next_shot_len - 1
                interval_features = features.get_shot_features(start, end)
                
                # 使用get_score()计算综合评分
                score = score_config.get_score(query_features, interval_features)
                
                if score.combined > best_score:
                    best_score = score.combined
                    best_start = start
                    candidate.score = score
            
            # 更新候选的frame范围
            candidate.start_frame = best_start
            candidate.end_frame = best_start + next_shot_len - 1
            result_candidates.append(candidate)
    
    return result_candidates


# ==================== 阶段3：排序返回 ====================

def stage3_rank_and_return(candidates: List[ShotCandidate],
                           top_k: int = 10) -> List[ShotCandidate]:
    """排序并返回Top-K
    
    参数：
        candidates: 候选片段列表
        top_k: 返回结果数量
    
    返回：
        Top-K候选片段
    """
    candidates.sort(key=lambda x: x.score.combined, reverse=True)
    return candidates[:top_k]

def retrieve(video_features: Dict[str, VideoFeatures],
             candidates_pool: List[ShotCandidate],
             shots_list: List[ShotCandidate],
             score_config: ScoreConfig,
             next_shot_len: int,
             top_k: int = 10) -> List[ShotCandidate]:
    """检索相似shot的完整流程
    
    参数：
        video_features: 视频特征字典
        candidates_pool: 候选池（通过get_candidates_pool()获得）
        shots_list: 当前shots列表
        score_config: 评分配置对象
        next_shot_len: 滑动窗口长度
        top_k: 返回Top-K结果数量
    
    返回：
        Top-K相似候选片段
    """
    
    logger.debug(f"检索相似片段，当前shots数：{len(shots_list)}，候选池大小：{len(candidates_pool)}，需要的片段长度：{next_shot_len}，返回Top-{top_k}")

    if not candidates_pool:
        logger.warning("未找到候选片段")
        return []
    
    if not shots_list or shots_list[-1].video_id == "BLACK_SCREEN":
        logger.debug("shots_list为空，使用 None 检索")
        query_features = None
        filtered_pool = candidates_pool
    else:
    # 取最后一个作为查询特征
        query_shot = shots_list[-1]
        query_features = video_features[query_shot.video_id].get_shot_features(
            query_shot.start_frame, query_shot.end_frame
        )
        
        banned_shot = set([(shot.video_id, shot.shot_idx) for shot in shots_list])

        # 过滤：排除同一shot内的候选
        filtered_pool = [
            c for c in candidates_pool 
            if not ((c.video_id, c.shot_idx) in banned_shot)
        ]
    
    # 阶段2：精密评分
    ranked = stage2_precise_scoring(video_features, filtered_pool.copy(), query_features,
                                    score_config, next_shot_len)
    
    # 阶段3：排序返回
    result = stage3_rank_and_return(ranked, top_k)
    logger.debug(f"排序返回Top-{top_k}")

    # 如果没有结果，加入黑屏片段作为兜底
    if not result:
        logger.warning("未找到合适的候选片段，添加黑屏片段作为兜底")
        black_screen_candidate = ShotCandidate(
            video_id="BLACK_SCREEN",
            shot_idx=-1,
            start_frame=0,
            end_frame=next_shot_len - 1,
            score=Score(0, 0, 0, 0, 0, 0)
        )
        result.append(black_screen_candidate)

    return result

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)


