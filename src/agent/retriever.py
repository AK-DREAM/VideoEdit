"""三阶段检索：快速索引检索 → 精密区间评分 → 排序返回"""

import numpy as np
from typing import List, Dict
from dataclasses import dataclass
from copy import deepcopy
import logging
import os
from pathlib import Path
import pickle

from ..features.features import VideoFeatures, ShotFeatures
from ..eval.metrics import get_saliency_score, get_semantic_score, get_motion_score, get_energy_score

logger = logging.getLogger('retriever')

video_features: Dict[str, VideoFeatures] = {}

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
    energy_value: float
    energy_weight: float

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


# ==================== 阶段1：快速检索 ====================

def _stage1_retrieve_candidates(video_features: Dict[str, VideoFeatures],
                                 query_embed: np.ndarray,
                                 pool_size: int = 100) -> List[ShotCandidate]:
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


# ==================== 公共接口 ====================

def add_video_features(video_id: str, features: VideoFeatures) -> None:
    """添加视频特征到全局变量
    
    参数：
        video_id: 视频ID
        features: 视频特征对象
    """
    global video_features
    video_features[video_id] = features


def get_candidates_pool(query_embed: np.ndarray,
                        pool_size: int = 100) -> List[ShotCandidate]:
    """获取候选片段池（基于CLIP相似度快速过滤）
    
    参数：
        query_embed: 查询嵌入向量
        pool_size: 候选池大小
    
    返回：
        候选片段列表（按CLIP分数排序）
    """
    global video_features
    
    logger.info(f"候选池检索（{pool_size}）")
    return _stage1_retrieve_candidates(video_features, query_embed, pool_size)


def retrieve(candidates_pool: List[ShotCandidate],
             shots_list: List[ShotCandidate],
             score_config: ScoreConfig,
             next_shot_len: int,
             top_k: int = 10) -> List[ShotCandidate]:
    """检索相似shot的完整流程
    
    参数：
        candidates_pool: 候选池（通过get_candidates_pool()获得）
        shots_list: 当前shots列表
        score_config: 评分配置对象
        next_shot_len: 滑动窗口长度
        top_k: 返回Top-K结果数量
    
    返回：
        Top-K相似候选片段
    """
    global video_features
    
    logger.debug(f"检索相似片段，当前shots数：{len(shots_list)}，候选池大小：{len(candidates_pool)}，需要的片段长度：{next_shot_len}，返回Top-{top_k}")

    if not candidates_pool:
        logger.warning("未找到候选片段")
        return []
    
    if not shots_list:
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
    return result




if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)


