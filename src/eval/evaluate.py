from .metrics import get_saliency_score, get_semantic_score, get_motion_score
from ..features import VideoFeatures, ShotFeatures
from typing import List

def evaluate_scores(features: VideoFeatures):
    shots: List[ShotFeatures] = []

    for curr_shot in features.shots:
        shots.append(features.get_shot_features(curr_shot["start"], curr_shot["end"]))
    
    saliency_scores = []
    semantic_scores = []
    motion_scores = []

    for i in range(len(shots) - 1):
        saliency_scores.append(get_saliency_score(shots[i].end_saliency, shots[i+1].start_saliency))
        semantic_scores.append(get_semantic_score(shots[i].clip_embed, shots[i+1].clip_embed))
        motion_scores.append(get_motion_score(shots[i].end_flow, shots[i+1].start_flow))

    return saliency_scores, semantic_scores, motion_scores