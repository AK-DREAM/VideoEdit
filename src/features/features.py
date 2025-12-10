from dataclasses import dataclass, field
import numpy as np
from typing import List, Optional, Tuple

@dataclass
class KeyFrameFeatures:
    frame_idx: int
    saliency: np.ndarray
    clip_embed: np.ndarray
    optical_flow: np.ndarray

@dataclass
class ShotFeatures:
    start_frame: int
    end_frame: int
    start_saliency: np.ndarray
    end_saliency: np.ndarray
    clip_embed: np.ndarray
    start_flow: np.ndarray
    end_flow: np.ndarray

@dataclass
class VideoFeatures:
    video_id: str
    shots: list[dict[str, int]] # {"start": int, "end": int}
    keyframes: list[KeyFrameFeatures]
    keyframe_interval: int = 4
    
    # The first keyframe doesn't have an "optical_flow"

    def __init__(self, video_id, num_frames):
        self.video_id = video_id
        keyframe_indices = list(range(0, num_frames, self.keyframe_interval))
        self.keyframes = [
            KeyFrameFeatures(
                frame_idx = f,
                saliency = None,
                clip_embed = None,
                optical_flow = None,
            )
            for f in keyframe_indices
        ]

    def _next_kf(self, idx: int):
        return (idx + self.keyframe_interval - 1) // self.keyframe_interval
    
    def _last_kf(self, idx: int):
        return idx // self.keyframe_interval

    def get_start_saliency(self, st: int, ed: int):
        idx = self._next_kf(st)
        return self.keyframes[idx].saliency

    def get_end_saliency(self, st: int, ed: int):
        idx = self._last_kf(ed)
        return self.keyframes[idx].saliency 

    def get_start_flow(self, st: int, ed: int):
        idx = self._next_kf(st + self.keyframe_interval)
        return self.keyframes[idx].optical_flow

    def get_end_flow(self, st: int, ed: int):
        idx = self._last_kf(ed)
        return self.keyframes[idx].optical_flow

    def get_clip_embed(self, st: int, ed: int):
        s_idx = self._next_kf(st)
        e_idx = self._last_kf(ed)

        embeds = [self.keyframes[i].clip_embed for i in range(s_idx, e_idx + 1)]
        return np.mean(embeds, axis=0)

    def get_shot_features(self, st: int, ed: int) -> ShotFeatures:
        return ShotFeatures(
            start_frame = st,
            end_frame = ed,
            start_saliency = self.get_start_saliency(st, ed),
            end_saliency = self.get_end_saliency(st, ed),
            clip_embed = self.get_clip_embed(st, ed),
            start_flow = self.get_start_flow(st, ed),
            end_flow = self.get_end_flow(st, ed),
        )