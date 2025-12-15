from dataclasses import dataclass
import numpy as np
from typing import List, Dict, Optional, Tuple
from ..eval.metrics import SaliencyFeature, MotionFeature

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
    start_saliency: SaliencyFeature
    end_saliency: SaliencyFeature
    clip_embed: np.ndarray
    start_flow: MotionFeature
    end_flow: MotionFeature
    energy_value: float

@dataclass
class VideoFeatures:
    video_id: str
    shots: List[Dict[str, int]] # {"start": int, "end": int}
    keyframes: List[KeyFrameFeatures]
    keyframe_energy: np.ndarray | None = None
    keyframe_saliency: List[SaliencyFeature] | None = None
    keyframe_motion: List[MotionFeature] | None = None
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

    def pre_calc(self):
        # init keyframe energy
        self.keyframe_energy = np.array([
            np.mean(np.linalg.norm(kf.optical_flow, axis=2))
            if kf.optical_flow is not None else 0.0
            for kf in self.keyframes
        ])
        # init keyframe saliency
        self.keyframe_saliency = [
            SaliencyFeature(kf.saliency)
            for kf in self.keyframes
        ]
        # init keyframe motion
        self.keyframe_motion = [
            MotionFeature(kf.optical_flow)
            if kf.optical_flow is not None else MotionFeature(np.zeros((2,1,1)))
            for kf in self.keyframes
        ]

    def _next_kf(self, idx: int):
        return (idx + self.keyframe_interval - 1) // self.keyframe_interval
    
    def _last_kf(self, idx: int):
        return idx // self.keyframe_interval

    def get_start_saliency(self, st: int, ed: int):
        idx = self._next_kf(st)
        assert self.keyframe_saliency is not None
        return self.keyframe_saliency[idx]

    def get_end_saliency(self, st: int, ed: int):
        idx = self._last_kf(ed)
        assert self.keyframe_saliency is not None
        return self.keyframe_saliency[idx] 

    def get_start_flow(self, st: int, ed: int):
        idx = self._next_kf(st + self.keyframe_interval)
        assert self.keyframe_motion is not None
        return self.keyframe_motion[idx]

    def get_end_flow(self, st: int, ed: int):
        idx = self._last_kf(ed)
        assert self.keyframe_motion is not None
        return self.keyframe_motion[idx]
    
    def get_energy_value(self, st: int, ed: int):
        s_idx = self._next_kf(st + self.keyframe_interval)
        e_idx = self._last_kf(ed)
        assert self.keyframe_energy is not None
        return float(np.mean(self.keyframe_energy[s_idx:e_idx + 1]))


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
            energy_value = self.get_energy_value(st, ed)
        )