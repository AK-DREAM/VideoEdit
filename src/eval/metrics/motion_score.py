import numpy as np
from typing import Tuple
from dataclasses import dataclass

@dataclass
class MotionFeature:
    top_flows: Tuple[float, float]

    def __init__(self, flow_field: np.ndarray, top_ratio: float=0.05):
        self.top_flows = _extract_top_flows(flow_field, top_ratio)
        

def _extract_top_flows(flow_field: np.ndarray, top_ratio: float) -> Tuple[float, float]:
    u = flow_field[0].reshape(-1)
    v = flow_field[1].reshape(-1)
    mag = np.sqrt(u**2 + v**2)
    k = max(1, int(len(mag) * top_ratio))
    idx = np.argpartition(mag, -k)[-k:]
    top_u = u[idx]
    top_v = v[idx]
    return (np.mean(top_u), np.mean(top_v))

def _sigmoid(x: float) -> float:
    return 1 / (1 + np.exp(-x))

def get_motion_score(feat1: MotionFeature, feat2: MotionFeature) -> float:    
    (u1, v1) = feat1.top_flows
    (u2, v2) = feat2.top_flows
    m1 = np.sqrt(u1**2+v1**2)
    m2 = np.sqrt(u2**2+v2**2)
    # mag_sim = 1 - np.tanh(np.abs(m1 - m2) / 80.0)
    mag_sim = 40.0 / (np.abs(m1 - m2) + 40.0)
    dir_sim = np.abs((u1 * u2 + v1 * v2) / (m1 * m2))
    # dir_conf = _sigmoid((m1 - 8) / 2) * _sigmoid((m2 - 8) / 2)
    dir_conf = (m1 / (m1 + 2)) * (m2 / (m2 + 2))
    return mag_sim * (1 - dir_conf) + dir_sim * dir_conf