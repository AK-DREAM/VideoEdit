import numpy as np
from dataclasses import dataclass

@dataclass
class SaliencyFeature:
    x_cdf: np.ndarray
    y_cdf: np.ndarray

    def __init__(self, map: np.ndarray):
        m = map / (np.sum(map) + 1e-10)
        self.x_cdf = np.cumsum(np.sum(m, axis=0))
        self.y_cdf = np.cumsum(np.sum(m, axis=1))

def _emd_dist(cdf1: np.ndarray, cdf2: np.ndarray) -> float:
    dist = np.sum(np.abs(cdf1 - cdf2))
    return dist / len(cdf1)

def _get_fast_emd(feat1: SaliencyFeature, feat2: SaliencyFeature) -> float:

    dist_x = _emd_dist(feat1.x_cdf, feat2.x_cdf)
    dist_y = _emd_dist(feat1.y_cdf, feat2.y_cdf)

    norm_dist = np.sqrt((dist_x)**2 + (dist_y)**2)
    
    return norm_dist

def get_saliency_score(feat1: SaliencyFeature, feat2: SaliencyFeature) -> float:
    return 1 - _get_fast_emd(feat1, feat2)