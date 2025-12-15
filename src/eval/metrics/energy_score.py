import numpy as np



def get_energy_score(v: float, v0: float) -> float:
    """
    参数
    v: 期望速度
    v0: 实际速度
    """
    log_v = np.log(v)
    log_v0 = np.log(v0)
    return np.exp(- (log_v - log_v0) ** 2 * 2)