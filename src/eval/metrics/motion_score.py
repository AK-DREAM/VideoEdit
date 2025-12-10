import numpy as np

def _extract_top_flows(flow_field, top_ratio):
    u = flow_field[0].reshape(-1)
    v = flow_field[1].reshape(-1)
    mag = np.sqrt(u**2 + v**2)
    k = max(1, int(len(mag) * top_ratio))
    idx = np.argpartition(mag, -k)[-k:]
    top_u = u[idx]
    top_v = v[idx]
    return (np.mean(top_u), np.mean(top_v))

def _sigmoid(x):
    return 1 / (1 + np.exp(-x))

def get_motion_score(flow1, flow2, top_ratio=0.05):    
    (u1, v1) = _extract_top_flows(flow1, top_ratio)
    (u2, v2) = _extract_top_flows(flow2, top_ratio)
    m1 = np.sqrt(u1**2+v1**2)
    m2 = np.sqrt(u2**2+v2**2)
    # mag_sim = 1 - np.tanh(np.abs(m1 - m2) / 80.0)
    mag_sim = 40.0 / (np.abs(m1 - m2) + 40.0)
    dir_sim = np.abs((u1 * u2 + v1 * v2) / (m1 * m2))
    # dir_conf = _sigmoid((m1 - 8) / 2) * _sigmoid((m2 - 8) / 2)
    dir_conf = (m1 / (m1 + 2)) * (m2 / (m2 + 2))
    return mag_sim * (1 - dir_conf) + dir_sim * dir_conf