import numpy as np

def _emd_dist(H1, H2):
    cdf1 = np.cumsum(H1)
    cdf2 = np.cumsum(H2)
    dist = np.sum(np.abs(cdf1 - cdf2))
    return dist / len(H1)

def _get_fast_emd(map1, map2):
    m1 = map1 / (np.sum(map1) + 1e-10)
    m2 = map2 / (np.sum(map2) + 1e-10)

    x_pdf1 = np.sum(m1, axis=0)
    x_pdf2 = np.sum(m2, axis=0)
    
    y_pdf1 = np.sum(m1, axis=1)
    y_pdf2 = np.sum(m2, axis=1)

    dist_x = _emd_dist(x_pdf1, x_pdf2)
    dist_y = _emd_dist(y_pdf1, y_pdf2)

    norm_dist = np.sqrt((dist_x)**2 + (dist_y)**2)
    
    return norm_dist

def get_saliency_score(map1, map2):
    return 1 - _get_fast_emd(map1, map2)