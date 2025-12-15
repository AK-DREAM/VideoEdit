import numpy as np

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    # vec1 = vec1.flatten()
    # vec2 = vec2.flatten()
    dot_product = np.dot(vec1, vec2)
    # norm_vec1 = np.linalg.norm(vec1)
    norm_vec1 = np.sqrt(np.dot(vec1, vec1))
    # norm_vec2 = np.linalg.norm(vec2)
    norm_vec2 = np.sqrt(np.dot(vec2, vec2))

    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0

    return dot_product / (norm_vec1 * norm_vec2)

def get_semantic_score(vec1: np.ndarray, vec2: np.ndarray) -> float:
    return cosine_similarity(vec1, vec2)