import math
import numpy as np

def compute_chaos_vector(
    vectors,
    frame_area=None,
    angle_bins=16,
    dbscan_eps=75.0,
    dbscan_min_samples=3,
    conflict_sigma=10.0
):
    """
    traffic disorder score using sigmoid count gate
    
    vectors: list of (sx, sy, ex, ey)
    return: disorder score in [0, 1]
    """
    N = len(vectors)
    if N < 2:
        return 0.0

    # 1. compute angles
    angles = []
    for sx, sy, ex, ey in vectors:
        angles.append(math.atan2(ey - sy, ex - sx))

    angles = np.asarray(angles)

    # 2. directional disorder
    R = math.hypot(np.cos(angles).sum(), np.sin(angles).sum()) / N
    direction_disorder = 1.0 - R   # [0,1]

    # 3. count gate (sigmoid)
    N0 = 8     # threshold for "crowded"
    k = 3.0
    count_factor = 1.0 / (1.0 + math.exp(-(N - N0) / k))

    # 4. final disorder
    disorder = direction_disorder * count_factor

    return float(max(0.0, min(1.0, disorder)))


# alias for backward compatibility
compute_chaos_score = compute_chaos_vector
