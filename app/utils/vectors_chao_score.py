import math
# from collections import Counter
from itertools import combinations

import numpy as np

# Input:
# vectors: list of tuples (start_x, start_y, end_x, end_y)
# roi_bbox: (xmin, ymin, xmax, ymax) optional for grid-based local mixture
# All coordinates in same unit.

def compute_angles_and_lengths(vectors):
    angles = []
    lengths = []
    midpoints = []
    segments = []
    for sx, sy, ex, ey in vectors:
        dx = ex - sx
        dy = ey - sy
        angle = math.atan2(dy, dx) % (2 * math.pi)
        length = math.hypot(dx, dy)
        angles.append(angle)
        lengths.append(length)
        midpoints.append(((sx + ex) / 2.0, (sy + ey) / 2.0))
        segments.append(((sx, sy), (ex, ey)))
    return np.array(angles), np.array(lengths), midpoints, segments

def mean_resultant_length(angles):
    c = np.cos(angles).sum()
    s = np.sin(angles).sum()
    R = math.hypot(c, s) / len(angles) if len(angles) > 0 else 0.0
    return R

def angular_entropy(angles, k_bins=12):
    if len(angles) == 0:
        return 0.0
    bins = np.linspace(0.0, 2 * math.pi, k_bins + 1)
    counts, _ = np.histogram(angles, bins=bins)
    probs = counts.astype(float) / counts.sum()
    probs = probs[probs > 0]
    H = -np.sum(probs * np.log(probs))
    H_norm = H / math.log(k_bins)
    return H_norm

def grid_local_mixture(angles, midpoints, counts_lengths, roi_bbox, grid_size=4, k_bins=8):
    if roi_bbox is None or len(angles) == 0:
        return 0.0
    xmin, ymin, xmax, ymax = roi_bbox
    gx = grid_size
    gy = grid_size
    cell_scores = []
    weights = []
    for ix in range(gx):
        for iy in range(gy):
            x0 = xmin + (xmax - xmin) * ix / gx
            x1 = xmin + (xmax - xmin) * (ix + 1) / gx
            y0 = ymin + (ymax - ymin) * iy / gy
            y1 = ymin + (ymax - ymin) * (iy + 1) / gy
            idxs = []
            for i, (mx, my) in enumerate(midpoints):
                if x0 <= mx < x1 and y0 <= my < y1:
                    idxs.append(i)
            if not idxs:
                continue
            sub_angles = angles[idxs]
            sub_w = np.sum(counts_lengths[idxs])
            R_cell = mean_resultant_length(sub_angles)
            score_cell = 1.0 - R_cell
            cell_scores.append(score_cell * sub_w)
            weights.append(sub_w)
    if not weights:
        return 0.0
    return float(sum(cell_scores) / sum(weights))

def angle_diff(a, b):
    d = abs(a - b) % (2 * math.pi)
    if d > math.pi:
        d = 2 * math.pi - d
    return d

def seg_segment_min_dist(a1, a2, b1, b2):
    # Compute min distance between two line segments in 2D
    # Helper functions
    def dot(u, v):
        return u[0] * v[0] + u[1] * v[1]
    def norm2(u):
        return u[0] * u[0] + u[1] * u[1]
    def sub(u, v):
        return (u[0] - v[0], u[1] - v[1])
    def clamp(x, a, b):
        return max(a, min(b, x))
    u = sub(a2, a1)
    v = sub(b2, b1)
    w = sub(a1, b1)
    a = norm2(u)
    b = dot(u, v)
    c = norm2(v)
    d = dot(u, w)
    e = dot(v, w)
    D = a * c - b * b
    sc, sN, sD = 0.0, D, D
    tc, tN, tD = 0.0, D, D
    SMALL = 1e-9
    if D < SMALL:
        sN = 0.0
        sD = 1.0
        tN = e
        tD = c
    else:
        sN = (b * e - c * d)
        tN = (a * e - b * d)
        if sN < 0:
            sN = 0
            tN = e
            tD = c
        elif sN > sD:
            sN = sD
            tN = e + b
            tD = c
    if tN < 0:
        tN = 0
        if -d < 0:
            sN = 0
        elif -d > a:
            sN = sD
        else:
            sN = -d
            sD = a
    elif tN > tD:
        tN = tD
        if (-d + b) < 0:
            sN = 0
        elif (-d + b) > a:
            sN = sD
        else:
            sN = (-d + b)
            sD = a
    sc = 0.0 if abs(sN) < SMALL else sN / sD
    tc = 0.0 if abs(tN) < SMALL else tN / tD
    dP = (w[0] + sc * u[0] - tc * v[0], w[1] + sc * u[1] - tc * v[1])
    return math.hypot(dP[0], dP[1])

def pairwise_conflict_index(angles, lengths, segments, sigma=5.0):
    n = len(angles)
    if n < 2:
        return 0.0
    total = 0.0
    for i, j in combinations(range(n), 2):
        dtheta = angle_diff(angles[i], angles[j])
        f_dir = (1.0 - math.cos(dtheta)) / 2.0
        d_ij = seg_segment_min_dist(segments[i][0], segments[i][1], segments[j][0], segments[j][1])
        prox = math.exp(-d_ij / sigma)
        contrib = lengths[i] * lengths[j] * f_dir * prox
        total += contrib
    # Normalization: divide by sum(lengths)^2 / 2 as a rough upper bound
    denom = (lengths.sum() ** 2) / 2.0
    if denom <= 0:
        return 0.0
    return float(total / denom)

def compute_chaos_score(vectors, roi_bbox=None,
                        angle_bins=16, grid_size=4, sigma=5.0,
                        weights=(0.25, 0.20, 0.25, 0.30)):
    angles, lengths, midpoints, segments = compute_angles_and_lengths(vectors)
    if len(angles) == 0:
        return {
            "final_score": 0.0,
            "chaos_angle": 0.0,
            "entropy_index": 0.0,
            "local_mixture": 0.0,
            "conflict_index": 0.0
        }
    R = mean_resultant_length(angles)
    chaos_angle = 1.0 - R
    entropy_index = angular_entropy(angles, k_bins=angle_bins)
    counts_lengths = lengths  # Could use ones or lengths as weights per-vehicle
    local_mixture = grid_local_mixture(angles, midpoints, counts_lengths, roi_bbox, grid_size=grid_size, k_bins=angle_bins)
    conflict_index = pairwise_conflict_index(angles, lengths, segments, sigma=sigma)
    a, b, c, d = weights
    final = a * chaos_angle + b * entropy_index + c * local_mixture + d * conflict_index
    final = max(0.0, min(1.0, final))
    return {
        "final_score": final,
        "chaos_angle": chaos_angle,
        "entropy_index": entropy_index,
        "local_mixture": local_mixture,
        "conflict_index": conflict_index
    }
