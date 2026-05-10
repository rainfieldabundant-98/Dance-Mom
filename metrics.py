"""
Mathematical scoring engine for the dance comparison tool.
Compares teacher and student pose sequences and outputs quantitative scores.

Scoring dimensions (weighted):
  1. Joint Angle Similarity   (40%) — 8 key joint angles compared frame-by-frame
  2. Movement Amplitude Ratio (25%) — wrist/ankle displacement ratio
  3. Motion Smoothness        (20%) — jerk proxy via acceleration variance
  4. Lower Body Stiffness     (15%) — upper vs lower body movement ratio

All scores are normalised to 0–100.
"""

import numpy as np


# ==============================================================================
# Constants
# ==============================================================================

JOINT_TRIPLES: dict[str, tuple[int, int, int]] = {
    'left_elbow':    (11, 13, 15),   # shoulder – elbow – wrist
    'right_elbow':   (12, 14, 16),
    'left_shoulder': (13, 11, 23),   # elbow – shoulder – hip
    'right_shoulder':(14, 12, 24),
    'left_hip':      (11, 23, 25),   # shoulder – hip – knee
    'right_hip':     (12, 24, 26),
    'left_knee':     (23, 25, 27),   # hip – knee – ankle
    'right_knee':    (24, 26, 28),
}

# Landmark indices for amplitude analysis
WRIST_ANKLE_INDICES = [15, 16, 27, 28]   # wrists + ankles

# Landmark indices for smoothness analysis
SMOOTHNESS_INDICES = [15, 16, 25, 26, 27, 28]  # wrists, knees, ankles

# Upper / lower body groups for stiffness
UPPER_BODY_INDICES = [11, 12, 13, 14, 15, 16]  # shoulders, elbows, wrists
LOWER_BODY_INDICES = [23, 24, 25, 26, 27, 28]  # hips, knees, ankles

# Composite weights (must sum to 1.0)
W_JOINT_ANGLE = 0.40
W_AMPLITUDE   = 0.25
W_SMOOTHNESS  = 0.20
W_STIFFNESS   = 0.15


# ==============================================================================
# 1. Coordinate Normalisation
# ==============================================================================

def normalize_landmarks(landmarks_pixel: list) -> np.ndarray | None:
    """
    Normalise pixel landmark coordinates to a body-centric frame.

    Origin  = geometric centre of shoulders (11,12) + hips (23,24).
    Scale   = torso length (mid-shoulder to mid-hip Euclidean distance).
              If torso length is near-zero the raw coordinates are returned
              centred (unscaled) to avoid division-by-zero.

    Args:
        landmarks_pixel: list of 33 (x, y) tuples, or None.

    Returns:
        (33, 2) float32 numpy array of normalised coordinates, or None if
        input is None or torso length is invalid.
    """
    if landmarks_pixel is None:
        return None

    pts = np.array(landmarks_pixel, dtype=np.float32)  # (33, 2)

    # Origin: centre of shoulders + hips
    mid_shoulder = (pts[11] + pts[12]) / 2.0
    mid_hip      = (pts[23] + pts[24]) / 2.0
    origin       = (mid_shoulder + mid_hip) / 2.0

    # Centre all points
    centred = pts - origin  # (33, 2)

    # Torso length for scale
    torso_vec  = mid_shoulder - mid_hip
    torso_len  = float(np.linalg.norm(torso_vec))

    if torso_len < 1e-6:
        # Torso collapsed — return centred but unscaled
        return centred.astype(np.float32)

    # Scale so torso = 1 unit
    normalised = centred / torso_len
    return normalised.astype(np.float32)


# ==============================================================================
# 2. Valid-pair extraction
# ==============================================================================

def get_valid_pairs(teacher_data: list[dict],
                    student_data: list[dict]) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """
    Filter frames where BOTH teacher and student have valid landmarks.

    Args:
        teacher_data: list of per-frame dicts from video_processor.
        student_data: list of per-frame dicts from video_processor.

    Returns:
        List of (frame_idx, t_normalised, s_normalised) tuples.
        t_normalised and s_normalised are (33, 2) float32 arrays.
    """
    valid: list[tuple[int, np.ndarray, np.ndarray]] = []

    for i, (t_frame, s_frame) in enumerate(zip(teacher_data, student_data)):
        t_lm = t_frame.get('landmarks')
        s_lm = s_frame.get('landmarks')

        if t_lm is None or s_lm is None:
            continue

        t_pixel = t_lm.get('landmarks_pixel')
        s_pixel = s_lm.get('landmarks_pixel')

        if t_pixel is None or s_pixel is None:
            continue

        t_norm = normalize_landmarks(t_pixel)
        s_norm = normalize_landmarks(s_pixel)

        if t_norm is None or s_norm is None:
            continue

        valid.append((i, t_norm, s_norm))

    return valid


# ==============================================================================
# 3. Joint Angle Similarity (40 %)
# ==============================================================================

def compute_joint_angle(a: np.ndarray, b: np.ndarray,
                        c: np.ndarray) -> float:
    """
    Compute the interior angle ABC in degrees.  *b* is the vertex point.

    Args:
        a, b, c: (2,) or (D,) coordinate arrays.

    Returns:
        Angle in degrees [0, 180].
    """
    ba = a - b
    bc = c - b

    dot = float(np.dot(ba, bc))
    norm = float(np.linalg.norm(ba) * np.linalg.norm(bc))

    if norm < 1e-10:
        return 0.0

    # Clamp to [-1, 1] to avoid floating-point drift past acos domain
    cos_angle = max(-1.0, min(1.0, dot / norm))
    angle_rad = np.arccos(cos_angle)
    return float(np.degrees(angle_rad))


def joint_angle_similarity(
    valid_pairs: list[tuple[int, np.ndarray, np.ndarray]],
) -> dict:
    """
    Compare all 8 joint angles across every valid frame pair.

    Args:
        valid_pairs: output of get_valid_pairs().

    Returns:
        {
            'per_joint': {
                '<name>': {
                    'teacher_mean': float,   # degrees
                    'student_mean': float,   # degrees
                    'diff': float,           # abs difference
                    'similarity': float,     # 0–1, 1 = identical
                }, ...
            },
            'overall_angle_score': float,   # 0–100
        }
    """
    if not valid_pairs:
        return {
            'per_joint': {},
            'overall_angle_score': 0.0,
        }

    # Accumulate per-joint lists of (teacher_angle, student_angle)
    accum: dict[str, tuple[list[float], list[float]]] = {
        name: ([], []) for name in JOINT_TRIPLES
    }

    for _frame_idx, t_norm, s_norm in valid_pairs:
        for name, (i1, i2, i3) in JOINT_TRIPLES.items():
            t_angle = compute_joint_angle(t_norm[i1], t_norm[i2], t_norm[i3])
            s_angle = compute_joint_angle(s_norm[i1], s_norm[i2], s_norm[i3])
            accum[name][0].append(t_angle)
            accum[name][1].append(s_angle)

    per_joint: dict = {}
    similarities: list[float] = []

    for name, (t_list, s_list) in accum.items():
        t_mean = float(np.mean(t_list))
        s_mean = float(np.mean(s_list))

        # ---- Frame-by-frame comparison (NOT mean-vs-mean) ----
        per_frame_diffs = [abs(t - s) for t, s in zip(t_list, s_list)]
        avg_frame_diff = float(np.mean(per_frame_diffs))

        # Normalize by teacher's actual movement range (or 30deg minimum
        # for near-static poses).  This penalises wrong timing even when
        # the average angles happen to be similar.
        teacher_range = float(max(t_list) - min(t_list)) if len(t_list) >= 2 else 30.0
        denom = max(teacher_range, 30.0)

        # Similarity: 1 − normalised error, clamped to [0, 1]
        sim_raw = 1.0 - avg_frame_diff / denom
        sim = max(0.0, min(1.0, sim_raw))

        per_joint[name] = {
            'teacher_mean': round(t_mean, 2),
            'student_mean': round(s_mean, 2),
            'diff':         round(avg_frame_diff, 2),     # now per-frame avg diff
            'similarity':   round(sim, 4),
            'teacher_range': round(teacher_range, 2),     # for reference
        }
        similarities.append(sim)

    overall_angle_score = float(np.mean(similarities)) * 100.0

    return {
        'per_joint':           per_joint,
        'overall_angle_score': round(overall_angle_score, 2),
    }


# ==============================================================================
# 3b. Per-Frame Angle Data (for skeleton diff visualization)
# ==============================================================================

def get_per_frame_angle_data(
    valid_pairs: list[tuple[int, np.ndarray, np.ndarray]],
) -> dict:
    """
    Expose per-frame joint angles for both subjects — used by the skeleton
    diff grid so users can see WHICH specific poses match well vs poorly.

    Args:
        valid_pairs: output of get_valid_pairs().

    Returns:
        {
            'frame_indices': [0, 1, 3, 5, ...],         # original frame numbers
            'num_frames': int,
            'joints': {
                'left_elbow': {
                    'teacher': [t_angle_f0, t_angle_f1, ...],
                    'student': [s_angle_f0, s_angle_f1, ...],
                    'diff':    [abs_diff_f0, abs_diff_f1, ...],
                },
                ...
            },
            'per_frame_scores': [
                {
                    'frame_idx': int,
                    'joint_diffs': {'left_elbow': diff, ...},
                    'overall_similarity': float,    # 0-100
                    'best_joints': ['left_elbow', ...],     # smallest diffs
                    'worst_joints': ['right_knee', ...],    # largest diffs
                },
                ...
            ],
        }
    """
    result: dict = {
        'frame_indices': [],
        'num_frames': 0,
        'joints': {},
        'per_frame_scores': [],
    }

    if not valid_pairs:
        return result

    joint_names = list(JOINT_TRIPLES.keys())
    num_frames = len(valid_pairs)
    result['num_frames'] = num_frames

    # Init per-joint accumulators
    accum: dict[str, tuple[list[float], list[float], list[float]]] = {
        name: ([], [], []) for name in joint_names  # (teacher, student, diff)
    }

    for frame_idx, t_norm, s_norm in valid_pairs:
        result['frame_indices'].append(frame_idx)
        frame_joint_diffs: dict[str, float] = {}

        for name, (i1, i2, i3) in JOINT_TRIPLES.items():
            t_angle = compute_joint_angle(t_norm[i1], t_norm[i2], t_norm[i3])
            s_angle = compute_joint_angle(s_norm[i1], s_norm[i2], s_norm[i3])
            diff = abs(t_angle - s_angle)
            accum[name][0].append(t_angle)
            accum[name][1].append(s_angle)
            accum[name][2].append(diff)
            frame_joint_diffs[name] = diff

        # Per-frame overall similarity (inverse of mean diff, 0-100)
        diffs = list(frame_joint_diffs.values())
        avg_diff = sum(diffs) / len(diffs) if diffs else 0
        sim = max(0.0, min(100.0, 100.0 * (1.0 - avg_diff / 30.0)))

        # Sort joints by diff (smallest = best, largest = worst)
        sorted_joints = sorted(frame_joint_diffs.items(), key=lambda x: x[1])
        best_joints = [j for j, d in sorted_joints[:2] if d < 15]
        worst_joints = [j for j, d in sorted_joints[-2:] if d > 15]

        result['per_frame_scores'].append({
            'frame_idx': frame_idx,
            'joint_diffs': frame_joint_diffs,
            'overall_similarity': round(sim, 1),
            'best_joints': best_joints,
            'worst_joints': worst_joints,
        })

    # Build per-joint arrays
    for name in joint_names:
        result['joints'][name] = {
            'teacher': accum[name][0],
            'student': accum[name][1],
            'diff': accum[name][2],
        }

    return result


def pick_diff_keyframes(
    per_frame_data: dict,
    num_keyframes: int = 5,
) -> list[int]:
    """
    Pick evenly-spaced indices into the per-frame data, plus ensure the
    best-match and worst-match frames are included if not already covered.

    Args:
        per_frame_data: output of get_per_frame_angle_data().
        num_keyframes:  number of keyframes to return (default 5).

    Returns:
        List of indices into per_frame_data['per_frame_scores'].
    """
    scores = per_frame_data.get('per_frame_scores', [])
    n = len(scores)
    if n == 0:
        return []
    if n <= num_keyframes:
        return list(range(n))

    # Evenly spaced
    step = (n - 1) / (num_keyframes - 1)
    picked = {int(round(i * step)) for i in range(num_keyframes)}

    # Ensure best and worst frames are included
    sims = [s['overall_similarity'] for s in scores]
    best_idx = int(np.argmax(sims))
    worst_idx = int(np.argmin(sims))
    picked.add(best_idx)
    picked.add(worst_idx)

    return sorted(picked)


# ==============================================================================
# 4. Movement Amplitude Ratio (25 %)
# ==============================================================================

def amplitude_ratio(
    teacher_normalized: list[np.ndarray],
    student_normalized: list[np.ndarray],
) -> dict:
    """
    Compare movement amplitude for wrists and ankles.

    Amplitude = max Euclidean displacement from the mean position across all
    frames.  ratio = student / teacher.  1.0 = perfect match.

    score = 100 * max(0, 1 − |ratio − 1|)

    Args:
        teacher_normalized: list of (33, 2) normalised arrays per frame.
        student_normalized: list of (33, 2) normalised arrays per frame.

    Returns:
        {
            'per_joint': {
                'left_wrist':  {'teacher_amp': ..., 'student_amp': ...,
                                'ratio': ..., 'score': ...},
                ...  # left_wrist, right_wrist, left_ankle, right_ankle
            },
            'overall_amplitude_score': float,   # 0–100
        }
    """
    joint_names = ['left_wrist', 'right_wrist', 'left_ankle', 'right_ankle']

    if not teacher_normalized or not student_normalized:
        return {
            'per_joint': {},
            'overall_amplitude_score': 0.0,
        }

    t_stack = np.stack(teacher_normalized, axis=0)  # (F, 33, 2)
    s_stack = np.stack(student_normalized, axis=0)   # (F, 33, 2)

    per_joint: dict = {}
    scores: list[float] = []

    for idx, name in zip(WRIST_ANKLE_INDICES, joint_names):
        # Mean position across all frames
        t_mean_pos = np.mean(t_stack[:, idx, :], axis=0)  # (2,)
        s_mean_pos = np.mean(s_stack[:, idx, :], axis=0)  # (2,)

        # Max Euclidean displacement from mean
        t_disp = np.linalg.norm(t_stack[:, idx, :] - t_mean_pos, axis=1)
        s_disp = np.linalg.norm(s_stack[:, idx, :] - s_mean_pos, axis=1)

        t_amp = float(np.max(t_disp)) if len(t_disp) > 0 else 0.0
        s_amp = float(np.max(s_disp)) if len(s_disp) > 0 else 0.0

        ratio = s_amp / max(t_amp, 1e-8)
        # Score: symmetric penalty around 1.0
        score = 100.0 * max(0.0, 1.0 - abs(ratio - 1.0))

        per_joint[name] = {
            'teacher_amp': round(t_amp, 4),
            'student_amp': round(s_amp, 4),
            'ratio':       round(ratio, 4),
            'score':       round(score, 2),
        }
        scores.append(score)

    overall_amplitude_score = float(np.mean(scores)) if scores else 0.0

    return {
        'per_joint':               per_joint,
        'overall_amplitude_score': round(overall_amplitude_score, 2),
    }


# ==============================================================================
# 5. Motion Smoothness (20 %)
# ==============================================================================

def _sequence_smoothness(normalized_sequence: list[np.ndarray]) -> float:
    """
    Internal: compute smoothness for a single sequence.

    Smoothness = 1 / (1 + mean(|acceleration|))
    Higher value = smoother motion.

    Returns a float in (0, 1].
    """
    if len(normalized_sequence) < 3:
        return 1.0  # not enough frames to assess — treat as perfectly smooth

    stack = np.stack(normalized_sequence, axis=0)  # (F, 33, 2)

    # Only consider key points
    key_pts = stack[:, SMOOTHNESS_INDICES, :]  # (F, K, 2)

    # velocity[t] = position[t] - position[t-1]   for t >= 1
    velocity = np.diff(key_pts, axis=0)  # (F-1, K, 2)

    # acceleration[t] = velocity[t] - velocity[t-1]   for t >= 1
    acceleration = np.diff(velocity, axis=0)  # (F-2, K, 2)

    # Mean absolute acceleration across all key-points and axes
    mean_abs_accel = float(np.mean(np.abs(acceleration)))

    smoothness = 1.0 / (1.0 + mean_abs_accel)
    return smoothness


def motion_smoothness(normalized_sequence: list[np.ndarray]) -> dict:
    """
    Measure frame-to-frame velocity variance (jerk proxy) for one subject.

    Smoothness = 1 / (1 + mean(|acceleration|)) over key points
    (wrists, knees, ankles).

    Args:
        normalized_sequence: list of (33, 2) normalised arrays per frame.

    Returns:
        {'smoothness': float}  # 0–1, higher = smoother
    """
    s = _sequence_smoothness(normalized_sequence)
    return {'smoothness': round(s, 4)}


def motion_smoothness_comparison(
    teacher_normalized: list[np.ndarray],
    student_normalized: list[np.ndarray],
) -> dict:
    """
    Compare motion smoothness between teacher and student.

    Args:
        teacher_normalized: list of (33, 2) normalised arrays per frame.
        student_normalized: list of (33, 2) normalised arrays per frame.

    Returns:
        {
            'teacher_smoothness': float,   # 0–1
            'student_smoothness': float,   # 0–1
            'smoothness_score': float,     # 0–100  (student relative to teacher)
        }
    """
    t_smooth = _sequence_smoothness(teacher_normalized)
    s_smooth = _sequence_smoothness(student_normalized)

    # Score: how close student is to teacher
    if t_smooth > 1e-8:
        ratio = s_smooth / t_smooth
    else:
        ratio = 1.0

    smoothness_score = 100.0 * min(1.0, ratio)

    return {
        'teacher_smoothness': round(t_smooth, 4),
        'student_smoothness': round(s_smooth, 4),
        'smoothness_score':   round(smoothness_score, 2),
    }


# ==============================================================================
# 6. Lower Body Stiffness (15 %)
# ==============================================================================

def _sequence_stiffness(normalized_sequence: list[np.ndarray]) -> float:
    """
    Internal: compute lower-body stiffness ratio for a single sequence.

    stiffness = lower_variance / (upper_variance + lower_variance + ε)
    Low value = stiff legs (common beginner issue).

    Returns a float in [0, 1].
    """
    if len(normalized_sequence) < 2:
        return 0.5  # neutral default

    stack = np.stack(normalized_sequence, axis=0)  # (F, 33, 2)

    upper_pts = stack[:, UPPER_BODY_INDICES, :]  # (F, U, 2)
    lower_pts = stack[:, LOWER_BODY_INDICES, :]  # (F, L, 2)

    # Total variance across all frames for each group
    upper_var = float(np.var(upper_pts))
    lower_var = float(np.var(lower_pts))

    stiffness = lower_var / (upper_var + lower_var + 1e-8)
    return stiffness


def lower_body_stiffness(normalized_sequence: list[np.ndarray]) -> dict:
    """
    Measure lower-body stiffness ratio for one subject.

    Ratio of lower-body to upper-body movement variance.
    Low value = stiff legs (common beginner issue).

    Args:
        normalized_sequence: list of (33, 2) normalised arrays per frame.

    Returns:
        {'stiffness': float}  # 0–1
    """
    s = _sequence_stiffness(normalized_sequence)
    return {'stiffness': round(s, 4)}


def lower_body_stiffness_comparison(
    teacher_normalized: list[np.ndarray],
    student_normalized: list[np.ndarray],
) -> dict:
    """
    Compare lower-body stiffness between teacher and student.

    Args:
        teacher_normalized: list of (33, 2) normalised arrays per frame.
        student_normalized: list of (33, 2) normalised arrays per frame.

    Returns:
        {
            'teacher_stiffness': float,   # 0–1
            'student_stiffness': float,   # 0–1
            'stiffness_score': float,     # 0–100  (student relative to teacher)
        }
    """
    t_stiff = _sequence_stiffness(teacher_normalized)
    s_stiff = _sequence_stiffness(student_normalized)

    # Score: higher stiffness = better (more leg movement).  Student is
    # penalised if they are stiffer (lower value) than the teacher.
    if t_stiff > 1e-8:
        ratio = s_stiff / t_stiff
    else:
        ratio = 1.0

    stiffness_score = 100.0 * min(1.0, ratio)

    return {
        'teacher_stiffness': round(t_stiff, 4),
        'student_stiffness': round(s_stiff, 4),
        'stiffness_score':   round(stiffness_score, 2),
    }


# ==============================================================================
# 7. Master Function
# ==============================================================================

def compute_all_metrics(processed_data: dict) -> dict:
    """
    Master scoring function.

    Processes all frames, normalises coordinates, and computes every metric.

    Args:
        processed_data: output of video_processor.process_video_pair().
            Expected keys: 'teacher', 'student', 'num_frames'.

    Returns:
        {
            'joint_angles':     {...},   # from joint_angle_similarity
            'amplitude':        {...},   # from amplitude_ratio
            'smoothness':       {...},   # from motion_smoothness_comparison
            'stiffness':        {...},   # from lower_body_stiffness_comparison
            'overall_score':    float,   # 0–100 weighted composite
            'num_valid_frames': int,
            'num_total_frames': int,
            'breakdown': {
                'joint_angle_similarity': float,   # 0–100  (40 %)
                'amplitude_ratio':        float,   # 0–100  (25 %)
                'motion_smoothness':      float,   # 0–100  (20 %)
                'lower_body_stiffness':   float,   # 0–100  (15 %)
            },
        }
    """
    teacher_data = processed_data['teacher']
    student_data = processed_data['student']
    num_total    = processed_data['num_frames']

    print("=" * 60)
    print("Computing dance metrics ...")

    # ---- 7a. Extract valid frame pairs ---------------------------------------
    print("  Extracting valid frame pairs ...")
    valid_pairs = get_valid_pairs(teacher_data, student_data)
    num_valid = len(valid_pairs)
    print(f"  Valid frames: {num_valid}/{num_total}")

    if num_valid == 0:
        print("  [WARN] No valid frame pairs — returning zero scores.")
        return {
            'joint_angles':     {'per_joint': {}, 'overall_angle_score': 0.0},
            'amplitude':        {'per_joint': {}, 'overall_amplitude_score': 0.0},
            'smoothness':       {'teacher_smoothness': 0.0, 'student_smoothness': 0.0,
                                 'smoothness_score': 0.0},
            'stiffness':        {'teacher_stiffness': 0.0, 'student_stiffness': 0.0,
                                 'stiffness_score': 0.0},
            'per_frame_angles': {'frame_indices': [], 'num_frames': 0,
                                'joints': {}, 'per_frame_scores': []},
            'overall_score':    0.0,
            'num_valid_frames': 0,
            'num_total_frames': num_total,
            'breakdown': {
                'joint_angle_similarity': 0.0,
                'amplitude_ratio':        0.0,
                'motion_smoothness':      0.0,
                'lower_body_stiffness':   0.0,
            },
        }

    # ---- 7b. Build per-subject normalised sequences --------------------------
    t_normalized: list[np.ndarray] = [t for (_, t, _) in valid_pairs]
    s_normalized: list[np.ndarray] = [s for (_, _, s) in valid_pairs]

    # ---- 7c. Joint angle similarity (40 %) -----------------------------------
    print("  Computing joint angle similarity ...")
    joint_angles = joint_angle_similarity(valid_pairs)
    angle_score  = joint_angles['overall_angle_score']
    print(f"    Angle score: {angle_score:.1f}/100")

    # ---- 7d. Amplitude ratio (25 %) ------------------------------------------
    print("  Computing movement amplitude ratio ...")
    amplitude = amplitude_ratio(t_normalized, s_normalized)
    amp_score = amplitude['overall_amplitude_score']
    print(f"    Amplitude score: {amp_score:.1f}/100")

    # ---- 7e. Motion smoothness (20 %) ----------------------------------------
    print("  Computing motion smoothness ...")
    smoothness = motion_smoothness_comparison(t_normalized, s_normalized)
    smooth_score = smoothness['smoothness_score']
    print(f"    Smoothness score: {smooth_score:.1f}/100  "
          f"(teacher={smoothness['teacher_smoothness']:.3f}, "
          f"student={smoothness['student_smoothness']:.3f})")

    # ---- 7f. Lower body stiffness (15 %) -------------------------------------
    print("  Computing lower body stiffness ...")
    stiffness = lower_body_stiffness_comparison(t_normalized, s_normalized)
    stiff_score = stiffness['stiffness_score']
    print(f"    Stiffness score: {stiff_score:.1f}/100  "
          f"(teacher={stiffness['teacher_stiffness']:.3f}, "
          f"student={stiffness['student_stiffness']:.3f})")

    # ---- 7g. Per-frame angle data (for skeleton diff grid) -----------------
    print("  Computing per-frame angle data ...")
    per_frame_angles = get_per_frame_angle_data(valid_pairs)
    print(f"    {per_frame_angles['num_frames']} frames with angle data")

    # ---- 7h. Weighted composite ----------------------------------------------
    overall = (
        W_JOINT_ANGLE * angle_score +
        W_AMPLITUDE   * amp_score +
        W_SMOOTHNESS  * smooth_score +
        W_STIFFNESS   * stiff_score
    )

    print(f"\n  Overall Score: {overall:.1f}/100")

    return {
        'joint_angles':     joint_angles,
        'amplitude':        amplitude,
        'smoothness':       smoothness,
        'stiffness':        stiffness,
        'per_frame_angles': per_frame_angles,
        'overall_score':    round(overall, 2),
        'num_valid_frames': num_valid,
        'num_total_frames': num_total,
        'breakdown': {
            'joint_angle_similarity': angle_score,
            'amplitude_ratio':        amp_score,
            'motion_smoothness':      smooth_score,
            'lower_body_stiffness':   stiff_score,
        },
    }


# ==============================================================================
# 8. Main — self-contained test
# ==============================================================================

if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')
    from video_processor import process_video_pair

    data = process_video_pair("1teacher.mp4", "1me.mp4", num_frames=30)
    results = compute_all_metrics(data)

    print("\n" + "=" * 60)
    print("DANCE ANALYSIS — Quantitative Scores")
    print("=" * 60)
    print(f"Valid frames: {results['num_valid_frames']}/{results['num_total_frames']}")
    print(f"\nOverall Score: {results['overall_score']:.1f}/100")
    print(f"\nBreakdown:")
    for name, score in results['breakdown'].items():
        print(f"  {name}: {score:.1f}/100")

    if results['joint_angles']['per_joint']:
        print(f"\nJoint Angle Details:")
        for joint, info in results['joint_angles']['per_joint'].items():
            print(f"  {joint}: teacher={info['teacher_mean']:.1f}deg "
                  f"student={info['student_mean']:.1f}deg "
                  f"diff={info['diff']:.1f}deg  "
                  f"sim={info['similarity']:.3f}")

    if results['amplitude']['per_joint']:
        print(f"\nAmplitude Details:")
        for joint, info in results['amplitude']['per_joint'].items():
            print(f"  {joint}: t_amp={info['teacher_amp']:.4f} "
                  f"s_amp={info['student_amp']:.4f} "
                  f"ratio={info['ratio']:.3f} "
                  f"score={info['score']:.1f}")
