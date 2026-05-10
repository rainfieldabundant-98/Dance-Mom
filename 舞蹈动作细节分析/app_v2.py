"""
Gradio web interface v2 — 增强版舞蹈动作分析工具

基于 app.py 改进：
  - 骨架对比网格：5帧关键帧，逐帧展示关节角度差异（绿/黄/红颜色编码）
  - 自动高亮最佳匹配帧和最需练习帧
  - 保持两Tab结构：视频对比分析 / 单图骨骼测试
  - 自定义 CSS（gold/rose/teal 色板）
"""

import os
import sys
import io
import base64
import time
import traceback
import cv2
import numpy as np
import gradio as gr
from datetime import datetime

from pose_detector import PoseDetector, POSE_CONNECTIONS
from video_processor import process_video_pair, generate_comparison_video
from metrics import compute_all_metrics, pick_diff_keyframes, JOINT_TRIPLES
from llm_evaluator import generate_coaching_report


OUTPUT_DIR = "outputs"

# ---------------------------------------------------------------------------
# CSS design tokens (from kuaijuan-mama-v2.html)
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
:root {
  --gold: #C9A84C; --gold-l: #F5EDD6; --gold-d: #8B6914;
  --rose: #C0515A; --rose-l: #FAF0F0; --rose-d: #7D2530;
  --teal: #2A7B6F; --teal-l: #E8F4F2;
  --ink: #1A1410; --ink-s: #4A3F35; --muted: #9A8F85;
  --bg: #FFFDF8; --bg2: #FAF6EE; --bdr: rgba(201,168,76,.2);
}
footer { display: none !important; }
"""

# Joint name mapping for display
JOINT_NAMES_CN = {
    'left_elbow': '左肘', 'right_elbow': '右肘',
    'left_shoulder': '左肩', 'right_shoulder': '右肩',
    'left_hip': '左髋', 'right_hip': '右髋',
    'left_knee': '左膝', 'right_knee': '右膝',
}

# Landmark indices for major joints (label-worthy)
KEY_JOINT_INDICES = {
    'left_elbow': 13, 'right_elbow': 14,
    'left_shoulder': 11, 'right_shoulder': 12,
    'left_hip': 23, 'right_hip': 24,
    'left_knee': 25, 'right_knee': 26,
}


# ==============================================================================
# Video frame preview helper
# ==============================================================================

def extract_frame_at_offset(video_path: str | None, offset: int) -> np.ndarray | None:
    """Extract a single frame from video at given frame offset. Returns BGR image."""
    if video_path is None:
        return None
    try:
        cap = cv2.VideoCapture(video_path)
        # Jump to offset frame
        offset = max(0, offset)
        cap.set(cv2.CAP_PROP_POS_FRAMES, offset)
        ret, frame = cap.read()
        cap.release()
        if ret:
            return frame
        # Fallback: try reading from start
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        return frame if ret else None
    except Exception:
        return None


def preview_offset_frame(student_video: str | None, offset: int) -> np.ndarray | None:
    """Extract frame at offset, convert to RGB for Gradio Image display."""
    frame = extract_frame_at_offset(student_video, offset)
    if frame is None:
        return None
    h, w = frame.shape[:2]
    # Resize to a reasonable preview size while keeping aspect ratio
    max_w = 480
    if w > max_w:
        scale = max_w / w
        new_w = int(w * scale)
        new_h = int(h * scale)
        frame = cv2.resize(frame, (new_w, new_h))
    cv2.putText(frame, f"Offset: {offset} frames", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


# ==============================================================================
# Joint color mapping
# ==============================================================================

def joint_diff_color(diff_deg: float) -> tuple[int, int, int]:
    """Map angle difference to BGR color: green (good) -> yellow -> red (bad)."""
    if diff_deg < 15:
        return (0, 180, 80)    # green BGR
    elif diff_deg < 30:
        return (0, 180, 255)   # yellow BGR
    else:
        return (60, 80, 240)   # red BGR


def joint_diff_hex(diff_deg: float) -> str:
    """Map angle difference to hex color for HTML."""
    if diff_deg < 15:
        return "#22c55e"
    elif diff_deg < 30:
        return "#f59e0b"
    else:
        return "#ef4444"


# ==============================================================================
# Skeleton comparison rendering — side-by-side, body-aligned & normalized
# ==============================================================================

def _body_center(lm_pixel: list) -> tuple[float, float] | None:
    """Compute upper-body centre = midpoint of shoulders (landmarks 11 & 12)."""
    if lm_pixel is None:
        return None
    x11, y11 = lm_pixel[11]
    x12, y12 = lm_pixel[12]
    return ((x11 + x12) / 2.0, (y11 + y12) / 2.0)


def _torso_length(lm_pixel: list) -> float:
    """Distance between shoulder midpoint and hip midpoint (landmarks 23,24)."""
    if lm_pixel is None:
        return 1.0
    mx_shoulder = (lm_pixel[11][0] + lm_pixel[12][0]) / 2.0
    my_shoulder = (lm_pixel[11][1] + lm_pixel[12][1]) / 2.0
    mx_hip = (lm_pixel[23][0] + lm_pixel[24][0]) / 2.0
    my_hip = (lm_pixel[23][1] + lm_pixel[24][1]) / 2.0
    return max(1.0, np.sqrt((mx_shoulder - mx_hip) ** 2 + (my_shoulder - my_hip) ** 2))


def _scale_frame_and_landmarks(
    frame: np.ndarray,
    lm_pixel: list | None,
    scale: float,
) -> tuple[np.ndarray, list | None]:
    """Scale frame and landmark coordinates by *scale*. Landmarks stay native."""
    if scale == 1.0:
        return frame, lm_pixel
    h, w = frame.shape[:2]
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    new_w = max(1, new_w)
    new_h = max(1, new_h)
    scaled_frame = cv2.resize(frame, (new_w, new_h))
    if lm_pixel is not None:
        scaled_lm = [(int(round(x * scale)), int(round(y * scale))) for x, y in lm_pixel]
    else:
        scaled_lm = None
    return scaled_frame, scaled_lm


def draw_skeleton_on_frame(
    frame: np.ndarray,
    lm_pixel: list | None,
    line_color: tuple[int, int, int],
    point_color: tuple[int, int, int],
    joint_diffs: dict[str, float] | None = None,
    thickness: int = 2,
    point_radius: int = 4,
) -> None:
    """Draw a single skeleton on *frame* (modifies in-place). Joints are color-coded if diffs given."""
    if lm_pixel is None:
        return

    for start, end in POSE_CONNECTIONS:
        x1, y1 = lm_pixel[start]
        x2, y2 = lm_pixel[end]
        cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), line_color, thickness)

    for idx, (x, y) in enumerate(lm_pixel):
        jname = None
        for name, lm_idx in KEY_JOINT_INDICES.items():
            if lm_idx == idx:
                jname = name
                break
        if jname and joint_diffs and jname in joint_diffs:
            color = joint_diff_color(joint_diffs[jname])
            r = point_radius + 2
        else:
            color = point_color
            r = point_radius
        cv2.circle(frame, (int(x), int(y)), r, color, -1)


def draw_skeleton_comparison_side_by_side(
    teacher_frame: np.ndarray,
    student_frame: np.ndarray,
    teacher_lm_pixel: list | None,
    student_lm_pixel: list | None,
    joint_diffs: dict[str, float] | None,
) -> np.ndarray:
    """
    Side-by-side skeleton comparison with body-size normalization.

    1. Scale student frame+landmarks so torso length matches teacher
    2. Place both frames on a combined canvas so body centres (shoulder midpoint)
       are at the same vertical position
    3. Draw skeletons at NATIVE positions on each sub-image
       → skeleton always matches the person in the frame

    Returns: [teacher | student] combined BGR image.
    """
    GAP = 4               # pixels between the two panels
    LABEL_H = 36          # reserved space at top for labels

    t_h, t_w = teacher_frame.shape[:2]
    s_h, s_w = student_frame.shape[:2]

    # ---- 1. Compute body metrics -----------------------------------------
    t_center = _body_center(teacher_lm_pixel)
    s_center = _body_center(student_lm_pixel)
    t_torso = _torso_length(teacher_lm_pixel)
    s_torso = _torso_length(student_lm_pixel)

    # Default centres if detection missed
    if t_center is None:
        t_center = (t_w / 2.0, t_h / 2.0)
    if s_center is None:
        s_center = (s_w / 2.0, s_h / 2.0)

    # ---- 2. Scale student to match teacher's torso length -----------------
    scale = t_torso / s_torso if s_torso > 0 else 1.0
    scale = max(0.5, min(2.0, scale))  # clamp to reasonable range
    s_frame_scaled, s_lm_scaled = _scale_frame_and_landmarks(
        student_frame, student_lm_pixel, scale)
    s_h_s, s_w_s = s_frame_scaled.shape[:2]
    s_center_s = (s_center[0] * scale, s_center[1] * scale)

    # ---- 3. Layout: centre both body centres at the same y on canvas -----
    body_y = int(max(t_center[1], s_center_s[1]) + LABEL_H)

    # Teacher placement
    t_x = 0
    t_y = body_y - int(t_center[1])

    # Student placement (to the right of teacher)
    s_x = t_w + GAP
    s_y = body_y - int(s_center_s[1])

    # Canvas size
    canvas_w = t_w + GAP + s_w_s
    # We need enough space above and below
    canvas_h = max(t_y + t_h, s_y + s_h_s, body_y + max(t_h - t_center[1], s_h_s - s_center_s[1]))
    # Make canvas_h at least body_y + some padding
    canvas_h = max(canvas_h, body_y + max(t_h - int(t_center[1]), s_h_s - int(s_center_s[1])) + 20)

    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    # ---- 4. Paste frames & draw skeletons at NATIVE positions ------------

    # Teacher
    ty0, ty1 = max(0, t_y), min(canvas_h, t_y + t_h)
    fy0, fy1 = max(0, -t_y), min(t_h, canvas_h - t_y)
    if fy1 > fy0 and ty1 > ty0:
        canvas[ty0:ty1, 0:t_w] = teacher_frame[fy0:fy1, 0:t_w]
    draw_skeleton_on_frame(
        canvas,  # whole canvas; we restrict coords via landmarks being native
        teacher_lm_pixel,
        line_color=(200, 120, 0),
        point_color=(200, 120, 0),
        thickness=2, point_radius=4,
    )
    # But wait — draw_skeleton_on_frame draws on the canvas at native landmark coords.
    # We need to offset teacher landmarks by (t_x, t_y) since the frame was placed shifted.
    # Actually we can draw on the frame BEFORE pasting. That's cleaner.
    # Let me fix: draw on copies before placing.

    # Redo more cleanly:
    # For each side: copy the frame, draw skeleton on it, then paste into canvas.

    del canvas  # will recreate

    # ---- Teacher panel ---------------------------------------------------
    t_panel = teacher_frame.copy()
    draw_skeleton_on_frame(t_panel, teacher_lm_pixel,
                           line_color=(200, 120, 0), point_color=(200, 120, 0),
                           thickness=2, point_radius=4)

    # ---- Student panel (scaled) ------------------------------------------
    s_panel = s_frame_scaled.copy()
    draw_skeleton_on_frame(s_panel, s_lm_scaled,
                           line_color=(80, 60, 220), point_color=(80, 60, 220),
                           joint_diffs=joint_diffs,
                           thickness=3, point_radius=4)

    # ---- Label panels ----------------------------------------------------
    cv2.putText(t_panel, "Teacher", (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 120, 0), 2)
    cv2.putText(s_panel, "Student", (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (80, 60, 220), 2)

    # ---- Build combined canvas -------------------------------------------
    t_h_p, t_w_p = t_panel.shape[:2]
    s_h_p, s_w_p = s_frame_scaled.shape[:2]

    canvas_w = t_w_p + GAP + s_w_p
    # Enough space above and below the body centres
    t_below = t_h_p - int(t_center[1])
    s_below = s_h_p - int(s_center_s[1])
    canvas_h = max(t_y + t_h_p, s_y + s_h_p, body_y + max(t_below, s_below) + 20)

    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    # Paste teacher panel
    ty0, ty1 = max(0, t_y), min(canvas_h, t_y + t_h_p)
    fy0, fy1 = max(0, -t_y), min(t_h_p, canvas_h - t_y)
    if fy1 > fy0 and ty1 > ty0:
        canvas[ty0:ty1, 0:t_w_p] = t_panel[fy0:fy1, :]

    # Paste student panel
    sy0, sy1 = max(0, s_y), min(canvas_h, s_y + s_h_p)
    sfy0, sfy1 = max(0, -s_y), min(s_h_p, canvas_h - s_y)
    if sfy1 > sfy0 and sy1 > sy0:
        canvas[sy0:sy1, s_x:s_x + s_w_p] = s_panel[sfy0:sfy1, :]

    # Divider line
    cv2.line(canvas, (t_w_p + GAP // 2, 0), (t_w_p + GAP // 2, canvas_h), (100, 100, 100), 1)

    return canvas


def frame_to_base64(img: np.ndarray, quality: int = 85) -> str:
    """Encode BGR image to base64 JPEG string."""
    _, buf = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return base64.b64encode(buf).decode('utf-8')


def generate_diff_grid(
    processed_data: dict,
    per_frame_data: dict,
    num_frames: int = 5,
) -> list[dict]:
    """
    Generate skeleton diff frames for the grid display.

    Returns list of dicts, each with:
        frame_idx, image_b64, score, best_joints, worst_joints, joint_diffs
    """
    teacher_data = processed_data['teacher']
    student_data = processed_data['student']
    scores = per_frame_data.get('per_frame_scores', [])

    if not scores:
        return []

    picked_indices = pick_diff_keyframes(per_frame_data, num_frames)
    if not picked_indices:
        return []

    results = []
    for data_idx in picked_indices:
        score_info = scores[data_idx]
        orig_frame_idx = score_info['frame_idx']

        t_entry = teacher_data[orig_frame_idx]
        s_entry = student_data[orig_frame_idx]

        t_frame = t_entry['frame']
        s_frame = s_entry['frame']

        t_lm = t_entry.get('landmarks')
        s_lm = s_entry.get('landmarks')
        t_pixel = t_lm.get('landmarks_pixel') if t_lm else None
        s_pixel = s_lm.get('landmarks_pixel') if s_lm else None

        # Side-by-side comparison with body-origin alignment
        annotated = draw_skeleton_comparison_side_by_side(
            t_frame, s_frame, t_pixel, s_pixel, score_info.get('joint_diffs')
        )

        # Add frame score at top
        cv2.putText(annotated, f"Frame {orig_frame_idx + 1}  |  Score: {score_info['overall_similarity']:.0f}",
                    (10, int(annotated.shape[0]) - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        results.append({
            'frame_idx': orig_frame_idx,
            'data_idx': data_idx,
            'image_b64': frame_to_base64(annotated),
            'score': score_info['overall_similarity'],
            'joint_diffs': score_info.get('joint_diffs', {}),
            'best_joints': score_info.get('best_joints', []),
            'worst_joints': score_info.get('worst_joints', []),
        })

    return results


# ==============================================================================
# HTML builders
# ==============================================================================

def get_grade(score: float) -> str:
    if score >= 90:
        return "优秀"
    elif score >= 80:
        return "良好"
    elif score >= 70:
        return "有待提高"
    else:
        return "需要练习"


def build_overall_joint_analysis(per_frame_data: dict) -> str:
    """
    Aggregate per-frame joint angle data across ALL frames.
    Shows which joints are consistently problematic vs consistently good.
    Returns HTML table ranked by improvement priority.
    """
    joints = per_frame_data.get('joints', {})
    if not joints:
        return ""

    summary: list[dict] = []
    for jname, data in joints.items():
        diffs = data.get('diff', [])
        if not diffs:
            continue
        mean_diff = float(np.mean(diffs))
        std_diff = float(np.std(diffs))
        # Consistency: lower std = more consistent problem (or consistent match)
        # Priority score: high mean + low std = urgent consistent problem
        #                high mean + high std = sometimes good sometimes bad
        # Sort by mean_diff descending
        cn = JOINT_NAMES_CN.get(jname, jname)

        # Per-frame similarity percentage
        avg_sim = max(15.0, 100.0 * (1.0 - mean_diff / 45.0))

        summary.append({
            'joint': cn,
            'mean_diff': mean_diff,
            'std_diff': std_diff,
            'similarity': avg_sim,
            'consistency': "稳定偏差" if std_diff < 8 else ("波动偏差" if std_diff < 15 else "不稳定"),
        })

    # Sort by mean_diff descending (largest diff = highest priority)
    summary.sort(key=lambda x: x['mean_diff'], reverse=True)

    rows = ""
    for i, item in enumerate(summary):
        jc = joint_diff_hex(item['mean_diff'])
        priority_icon = {0: "🥇", 1: "🥈", 2: "🥉"}.get(i, f"{i + 1}")
        bar_pct = min(100, item['mean_diff'] / 45.0 * 100)
        consistency_color = {
            "稳定偏差": "#ef4444",
            "波动偏差": "#f59e0b",
            "不稳定": "#3b82f6",
        }.get(item['consistency'], "#9ca3af")

        rows += f"""
        <tr>
            <td style="padding:10px 12px; text-align:center; font-weight:700; color:var(--ink);">
                {priority_icon}
            </td>
            <td style="padding:10px 12px; font-weight:600; color:var(--ink-s);">{item['joint']}</td>
            <td style="padding:10px 12px; color:{jc}; font-weight:700;">{item['mean_diff']:.0f}°</td>
            <td style="padding:10px 12px; color:var(--muted);">{item['std_diff']:.0f}°</td>
            <td style="padding:10px 12px;">
                <span style="display:inline-block; padding:2px 10px; border-radius:12px;
                       font-size:11px; font-weight:700; color:#fff; background:{consistency_color};">
                    {item['consistency']}</span>
            </td>
            <td style="padding:10px 12px;">
                <div style="background:#f3f4f6; border-radius:6px; height:6px; width:120px;">
                    <div style="background:{jc}; border-radius:6px; height:6px; width:{bar_pct:.0f}%;"></div>
                </div>
            </td>
        </tr>"""

    # Find top 3 focus areas and good areas
    top3 = summary[:3]
    bottom3 = summary[-3:]

    focus_text = " · ".join(f"**{s['joint']}**（均差 {s['mean_diff']:.0f}°）" for s in top3)
    good_text = " · ".join(f"**{s['joint']}**（均差仅 {s['mean_diff']:.0f}°）" for s in reversed(bottom3))

    return f"""
    <div style="font-family: -apple-system, 'Microsoft YaHei', sans-serif; max-width: 900px; margin: 0 auto;">
    <div style="background:#fff; border-radius:14px; padding:24px; margin:20px 0;
                box-shadow:0 2px 8px rgba(0,0,0,.06);">
        <h3 style="margin:0 0 4px 0; font-size:17px; color:var(--ink);">整体舞蹈分析 — 加强重点</h3>
        <p style="font-size:13px; color:var(--muted); margin:0 0 6px 0;">
            综合所有 {per_frame_data.get('num_frames', 0)} 帧数据，按角度差异从大到小排列。稳定偏差 = 始终偏大/偏小，波动偏差 = 时好时坏。
        </p>

        <div style="background:var(--rose-l); border:1px solid rgba(192,81,90,.2); border-radius:10px;
                    padding:14px 16px; margin-bottom:16px; font-size:14px; color:var(--rose-d); line-height:1.7;">
            <strong>需要重点加强：</strong>{focus_text}
        </div>
        <div style="background:var(--teal-l); border:1px solid rgba(42,123,111,.2); border-radius:10px;
                    padding:14px 16px; margin-bottom:16px; font-size:14px; color:#1A5A52; line-height:1.7;">
            <strong>已经做得很好的：</strong>{good_text}
        </div>

        <table style="width:100%; border-collapse:collapse; font-size:14px;">
            <thead>
                <tr style="color:var(--muted); font-size:12px; text-align:left; border-bottom:2px solid #f3f4f6;">
                    <th style="padding:8px 12px; width:40px;">优先级</th>
                    <th style="padding:8px 12px;">关节</th>
                    <th style="padding:8px 12px;">平均差异</th>
                    <th style="padding:8px 12px;">波动</th>
                    <th style="padding:8px 12px;">偏差类型</th>
                    <th style="padding:8px 12px; width:140px;">差异程度</th>
                </tr>
            </thead>
            <tbody>{rows}</tbody>
        </table>
    </div>
    </div>"""


def build_skeleton_diff_html(diff_grid: list[dict], page: int = 0,
                              per_page: int = 3) -> str:
    """
    Build paginated skeleton comparison grid HTML.
    Shows *per_page* frames at a time (default 3), with page navigation info.
    """
    if not diff_grid:
        return '<div style="color:var(--muted);padding:20px;">暂无逐帧对比数据</div>'

    total = len(diff_grid)
    total_pages = max(1, (total + per_page - 1) // per_page)
    page = max(0, min(page, total_pages - 1))
    start_idx = page * per_page
    page_items = diff_grid[start_idx:start_idx + per_page]

    # Find global best/worst for badges
    best = max(diff_grid, key=lambda x: x['score'])
    worst = min(diff_grid, key=lambda x: x['score'])

    frame_cards = ""
    for item in page_items:
        best_j = ", ".join(JOINT_NAMES_CN.get(j, j) for j in item['best_joints']) or "—"
        worst_j = ", ".join(JOINT_NAMES_CN.get(j, j) for j in item['worst_joints']) or "—"
        score = item['score']
        sc = joint_diff_hex(30.0 * (1.0 - score / 100.0))

        border_style = ""
        badge = ""
        if item['data_idx'] == best['data_idx']:
            border_style = 'border: 2px solid #22c55e; box-shadow: 0 0 12px rgba(34,197,94,.25);'
            badge = '<div style="background:#22c55e;color:#fff;font-size:11px;padding:3px 10px;border-radius:10px;margin-bottom:6px;display:inline-block;">&#11088; 最佳匹配</div>'
        elif item['data_idx'] == worst['data_idx']:
            border_style = 'border: 2px solid #ef4444; box-shadow: 0 0 12px rgba(239,68,68,.20);'
            badge = '<div style="background:#ef4444;color:#fff;font-size:11px;padding:3px 10px;border-radius:10px;margin-bottom:6px;display:inline-block;">&#127919; 最需练习</div>'

        frame_cards += f"""
        <div style="flex:0 0 calc(33.333% - 10px); min-width:260px; {border_style}
                    border-radius:12px; overflow:hidden; background:#fff;
                    box-shadow:0 2px 8px rgba(0,0,0,.06); text-align:center;">
            <div style="padding:10px 8px 6px;">
                {badge}
                <div style="font-size:14px; font-weight:600; color:var(--ink-s);">
                    第 {item['frame_idx'] + 1} 帧
                    <span style="font-size:11px; color:var(--muted);">（{item['frame_idx'] + 1}/{diff_grid[-1]['frame_idx'] + 1}）</span>
                </div>
            </div>
            <img src="data:image/jpeg;base64,{item['image_b64']}"
                 style="width:100%; display:block;" alt="Frame {item['frame_idx'] + 1}">
            <div style="padding:12px 10px;">
                <div style="font-size:28px; font-weight:800; color:{sc};">{score:.0f}
                    <span style="font-size:12px;color:var(--muted);"> 分</span></div>
                <div style="font-size:12px; color:var(--muted); margin-top:4px;">
                    <span style="color:#22c55e;">匹配: {best_j}</span>
                </div>
                <div style="font-size:12px; color:var(--muted);">
                    <span style="color:#ef4444;">注意: {worst_j}</span>
                </div>
            </div>
        </div>"""

    # Page indicator dots
    dots = ""
    for i in range(total_pages):
        if i == page:
            dots += '<span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:var(--gold);margin:0 3px;"></span>'
        else:
            dots += '<span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:#ddd;margin:0 3px;"></span>'

    return f"""
    <div style="font-family: -apple-system, 'Microsoft YaHei', sans-serif; max-width: 900px; margin: 0 auto;">

    <div style="background:#fff; border-radius:14px; padding:24px; margin:20px 0;
                box-shadow:0 2px 8px rgba(0,0,0,.06);">
        <h3 style="margin:0 0 4px 0; font-size:17px; color:var(--ink);">&#129656; 逐帧骨骼对比</h3>
        <p style="font-size:13px; color:var(--muted); margin:0 0 16px 0;">
            左=老师(蓝) 右=学生(红) · 身体原点已对齐 · 关节颜色 <span style="color:#22c55e;">●绿</span><span style="color:#f59e0b;">●黄</span><span style="color:#ef4444;">●红</span> = 差异由小到大
            &nbsp;|&nbsp; 共 {total} 帧，第 {page + 1}/{total_pages} 页
        </p>

        <div style="display:flex; gap:12px; padding:4px 0;">
            {frame_cards}
        </div>

        <div style="text-align:center; margin-top:16px;">
            {dots}
        </div>
        <div style="text-align:center; font-size:11px; color:var(--muted); margin-top:4px;">
            &#8592; 拖动下方滑动条浏览更多帧 &#8594;
        </div>
    </div>
    </div>"""


def build_score_cards_html(metrics: dict) -> str:
    """Improved score cards with visual design (replaces old metrics table)."""
    overall = metrics.get('overall_score', 0)
    grade = get_grade(overall)
    breakdown = metrics.get('breakdown', {})

    grade_colors = {
        "优秀": ("#22c55e", "#16a34a"),
        "良好": ("#3b82f6", "#2563eb"),
        "有待提高": ("#f59e0b", "#d97706"),
        "需要练习": ("#ef4444", "#dc2626"),
    }
    c_main, c_dark = grade_colors.get(grade, grade_colors["良好"])

    dims = [
        ("关节角度相似度", breakdown.get('joint_angle_similarity', 0), "各关节弯曲角度逐帧对比", 0.40),
        ("动作幅度比", breakdown.get('amplitude_ratio', 0), "四肢末端运动幅度匹配", 0.25),
        ("动作流畅度", breakdown.get('motion_smoothness', 0), "轨迹平滑连贯程度", 0.20),
        ("下肢灵活度", breakdown.get('lower_body_stiffness', 0), "腿部关节活动充分度", 0.15),
    ]

    dim_cards = ""
    for name, score, desc, weight in dims:
        dg = get_grade(score)
        dc = {"优秀": "#22c55e", "良好": "#3b82f6", "有待提高": "#f59e0b", "需要练习": "#ef4444"}[dg]
        dim_cards += f"""
        <div style="flex:1; min-width:140px; background:#fff; border-radius:12px; padding:18px 14px;
                    box-shadow:0 2px 8px rgba(0,0,0,.06); text-align:center; margin:4px;">
            <div style="font-size:12px; font-weight:600; color:var(--ink-s); margin-bottom:4px;">{name}</div>
            <div style="font-size:34px; font-weight:800; color:{dc};">{score:.0f}</div>
            <div style="font-size:11px; color:var(--muted); margin:4px 0;">{desc}</div>
            <div style="font-size:10px; color:var(--muted);">权重 {weight:.0%}</div>
            <div style="background:#f3f4f6; border-radius:8px; height:6px; margin-top:8px;">
                <div style="background:{dc}; border-radius:8px; height:6px; width:min({score:.0f}%,{100}%);"></div>
            </div>
            <div style="font-size:11px; color:{dc}; margin-top:4px; font-weight:700;">{dg}</div>
        </div>"""

    return f"""
    <div style="font-family: -apple-system, 'Microsoft YaHei', sans-serif; max-width: 900px; margin: 0 auto;">

    <div style="text-align:center; padding:24px 0 16px 0;">
        <div style="display:inline-block; width:100px; height:100px; border-radius:50%;
                    background:linear-gradient(135deg, {c_main}, {c_dark}); line-height:100px;
                    font-size:42px; font-weight:800; color:#fff; margin-bottom:8px;
                    box-shadow:0 4px 16px rgba(0,0,0,.12);">
            {overall:.0f}
        </div>
        <div style="font-size:20px; font-weight:700; color:{c_dark};">综合评分 · {grade}</div>
        <div style="font-size:13px; color:var(--muted); margin-top:4px;">
            {metrics.get('num_valid_frames', 0)} 有效帧分析
        </div>
    </div>

    <div style="display:flex; flex-wrap:wrap; gap:8px; margin:16px 0;">
        {dim_cards}
    </div>
    </div>"""


def render_diff_page(diff_grid: list, page: int, per_page: int = 3) -> str:
    """Re-render skeleton diff HTML for a given page (called on slider change)."""
    if not diff_grid:
        return '<div style="color:var(--muted);padding:20px;">暂无逐帧对比数据</div>'
    total_pages = max(1, (len(diff_grid) + per_page - 1) // per_page)
    page = max(0, min(page, total_pages - 1))
    return build_skeleton_diff_html(diff_grid, page=page, per_page=per_page)


def build_joint_detail_table(per_frame_data: dict, diff_grid: list[dict]) -> str:
    """Build a per-frame joint angle detail table for the best and worst frames."""
    if not diff_grid:
        return ""

    best = max(diff_grid, key=lambda x: x['score'])
    worst = min(diff_grid, key=lambda x: x['score'])

    def frame_table(item: dict, label: str, border_color: str):
        diffs = item.get('joint_diffs', {})
        if not diffs:
            return ""

        rows = ""
        for jname, diff in sorted(diffs.items(), key=lambda x: x[1]):
            cn = JOINT_NAMES_CN.get(jname, jname)
            jc = joint_diff_hex(diff)
            rows += f"""
            <tr>
                <td style="padding:8px 12px; font-weight:600; color:var(--ink-s);">{cn}</td>
                <td style="padding:8px 12px; color:{jc}; font-weight:600;">{diff:.0f}°</td>
                <td style="padding:8px 12px;">
                    <span style="display:inline-block; padding:2px 10px; border-radius:12px;
                           font-size:11px; font-weight:700; color:#fff; background:{jc};">
                        {get_grade(100 - diff)}</span>
                </td>
            </tr>"""

        return f"""
        <div style="flex:1; min-width:280px;">
            <div style="font-size:14px; font-weight:600; color:var(--ink-s); margin-bottom:10px;
                        padding:8px 12px; border-left:3px solid {border_color}; background:var(--bg2);
                        border-radius:0 8px 8px 0;">
                {label} — 第 {item['frame_idx'] + 1} 帧 ({item['score']:.0f}分)
            </div>
            <table style="width:100%; border-collapse:collapse; font-size:13px;">
                <tbody>{rows}</tbody>
            </table>
        </div>"""

    best_table = frame_table(best, "最佳匹配帧", "#22c55e")
    worst_table = frame_table(worst, "最需练习帧", "#ef4444")

    return f"""
    <div style="font-family: -apple-system, 'Microsoft YaHei', sans-serif; max-width: 900px; margin: 0 auto;">
    <div style="background:#fff; border-radius:14px; padding:24px; margin:20px 0;
                box-shadow:0 2px 8px rgba(0,0,0,.06);">
        <h3 style="margin:0 0 16px 0; font-size:17px; color:var(--ink);">逐帧关节角度详情</h3>
        <div style="display:flex; gap:16px; flex-wrap:wrap;">
            {best_table}
            {worst_table}
        </div>
    </div>
    </div>"""


def build_error_html(msg: str) -> str:
    return f"""<div style="text-align:center; padding:40px; font-family:'Microsoft YaHei',sans-serif;">
        <div style="font-size:48px; margin-bottom:16px;">&#9888;&#65039;</div>
        <div style="font-size:16px; color:#ef4444;">{msg}</div>
    </div>"""


# ==============================================================================
# Tab 1: Video comparison analysis
# ==============================================================================

def run_analysis(
    teacher_video: str | None,
    student_video: str | None,
    api_key: str,
    provider: str,
    num_frames: int,
    student_offset: int,
) -> tuple:
    """
    Returns (output_video_path, metrics_html, diff_html, coach_report_md,
             diff_grid_data, slider_update).
    """
    empty_slider = gr.update(maximum=1, value=0)
    if teacher_video is None:
        return None, build_error_html("请上传老师视频"), "", "", [], empty_slider
    if student_video is None:
        return None, build_error_html("请上传学生视频"), "", "", [], empty_slider

    try:
        now = lambda: datetime.now().strftime("%H:%M:%S")
        print(f"[{now()}] ===== 开始分析 =====")

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_video_path = os.path.join(OUTPUT_DIR, f"compare_{timestamp}.mp4")

        print(f"[{now()}] 正在提取骨骼关键点...")
        data = process_video_pair(teacher_video, student_video, num_frames,
                                  student_offset=student_offset)

        print(f"[{now()}] 正在生成对比视频...")
        generate_comparison_video(data, output_video_path, fps=5)

        print(f"[{now()}] 正在计算评分指标...")
        metrics = compute_all_metrics(data)
        overall = metrics.get('overall_score', 0)
        print(f"[{now()}] 综合评分: {overall:.1f}")

        print(f"[{now()}] 正在生成骨架对比网格...")
        per_frame_data = metrics.get('per_frame_angles', {})
        # Generate more keyframes for paginated browsing (7-9 frames, 3/page)
        diff_grid = generate_diff_grid(data, per_frame_data, num_frames=9)
        per_page = 3
        total_pages = max(1, (len(diff_grid) + per_page - 1) // per_page)
        diff_html = build_skeleton_diff_html(diff_grid, page=0, per_page=per_page)
        overall_joint_html = build_overall_joint_analysis(per_frame_data)
        joint_detail_html = build_joint_detail_table(per_frame_data, diff_grid)
        score_html = build_score_cards_html(metrics)

        # Combine: score + overall analysis + frame detail (diff grid is separate)
        metrics_html = score_html + overall_joint_html + joint_detail_html

        # AI coach report (only if API key provided)
        report_md = ""
        if api_key and api_key.strip() and provider:
            print(f"[{now()}] 正在生成 AI 教练点评...")
            report = generate_coaching_report(api_key.strip(), provider, metrics)
            report_md = f"## AI教练点评\n\n{report}"
        else:
            report_md = "*（未提供 API Key，跳过 AI 点评）*"

        print(f"[{now()}] ===== 分析完成 =====")
        slider_update = gr.update(maximum=max(1, total_pages - 1), value=0)
        return (output_video_path, metrics_html, diff_html, report_md,
                diff_grid, slider_update)

    except Exception as e:
        print(f"[ERROR] {e}")
        traceback.print_exc()
        return (None, build_error_html(f"分析过程出错: {str(e)}"), "", "",
                [], gr.update(maximum=1, value=0))


# ==============================================================================
# Tab 2: Single image test
# ==============================================================================

def run_image_test(image_path: str | None) -> np.ndarray | None:
    if image_path is None:
        return None

    detector = None
    try:
        detector = PoseDetector(model_path="pose_landmarker_lite.task",
                                num_poses=1, mode="image")
        image = cv2.imread(image_path)
        if image is None:
            return None

        result = detector.detect_center_person(image)
        if result is None or 'landmarks_pixel' not in result:
            return None

        annotated = detector.draw_pose(image.copy(), result['landmarks_pixel'])
        return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    except Exception as e:
        print(f"[ERROR] 骨骼检测失败: {e}")
        return None
    finally:
        if detector is not None:
            try:
                detector.close()
            except Exception:
                pass


# ==============================================================================
# Usage instructions
# ==============================================================================

USAGE_MD = """
## 舞蹈动作智能对比分析工具 v2

通过骨骼关键点检测技术，将学生视频与老师视频进行**逐帧**对比，展示每个动作的匹配情况。

---

### 核心改进（相比 v1）

- **逐帧骨架对比网格** — 不再取平均角度，而是展示5个关键帧的骨骼对比
- **关节颜色编码** — <span style="color:#22c55e;">●绿色</span> (<15°) / <span style="color:#f59e0b;">●黄色</span> (15-30°) / <span style="color:#ef4444;">●红色</span> (>30°)
- **自动高亮** — 最佳匹配帧和最需练习帧自动标注

---

### 评分维度

| 维度 | 权重 | 含义 |
|------|------|------|
| 关节角度相似度 | 40% | 肘、肩、膝、髋各关节弯曲角度逐帧对比 |
| 动作幅度比 | 25% | 手腕、脚踝运动幅度的匹配程度 |
| 动作流畅度 | 20% | 动作轨迹的平滑度和连贯性 |
| 下肢灵活度 | 15% | 腿部关节的活动充分程度 |

### 评分等级

- **优秀**（≥ 90 分）— 动作标准，与老师高度一致
- **良好**（≥ 80 分）— 动作较好，细节可优化
- **有待提高**（≥ 70 分）— 基本正确，需加强练习
- **需要练习**（< 70 分）— 差异较大，建议重点训练

---

### 隐私提示

- API Key 仅用于本次 AI 点评，不会被存储
- 上传的视频仅在本地处理，不上传云端
"""


# ==============================================================================
# Gradio app builder
# ==============================================================================

def create_app() -> gr.Blocks:
    with gr.Blocks(title="舞蹈动作智能对比分析 v2") as app:

        gr.Markdown("## 舞蹈动作智能对比分析工具 v2")
        gr.Markdown("上传老师和学生的舞蹈视频，逐帧对比骨骼动作，精准定位改进方向。")

        with gr.Tabs():

            # ===== Tab 1 =====
            with gr.Tab("视频对比分析"):
                gr.Markdown("### 上传视频")

                with gr.Row():
                    with gr.Column():
                        teacher_video = gr.Video(
                            label="老师视频", sources=["upload"], height=340)
                    with gr.Column():
                        student_video = gr.Video(
                            label="学生视频", sources=["upload"], height=340)

                with gr.Row():
                    api_key = gr.Textbox(
                        label="API Key", type="password",
                        placeholder="输入 DeepSeek 或 Gemini API Key（可选）")
                    provider = gr.Radio(
                        label="AI 点评模型",
                        choices=["deepseek", "gemini"], value="deepseek")
                    num_frames = gr.Slider(
                        label="采样帧数", minimum=10, maximum=60, value=30, step=5)
                    student_offset = gr.Slider(
                        label="学生视频偏移", minimum=0, maximum=60, value=0, step=1,
                        info="向前跳过学生视频的帧数，对齐动作起点")

                with gr.Row():
                    offset_preview = gr.Image(
                        label="偏移预览（学生视频起始帧）", interactive=False,
                        height=200, visible=True)

                student_offset.change(
                    fn=preview_offset_frame,
                    inputs=[student_video, student_offset],
                    outputs=[offset_preview],
                )

                analyze_btn = gr.Button("开始分析", variant="primary", size="lg")

                with gr.Row():
                    diff_page_slider = gr.Slider(
                        label="翻页浏览骨架对比", minimum=0, maximum=1, value=0, step=1,
                        info="拖动切换对比帧页面（每页3帧）", visible=True)

                diff_grid_state = gr.State([])

                output_video = gr.Video(label="对比视频", interactive=False, height=400)
                metrics_display = gr.HTML(label="评分报告")
                diff_display = gr.HTML(label="逐帧骨架对比")
                coach_report = gr.Markdown("")

                analyze_btn.click(
                    fn=run_analysis,
                    inputs=[teacher_video, student_video, api_key, provider,
                            num_frames, student_offset],
                    outputs=[output_video, metrics_display, diff_display,
                             coach_report, diff_grid_state, diff_page_slider],
                )

                diff_page_slider.change(
                    fn=render_diff_page,
                    inputs=[diff_grid_state, diff_page_slider],
                    outputs=[diff_display],
                )

            # ===== Tab 2 =====
            with gr.Tab("单图骨骼测试"):
                gr.Markdown("### 上传图片，测试骨骼关键点检测效果")

                with gr.Row():
                    with gr.Column(scale=1):
                        input_image = gr.Image(label="上传图片", type="filepath")
                        detect_btn = gr.Button("检测骨骼", variant="primary")
                    with gr.Column(scale=1):
                        output_image = gr.Image(label="检测结果", interactive=False)

                detect_btn.click(
                    fn=run_image_test,
                    inputs=[input_image],
                    outputs=[output_image],
                )

            # ===== Tab 3 =====
            with gr.Tab("使用说明"):
                gr.Markdown(USAGE_MD)

    return app


if __name__ == "__main__":
    app = create_app()
    app.launch(server_name="127.0.0.1", server_port=7861, share=False,
               css=CUSTOM_CSS,
               theme=gr.themes.Soft(primary_hue="amber", secondary_hue="rose"))
