"""
Video processing pipeline for the dance analysis tool.

Frames are kept at original resolution — no resize that would degrade
MediaPipe pose detection quality. Frames are uniformly sampled from the
ENTIRE video so the output covers the full dance duration.
"""

import os
import time
import cv2
import numpy as np
from pose_detector import PoseDetector, draw_pose


# ---------------------------------------------------------------------------
# 1. Frame extraction — uniform sampling across entire video
# ---------------------------------------------------------------------------

def extract_frames(video_path, num_frames=30):
    """
    Uniformly sample frames from the ENTIRE video at original resolution.

    Reads all frames once, then picks `num_frames` evenly-spaced indices
    to cover the full video duration. No resize — native resolution gives
    the best MediaPipe accuracy (same as image.py).

    Args:
        video_path: Path to the video file.
        num_frames: Number of frames to sample (spread across full video).

    Returns:
        List of (frame_index, frame) tuples: [(idx, BGR ndarray), ...].
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open video: {video_path}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Read all frames
    all_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        all_frames.append(frame)
    cap.release()

    total = len(all_frames)

    if total == 0:
        black = np.zeros((h or 540, w or 960, 3), dtype=np.uint8)
        print(f"  [WARN] {video_path}: 0 frames — padding with black")
        return [(0, black.copy()) for _ in range(num_frames)]

    # Uniform sampling across full duration
    if num_frames >= total:
        sampled = [(i, all_frames[i]) for i in range(total)]
    else:
        step = (total - 1) / (num_frames - 1) if num_frames > 1 else 0
        sampled = [(int(round(i * step)), all_frames[int(round(i * step))])
                   for i in range(num_frames)]

    duration = total / fps if fps > 0 else 0
    print(f"  {video_path}: {total} frames, {duration:.1f}s @ {fps:.0f}fps "
          f"-> sampled {len(sampled)} frames")
    return sampled


# ---------------------------------------------------------------------------
# 2. Paired processing (teacher + student)
# ---------------------------------------------------------------------------

def process_video_pair(teacher_path, student_path, num_frames=30):
    """
    Sample frames across the ENTIRE video duration (not just the beginning),
    run center-person pose detection on each sampled frame.

    Uses IMAGE mode since frames are not consecutive after uniform sampling.
    Frame-to-frame cache backfills occasional misses.

    Returns:
        dict with keys: teacher, student, num_frames, teacher_path, student_path
    """
    print("=" * 60)
    print("Extracting frames (uniform sample, full video) ...")
    t0 = time.time()

    teacher_samples = extract_frames(teacher_path, num_frames)
    student_samples = extract_frames(student_path, num_frames)

    print(f"  Frame extraction took {time.time() - t0:.1f}s")

    print("Initialising PoseDetector (IMAGE mode, frame cache enabled) ...")
    detector = PoseDetector(model_path="pose_landmarker_lite.task",
                            num_poses=1, mode="image")

    teacher_results = []
    student_results = []

    print(f"Detecting poses on teacher frames ({len(teacher_samples)} frames) ...")
    for i, (orig_idx, frame) in enumerate(teacher_samples):
        result = detector.detect_center_person(frame, cache_id="teacher")
        teacher_results.append({'frame': frame, 'landmarks': result})
        if (i + 1) % 5 == 0 or i == 0:
            status = "found" if result else "none"
            print(f"  Teacher frame {i + 1}/{len(teacher_samples)} (#{orig_idx}): {status}")

    print(f"Detecting poses on student frames ({len(student_samples)} frames) ...")
    for i, (orig_idx, frame) in enumerate(student_samples):
        result = detector.detect_center_person(frame, cache_id="student")
        student_results.append({'frame': frame, 'landmarks': result})
        if (i + 1) % 5 == 0 or i == 0:
            status = "found" if result else "none"
            print(f"  Student frame {i + 1}/{len(student_samples)} (#{orig_idx}): {status}")

    detector.close()

    teacher_found = sum(1 for r in teacher_results if r['landmarks'] is not None)
    student_found = sum(1 for r in student_results if r['landmarks'] is not None)
    print(f"  Teacher: {teacher_found}/{len(teacher_results)} frames with pose")
    print(f"  Student: {student_found}/{len(student_results)} frames with pose")
    print(f"  Pose detection took {time.time() - t0:.1f}s total")

    return {
        'teacher': teacher_results,
        'student': student_results,
        'num_frames': len(teacher_results),
        'teacher_path': teacher_path,
        'student_path': student_path,
    }


# ---------------------------------------------------------------------------
# 3. Comparison video — black canvas, original sizes
# ---------------------------------------------------------------------------

def generate_comparison_video(processed_data, output_path, fps=5):
    """
    Create side-by-side comparison video on a black canvas.

    Each frame keeps its original resolution. For teacher/student pairs
    with different sizes, the smaller one is centered in its half.
    Black fills any unused space.

    Teacher on the left (blue skeleton, "Teacher" label).
    Student on the right (red skeleton, "Student" label).
    """
    teacher_data = processed_data['teacher']
    student_data = processed_data['student']
    num_frames = processed_data['num_frames']

    # Determine max dimensions across all frames
    max_h = 0
    max_w_teacher = 0
    max_w_student = 0
    for i in range(num_frames):
        th, tw = teacher_data[i]['frame'].shape[:2]
        sh, sw = student_data[i]['frame'].shape[:2]
        max_h = max(max_h, th, sh)
        max_w_teacher = max(max_w_teacher, tw)
        max_w_student = max(max_w_student, sw)

    canvas_w = max_w_teacher + max_w_student
    canvas_h = max_h

    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (canvas_w, canvas_h))

    print(f"Generating comparison video ({num_frames} frames, canvas {canvas_w}x{canvas_h}) ...")
    for i in range(num_frames):
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

        t_entry = teacher_data[i]
        s_entry = student_data[i]

        t_frame = t_entry['frame'].copy()
        s_frame = s_entry['frame'].copy()

        # Draw skeletons
        t_lm = t_entry['landmarks']
        s_lm = s_entry['landmarks']

        if t_lm is not None and 'landmarks_pixel' in t_lm:
            draw_pose(t_frame, t_lm['landmarks_pixel'])
        if s_lm is not None and 'landmarks_pixel' in s_lm:
            draw_pose(s_frame, s_lm['landmarks_pixel'])

        # Labels on each frame
        cv2.putText(t_frame, "Teacher", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(s_frame, "Student", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Place teacher on left half, centered vertically
        th, tw = t_frame.shape[:2]
        ty = (canvas_h - th) // 2
        tx = (max_w_teacher - tw) // 2
        canvas[ty:ty + th, tx:tx + tw] = t_frame

        # Place student on right half, centered vertically
        sh, sw = s_frame.shape[:2]
        sy = (canvas_h - sh) // 2
        sx = max_w_teacher + (max_w_student - sw) // 2
        canvas[sy:sy + sh, sx:sx + sw] = s_frame

        writer.write(canvas)

        if (i + 1) % 5 == 0 or i == 0:
            print(f"  Wrote frame {i + 1}/{num_frames}")

    writer.release()
    print(f"  Comparison video saved to: {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# 4. Main entry point
# ---------------------------------------------------------------------------

def main():
    """Run the full pipeline with the default test videos."""
    teacher_path = "1teacher.mp4"
    student_path = "1me.mp4"
    output_path = "outputs/compare_result.mp4"

    total_start = time.time()

    print("=" * 60)
    print("Dance Analysis - Video Processing Pipeline")
    print("=" * 60)

    data = process_video_pair(teacher_path, student_path, num_frames=30)
    generate_comparison_video(data, output_path, fps=5)

    elapsed = time.time() - total_start
    print("=" * 60)
    print(f"Pipeline complete. Total elapsed: {elapsed:.1f}s")
    print(f"Output: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
