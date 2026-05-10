"""
Core pose detection module for the dance analysis tool.

Extracts and refactors the VERIFIED pose detection logic from image.py
into a reusable PoseDetector class.

Key improvements over the original:
- VIDEO mode with temporal tracking (eliminates skeleton jitter/drift)
- Single landmarker instance (no per-frame model reload)
- Frame-to-frame cache: reuse last landmarks on missed detection
- Center-person selection via shoulder position (more stable than nose)
- Lower detection confidence for better recall
"""

import os
os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'
os.environ['GLOG_minloglevel'] = '3'

import cv2
import numpy as np
from mediapipe import Image, ImageFormat
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ===================== Module-level constants =====================

POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),  # 脸部
    (9, 10), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),  # 身体+手臂
    (15, 17), (15, 19), (15, 21), (16, 18), (16, 20), (16, 22),  # 手腕
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26),  # 躯干+胯部
    (25, 27), (27, 29), (27, 31), (26, 28), (28, 30), (28, 32)   # 腿部
]

BaseOptions = python.BaseOptions
PoseLandmarker = vision.PoseLandmarker
PoseLandmarkerOptions = vision.PoseLandmarkerOptions
RunningMode = vision.RunningMode


# ===================== PoseDetector Class =====================

class PoseDetector:
    """
    Reusable pose detector with VIDEO-mode temporal tracking.

    Creates the PoseLandmarker ONCE and reuses it across all frames.
    Uses frame-to-frame cache to fill in missed detections.
    """

    def __init__(self, model_path="pose_landmarker_lite.task", num_poses=1,
                 mode="video"):
        """
        Args:
            model_path: Path to the MediaPipe pose landmarker .task file.
            num_poses: Max poses to detect. 1 is optimal when only one person
                       is expected (avoids false positives from noise).
            mode: "video" for temporal tracking (consecutive frames) or
                  "image" for independent single-frame detection.
        """
        self.mode = mode
        running_mode = RunningMode.VIDEO if mode == "video" else RunningMode.IMAGE

        if mode == "video":
            detector_options = PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                running_mode=RunningMode.VIDEO,
                num_poses=num_poses,
                # Favor recall for real-world phone videos (blur/occlusion/backlight).
                min_pose_detection_confidence=0.2,
                min_pose_presence_confidence=0.2,
                min_tracking_confidence=0.75,
            )
        else:
            detector_options = PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                running_mode=RunningMode.IMAGE,
                num_poses=num_poses,
            )

        self.landmarker = PoseLandmarker.create_from_options(detector_options)

        # Frame-to-frame cache for missed detections
        self._cache = {}
        # Timestamp tracker for VIDEO mode
        self._timestamp_ms = 0

    # ------------------------------------------------------------------
    # Core detection
    # ------------------------------------------------------------------

    def detect_center_person(self, frame: np.ndarray,
                             cache_id=None) -> dict | None:
        """
        Detect the center person's pose in a BGR frame.

        Uses shoulder position (landmark 11) for center selection, which is
        more stable than nose (landmark 0) when the person turns or bends.

        In VIDEO mode, timestamps are auto-incremented for temporal tracking.
        Missed detections fall back to the cached previous frame.

        Args:
            frame: BGR image as a numpy array (H, W, 3).
            cache_id: Optional key for per-subject frame-to-frame cache
                      (e.g. "teacher", "student"). When None, no caching.

        Returns:
            None if no person detected AND no cache available.
            Otherwise a dict with:
              - 'landmarks': list of 33 (x, y, z, visibility) tuples
              - 'landmarks_pixel': list of 33 (x_px, y_px) tuples
              - 'image_size': (h, w)
        """
        img_h, img_w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)

        if self.mode == "video":
            self._timestamp_ms += 200  # ~5fps equivalent
            results = self.landmarker.detect_for_video(mp_image, self._timestamp_ms)
        else:
            results = self.landmarker.detect(mp_image)

        best_landmark = None

        if results.pose_landmarks:
            min_center_dist = float('inf')
            for lm in results.pose_landmarks:
                # Use shoulder (landmark 11) for center — more stable than nose
                cx = lm[11].x
                cy = lm[11].y
                dist = abs(cx - 0.5) + abs(cy - 0.5)
                if dist < min_center_dist:
                    min_center_dist = dist
                    best_landmark = lm

        # Fall back to cache on missed detection
        if best_landmark is None and cache_id is not None:
            best_landmark = self._cache.get(cache_id)

        # Update cache
        if best_landmark is not None and cache_id is not None:
            self._cache[cache_id] = best_landmark

        if best_landmark is None:
            return None

        landmarks_norm = []
        landmarks_pixel = []
        for lm in best_landmark:
            landmarks_norm.append((lm.x, lm.y, lm.z, lm.visibility))
            landmarks_pixel.append((int(lm.x * img_w), int(lm.y * img_h)))

        return {
            "landmarks": landmarks_norm,
            "landmarks_pixel": landmarks_pixel,
            "image_size": (img_h, img_w),
        }

    # ------------------------------------------------------------------
    # Drawing (static method — no detector instance needed)
    # ------------------------------------------------------------------

    @staticmethod
    def draw_pose(frame: np.ndarray, landmarks_pixel: list,
                  skeleton_color=(255, 0, 0), point_color=(0, 255, 0),
                  thickness=5, point_radius=4) -> np.ndarray:
        """Draw skeleton + keypoints. Modifies frame in-place; returns it."""
        for start, end in POSE_CONNECTIONS:
            x1, y1 = landmarks_pixel[start]
            x2, y2 = landmarks_pixel[end]
            cv2.line(frame, (x1, y1), (x2, y2), skeleton_color, thickness)
        for x, y in landmarks_pixel:
            cv2.circle(frame, (x, y), point_radius, point_color, -1)
        return frame

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self):
        if self.landmarker is not None:
            self.landmarker.close()
            self.landmarker = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


# ===================== Module-level convenience functions =====================

def draw_pose(frame, landmarks_pixel,
              skeleton_color=(255, 0, 0), point_color=(0, 255, 0),
              thickness=5, point_radius=4):
    """Quick drawing without a PoseDetector instance (same as static method)."""
    return PoseDetector.draw_pose(frame, landmarks_pixel,
                                  skeleton_color, point_color,
                                  thickness, point_radius)
