import base64
import os
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from pose_detector import PoseDetector, draw_pose
from metrics import compute_all_metrics


APP_TITLE = "快卷吧妈妈 · 本地骨骼服务"
MODEL_PATH = str(Path(__file__).parent / "vendor" / "mediapipe" / "models" / "pose_landmarker_lite.task")


app = FastAPI(title=APP_TITLE)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _save_upload(upload: UploadFile, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("wb") as f:
        while True:
            chunk = upload.file.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)


def _cap_seek_seconds(cap: cv2.VideoCapture, seconds: float) -> None:
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, seconds) * 1000.0)


def _b64jpg(frame_bgr) -> str:
    ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        raise RuntimeError("Failed to encode jpg")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _pose_quality(pose: dict[str, Any] | None) -> float:
    """
    Heuristic quality score in [0, 1] based on visible landmark bbox size and visibility.
    Higher means the body is more fully in frame and should be easier to compare.
    """
    if not pose or not pose.get("landmarks"):
        return 0.0
    xs: list[float] = []
    ys: list[float] = []
    vis: list[float] = []
    for x, y, _z, v in pose["landmarks"]:
        vv = float(v if v is not None else 1.0)
        if vv < 0.15:
            continue
        xs.append(float(x))
        ys.append(float(y))
        vis.append(vv)
    if len(xs) < 6:
        return 0.0
    w = max(xs) - min(xs)
    h = max(ys) - min(ys)
    area = max(0.0, min(1.0, w * h * 1.8))
    vmean = float(sum(vis) / max(1, len(vis)))
    return float(max(0.0, min(1.0, 0.75 * area + 0.25 * vmean)))


def scan_segment_pick_keyframes(
    video_path: str,
    start_s: float,
    end_s: float,
    n: int,
    cache_id: str,
    detector: PoseDetector,
    scan_fps: float = 10.0,
) -> list[dict[str, Any]]:
    """
    Scan the segment sequentially (video mode tracking), then pick `n` keyframes
    near evenly-spaced target times, preferring frames with higher pose quality.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open video: {video_path}")

    start_s = float(max(0.0, start_s))
    end_s = float(max(start_s, end_s))
    span = max(0.0, end_s - start_s)
    if span <= 1e-3:
        cap.release()
        return []

    step_s = max(1.0 / 30.0, 1.0 / float(scan_fps))
    targets = [start_s + (span * i / float(max(1, n - 1))) for i in range(n)]

    _cap_seek_seconds(cap, start_s)
    samples: list[dict[str, Any]] = []
    next_sample_t = start_s

    while True:
        cur_t = float(cap.get(cv2.CAP_PROP_POS_MSEC) or (start_s * 1000.0)) / 1000.0
        if cur_t > end_s + 0.05:
            break
        ret, frame = cap.read()
        if not ret:
            break
        cur_t = float(cap.get(cv2.CAP_PROP_POS_MSEC) or (cur_t * 1000.0)) / 1000.0
        if cur_t + 1e-6 < next_sample_t:
            continue

        pose = detector.detect_center_person(frame, cache_id=cache_id)
        q = _pose_quality(pose)
        overlay = frame.copy()
        if pose and pose.get("landmarks_pixel"):
            draw_pose(overlay, pose["landmarks_pixel"])

        samples.append(
            {
                "t": float(cur_t),
                "pose_ok": bool(pose),
                "quality": float(q),
                "overlay": overlay,
            }
        )
        next_sample_t += step_s
        if cur_t >= end_s:
            break

    cap.release()
    if not samples:
        return []

    picked: list[dict[str, Any]] = []
    for tt in targets:
        # prefer the best-quality sample within a small window around target time
        window = max(0.18, 2.0 * step_s)
        candidates = [s for s in samples if abs(s["t"] - tt) <= window]
        if not candidates:
            candidates = samples
        best = max(candidates, key=lambda s: (s["pose_ok"], s["quality"], -abs(s["t"] - tt)))
        picked.append(
            {
                "t": float(tt),
                "picked_t": float(best["t"]),
                "pose_ok": bool(best["pose_ok"]),
                "quality": float(best["quality"]),
                "image_b64": _b64jpg(best["overlay"]),
            }
        )

    return picked


def _scan_segment_for_metrics(
    video_path: str,
    start_s: float,
    end_s: float,
    n: int,
    cache_id: str,
    detector: PoseDetector,
) -> list[dict[str, Any]]:
    """
    Similar to scan_segment_pick_keyframes, but keeps landmarks_pixel for metrics computation.
    Returns list entries compatible with metrics.get_valid_pairs() expectations.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open video: {video_path}")

    start_s = float(max(0.0, start_s))
    end_s = float(max(start_s, end_s))
    span = max(0.0, end_s - start_s)
    if span <= 1e-3:
        cap.release()
        return []

    targets = [start_s + (span * i / float(max(1, n - 1))) for i in range(n)]
    _cap_seek_seconds(cap, start_s)

    samples: list[dict[str, Any]] = []
    # scan at 10 fps by default in scan_segment_pick_keyframes; mirror here
    step_s = 0.1
    next_sample_t = start_s

    while True:
        cur_t = float(cap.get(cv2.CAP_PROP_POS_MSEC) or (start_s * 1000.0)) / 1000.0
        if cur_t > end_s + 0.05:
            break
        ret, frame = cap.read()
        if not ret:
            break
        cur_t = float(cap.get(cv2.CAP_PROP_POS_MSEC) or (cur_t * 1000.0)) / 1000.0
        if cur_t + 1e-6 < next_sample_t:
            continue
        pose = detector.detect_center_person(frame, cache_id=cache_id)
        q = _pose_quality(pose)
        samples.append({"t": float(cur_t), "pose": pose, "quality": float(q)})
        next_sample_t += step_s
        if cur_t >= end_s:
            break

    cap.release()
    if not samples:
        return []

    picked_entries: list[dict[str, Any]] = []
    for tt in targets:
        window = max(0.2, 2.0 * step_s)
        candidates = [s for s in samples if abs(s["t"] - tt) <= window]
        if not candidates:
            candidates = samples
        best = max(candidates, key=lambda s: (bool(s["pose"]), s["quality"], -abs(s["t"] - tt)))
        pose = best["pose"]
        lm_px = pose.get("landmarks_pixel") if pose else None
        picked_entries.append(
            {
                "frame": None,
                "landmarks": {"landmarks_pixel": lm_px} if lm_px else None,
                "t": float(tt),
                "picked_t": float(best["t"]),
            }
        )

    return picked_entries


def extract_keyframes_with_tracking(
    video_path: str,
    start_s: float,
    end_s: float,
    offsets_s: list[float],
    cache_id: str,
    detector: PoseDetector,
) -> list[dict[str, Any]]:
    """
    Read video sequentially from start_s to end_s, and when current time passes a target time,
    record that frame and run VIDEO-mode pose detection (temporal tracking).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open video: {video_path}")

    targets = [start_s + o for o in offsets_s]
    targets = [t for t in targets if t <= end_s + 1e-6]
    if not targets:
        cap.release()
        return []

    _cap_seek_seconds(cap, start_s)
    target_i = 0

    frames: list[dict[str, Any]] = []
    seconds = float(cap.get(cv2.CAP_PROP_POS_MSEC) or (start_s * 1000.0)) / 1000.0

    while target_i < len(targets):
        ret, frame = cap.read()
        if not ret:
            break
        seconds = float(cap.get(cv2.CAP_PROP_POS_MSEC) or (seconds * 1000.0)) / 1000.0

        # wait until current playback time reaches the target timestamp
        if seconds < targets[target_i]:
            continue

        pose = detector.detect_center_person(frame, cache_id=cache_id)
        overlay = frame.copy()
        if pose and pose.get("landmarks_pixel"):
            draw_pose(overlay, pose["landmarks_pixel"])

        frames.append(
            {
                "t": float(targets[target_i]),
                "pose_ok": bool(pose),
                "image_b64": _b64jpg(overlay),
            }
        )
        target_i += 1

    cap.release()
    return frames


def extract_overlay_frames(
    video_path: str,
    start_s: float,
    end_s: float,
    offsets_s: list[float],
    cache_id: str,
    detector: PoseDetector,
) -> list[Any]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open video: {video_path}")

    targets = [start_s + o for o in offsets_s]
    targets = [t for t in targets if t <= end_s + 1e-6]
    if not targets:
        cap.release()
        return []

    _cap_seek_seconds(cap, start_s)
    target_i = 0
    frames: list[Any] = []
    seconds = float(cap.get(cv2.CAP_PROP_POS_MSEC) or (start_s * 1000.0)) / 1000.0

    while target_i < len(targets):
        ret, frame = cap.read()
        if not ret:
            break
        seconds = float(cap.get(cv2.CAP_PROP_POS_MSEC) or (seconds * 1000.0)) / 1000.0
        if seconds < targets[target_i]:
            continue

        pose = detector.detect_center_person(frame, cache_id=cache_id)
        overlay = frame.copy()
        if pose and pose.get("landmarks_pixel"):
            draw_pose(overlay, pose["landmarks_pixel"])
        frames.append(overlay)
        target_i += 1

    cap.release()
    return frames


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/compare/keyframes")
async def compare_keyframes(
    ref_video: UploadFile = File(...),
    self_video: UploadFile = File(...),
    ref_start: float = Form(0),
    ref_end: float = Form(15),
    self_start: float = Form(0),
    self_end: float = Form(15),
) -> dict[str, Any]:
    if not os.path.exists(MODEL_PATH):
        return {
            "ok": False,
            "error": f"Model file missing: {MODEL_PATH}",
        }

    with tempfile.TemporaryDirectory(prefix="kqmm_") as td:
        td_path = Path(td)
        ref_path = td_path / "ref.mp4"
        self_path = td_path / "self.mp4"
        _save_upload(ref_video, ref_path)
        _save_upload(self_video, self_path)

        with PoseDetector(model_path=MODEL_PATH, num_poses=3, mode="video") as detector:
            ref_frames = scan_segment_pick_keyframes(
                str(ref_path), float(ref_start), float(ref_end), 7, "ref", detector
            )
            self_frames = scan_segment_pick_keyframes(
                str(self_path), float(self_start), float(self_end), 7, "self", detector
            )

        return {
            "ok": True,
            "ref": ref_frames,
            "self": self_frames,
        }


@app.post("/api/compare/metrics")
async def compare_metrics(
    ref_video: UploadFile = File(...),
    self_video: UploadFile = File(...),
    ref_start: float = Form(0),
    ref_end: float = Form(15),
    self_start: float = Form(0),
    self_end: float = Form(15),
) -> dict[str, Any]:
    """
    Compute quantitative dance metrics (angles/amplitude/smoothness/stability) for compare mode.
    Uses roughly-even timestamps across both selected segments and pairs them.
    """
    if not os.path.exists(MODEL_PATH):
        return {"ok": False, "error": f"Model file missing: {MODEL_PATH}"}

    with tempfile.TemporaryDirectory(prefix="kqmm_") as td:
        td_path = Path(td)
        ref_path = td_path / "ref.mp4"
        self_path = td_path / "self.mp4"
        _save_upload(ref_video, ref_path)
        _save_upload(self_video, self_path)

        with PoseDetector(model_path=MODEL_PATH, num_poses=3, mode="video") as detector:
            teacher = _scan_segment_for_metrics(str(ref_path), float(ref_start), float(ref_end), 30, "ref", detector)
            student = _scan_segment_for_metrics(str(self_path), float(self_start), float(self_end), 30, "self", detector)

        n = min(len(teacher), len(student))
        processed = {"teacher": teacher[:n], "student": student[:n], "num_frames": n}
        try:
            m = compute_all_metrics(processed)
        except Exception as e:
            return {"ok": False, "error": f"metrics failed: {e.__class__.__name__}: {e}"}

        # Compact view for LLM prompts (keep tokens small, avoid truncation).
        compact: dict[str, Any] = {
            "overall_score": m.get("overall_score", 0.0),
            "breakdown": m.get("breakdown", {}),
            "smoothness": m.get("smoothness", {}),
            "stiffness": m.get("stiffness", {}),
        }

        # Worst joints by similarity
        per_joint = (m.get("joint_angles") or {}).get("per_joint") or {}
        worst = []
        for name, info in per_joint.items():
            try:
                worst.append(
                    {
                        "joint": name,
                        "similarity": float(info.get("similarity", 0.0)),
                        "diff": float(info.get("diff", 0.0)),
                    }
                )
            except Exception:
                continue
        worst.sort(key=lambda x: x["similarity"])
        compact["worst_joints"] = worst[:6]

        # Amplitude summary (ratio based)
        amp = m.get("amplitude") or {}
        per_point = (amp.get("per_point") or amp.get("per_joint") or {})  # compatibility
        amp_list = []
        for pt, info in per_point.items():
            try:
                ratio = float(info.get("ratio", 0.0))
                amp_list.append(
                    {
                        "point": pt,
                        "ratio": ratio,
                        "score": float(info.get("score", info.get("amp_score", 0.0))),
                        "dir": ("smaller" if ratio and ratio < 0.85 else "larger" if ratio and ratio > 1.15 else "ok"),
                    }
                )
            except Exception:
                continue
        compact["amplitude_points"] = amp_list

        # Per-frame scores (if present) but cap length
        pfa = m.get("per_frame_angles") or {}
        if isinstance(pfa, dict) and isinstance(pfa.get("per_frame_scores"), list):
            compact["per_frame_scores"] = pfa["per_frame_scores"][:30]

        return {"ok": True, "metrics": m, "compact": compact}


@app.post("/api/compare/overlay_video")
async def compare_overlay_video(
    ref_video: UploadFile = File(...),
    self_video: UploadFile = File(...),
    ref_start: float = Form(0),
    ref_end: float = Form(15),
    self_start: float = Form(0),
    self_end: float = Form(15),
) -> StreamingResponse:
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model file missing: {MODEL_PATH}")

    with tempfile.TemporaryDirectory(prefix="kqmm_") as td:
        td_path = Path(td)
        ref_path = td_path / "ref.mp4"
        self_path = td_path / "self.mp4"
        _save_upload(ref_video, ref_path)
        _save_upload(self_video, self_path)

        with PoseDetector(model_path=MODEL_PATH, num_poses=3, mode="video") as detector:
            ref_frames = scan_segment_pick_keyframes(
                str(ref_path), float(ref_start), float(ref_end), 7, "ref", detector
            )
            self_frames = scan_segment_pick_keyframes(
                str(self_path), float(self_start), float(self_end), 7, "self", detector
            )

        ref_over = [cv2.imdecode(np.frombuffer(base64.b64decode(f["image_b64"]), np.uint8), cv2.IMREAD_COLOR) for f in ref_frames]
        self_over = [cv2.imdecode(np.frombuffer(base64.b64decode(f["image_b64"]), np.uint8), cv2.IMREAD_COLOR) for f in self_frames]
        n = min(len(ref_over), len(self_over))
        if n == 0:
            raise RuntimeError("No frames extracted")

        # Build side-by-side video
        max_h = max(max(ref_over[i].shape[0], self_over[i].shape[0]) for i in range(n))
        max_w_ref = max(ref_over[i].shape[1] for i in range(n))
        max_w_self = max(self_over[i].shape[1] for i in range(n))
        canvas_w = max_w_ref + max_w_self
        canvas_h = max_h

        tmp = tempfile.NamedTemporaryFile(prefix="kqmm_overlay_", suffix=".mp4", delete=False)
        tmp_path = tmp.name
        tmp.close()

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(tmp_path, fourcc, 2, (canvas_w, canvas_h))
        for i in range(n):
            canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

            t_frame = ref_over[i]
            s_frame = self_over[i]
            th, tw = t_frame.shape[:2]
            sh, sw = s_frame.shape[:2]
            ty = (canvas_h - th) // 2
            sy = (canvas_h - sh) // 2
            tx = (max_w_ref - tw) // 2
            sx = max_w_ref + (max_w_self - sw) // 2
            canvas[ty : ty + th, tx : tx + tw] = t_frame
            canvas[sy : sy + sh, sx : sx + sw] = s_frame
            writer.write(canvas)
        writer.release()

        def iterfile():
            with open(tmp_path, "rb") as f:
                yield from iter(lambda: f.read(1024 * 1024), b"")
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        return StreamingResponse(iterfile(), media_type="video/mp4")
