"""
Real-time Object Detection + Tracking
-------------------------------------
Pipeline : OpenCV capture -> YOLOv8 detection -> SORT tracking -> annotated display

Extras beyond the basics:
  * Motion trails that fade with age, one stable colour per track ID
  * Virtual tripwire that counts objects crossing in each direction
  * Dwell timer (seconds each ID has been on screen)
  * Live heatmap of where activity happens (toggle with H)
  * HUD with FPS, live object count, and per-class tallies

Install : pip install ultralytics opencv-python numpy scipy
Run     : python detect_track.py                      # webcam
          python detect_track.py --source video.mp4   # video file
Keys    : Q quit | H heatmap | T trails | SPACE pause
"""

import argparse
import time
from collections import defaultdict, deque

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO


# ----------------------------------------------------------------------------
# SORT: Kalman filter per object + Hungarian matching on IoU
# ----------------------------------------------------------------------------
def iou_matrix(a, b):
    """Pairwise IoU between two arrays of [x1, y1, x2, y2] boxes."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a = a[:, None, :]
    b = b[None, :, :]
    ix1 = np.maximum(a[..., 0], b[..., 0])
    iy1 = np.maximum(a[..., 1], b[..., 1])
    ix2 = np.minimum(a[..., 2], b[..., 2])
    iy2 = np.minimum(a[..., 3], b[..., 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    return inter / (area_a + area_b - inter + 1e-9)


def box_to_z(b):
    w, h = b[2] - b[0], b[3] - b[1]
    return np.array([b[0] + w / 2, b[1] + h / 2, w * h, w / max(h, 1e-6)])


def state_to_box(x):
    w = np.sqrt(max(x[2] * x[3], 1e-6))
    h = x[2] / max(w, 1e-6)
    return np.array([x[0] - w / 2, x[1] - h / 2, x[0] + w / 2, x[1] + h / 2])


class Track:
    """One tracked object: state = [cx, cy, area, ratio, vx, vy, varea]."""

    _next_id = 1

    def __init__(self, box, cls, conf):
        self.F = np.eye(7)
        self.F[0, 4] = self.F[1, 5] = self.F[2, 6] = 1.0
        self.H = np.eye(4, 7)
        self.R = np.diag([1, 1, 10, 10]).astype(float)
        self.P = np.diag([10, 10, 10, 10, 10000, 10000, 10000]).astype(float)
        self.Q = np.diag([1, 1, 1, 1, 0.01, 0.01, 0.0001]).astype(float)
        self.x = np.zeros(7)
        self.x[:4] = box_to_z(box)

        self.id = Track._next_id
        Track._next_id += 1
        self.cls, self.conf = cls, conf
        self.hits, self.age, self.since_update = 1, 0, 0
        self.born = time.time()
        self.trail = deque(maxlen=40)

    def predict(self):
        if self.x[6] + self.x[2] <= 0:
            self.x[6] = 0
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1
        self.since_update += 1
        return state_to_box(self.x)

    def update(self, box, cls, conf):
        z = box_to_z(box)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(7) - K @ self.H) @ self.P
        self.cls, self.conf = cls, conf
        self.hits += 1
        self.since_update = 0

    @property
    def box(self):
        return state_to_box(self.x)


class Sort:
    def __init__(self, max_age=20, min_hits=3, iou_thresh=0.3):
        self.max_age, self.min_hits, self.iou_thresh = max_age, min_hits, iou_thresh
        self.tracks = []
        self.frame_count = 0

    def update(self, dets):
        """dets: array of [x1, y1, x2, y2, conf, cls]. Returns confirmed tracks."""
        self.frame_count += 1
        preds = np.array([t.predict() for t in self.tracks]).reshape(-1, 4)
        det_boxes = dets[:, :4] if len(dets) else np.zeros((0, 4))

        matches, unmatched_d = [], list(range(len(dets)))
        if len(self.tracks) and len(dets):
            iou = iou_matrix(preds, det_boxes)
            rows, cols = linear_sum_assignment(-iou)
            for r, c in zip(rows, cols):
                if iou[r, c] >= self.iou_thresh:
                    matches.append((r, c))
            matched_d = {c for _, c in matches}
            unmatched_d = [i for i in range(len(dets)) if i not in matched_d]

        for r, c in matches:
            self.tracks[r].update(dets[c, :4], int(dets[c, 5]), float(dets[c, 4]))
        for i in unmatched_d:
            self.tracks.append(Track(dets[i, :4], int(dets[i, 5]), float(dets[i, 4])))

        self.tracks = [t for t in self.tracks if t.since_update <= self.max_age]
        return [
            t for t in self.tracks
            if t.since_update < 1 and (t.hits >= self.min_hits or self.frame_count <= self.min_hits)
        ]


# ----------------------------------------------------------------------------
# Visual helpers
# ----------------------------------------------------------------------------
def id_color(track_id):
    """Stable, vivid colour per ID (golden-ratio hue spacing)."""
    hue = int((track_id * 0.618033988749895 % 1.0) * 179)
    bgr = cv2.cvtColor(np.uint8([[[hue, 220, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
    return tuple(int(c) for c in bgr)


def draw_trail(frame, trail, color):
    pts = list(trail)
    for i in range(1, len(pts)):
        fade = i / len(pts)
        cv2.line(frame, pts[i - 1], pts[i], color, max(1, int(4 * fade)))


def draw_label(frame, text, x, y, color):
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (x, y - h - 8), (x + w + 6, y), color, -1)
    cv2.putText(frame, text, (x + 3, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


def draw_hud(frame, fps, live, tallies, crossed_down, crossed_up):
    lines = [f"FPS {fps:4.1f}", f"Tracking {live}", f"Line  v{crossed_down}  ^{crossed_up}"]
    lines += [f"{n}: {c}" for n, c in sorted(tallies.items(), key=lambda kv: -kv[1])[:4]]
    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (200, 18 + 22 * len(lines)), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (16, 30 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1, cv2.LINE_AA)


# ----------------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0", help="0 for webcam or a video path")
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--line", type=float, default=0.55, help="tripwire height (0-1)")
    ap.add_argument("--save", default="", help="optional output video path")
    args = ap.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open source: {args.source}")

    model = YOLO(args.model)
    names = model.names
    tracker = Sort(max_age=20, min_hits=3, iou_thresh=0.3)

    ok, frame = cap.read()
    if not ok:
        raise SystemExit("No frames received.")
    H, W = frame.shape[:2]
    line_y = int(H * args.line)
    heat = np.zeros((H, W), np.float32)

    writer = None
    if args.save:
        writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), 30, (W, H))

    last_side = {}                 # track id -> which side of the tripwire
    counted = set()
    down = up = 0
    show_heat, show_trails, paused = False, True, False
    fps, t_prev = 0.0, time.time()

    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                break

            # --- detection ---
            res = model(frame, conf=args.conf, verbose=False)[0]
            if len(res.boxes):
                dets = np.hstack([
                    res.boxes.xyxy.cpu().numpy(),
                    res.boxes.conf.cpu().numpy()[:, None],
                    res.boxes.cls.cpu().numpy()[:, None],
                ])
            else:
                dets = np.zeros((0, 6))

            # --- tracking ---
            active = tracker.update(dets)
            tallies = defaultdict(int)

            for t in active:
                x1, y1, x2, y2 = map(int, t.box)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                color = id_color(t.id)
                t.trail.append((cx, cy))
                tallies[names[t.cls]] += 1

                # heatmap accumulation
                cv2.circle(heat, (cx, cy), 25, 1.0, -1)

                # tripwire crossing logic
                side = cy > line_y
                if t.id in last_side and last_side[t.id] != side and t.id not in counted:
                    counted.add(t.id)
                    if side:
                        down += 1
                    else:
                        up += 1
                last_side[t.id] = side

                # drawing
                if show_trails:
                    draw_trail(frame, t.trail, color)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                dwell = time.time() - t.born
                draw_label(frame, f"#{t.id} {names[t.cls]} {t.conf:.0%} {dwell:.0f}s", x1, y1, color)

            if show_heat:
                norm = cv2.normalize(heat, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                colored = cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)
                frame = cv2.addWeighted(frame, 0.65, colored, 0.35, 0)
            heat *= 0.995  # slow fade so the map stays current

            # tripwire + HUD
            cv2.line(frame, (0, line_y), (W, line_y), (0, 255, 255), 2)
            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - t_prev, 1e-6))
            t_prev = now
            draw_hud(frame, fps, len(active), tallies, down, up)

            if writer:
                writer.write(frame)

        cv2.imshow("Detection + Tracking", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("h"):
            show_heat = not show_heat
        elif key == ord("t"):
            show_trails = not show_trails
        elif key == ord(" "):
            paused = not paused

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
