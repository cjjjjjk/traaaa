from flask import Blueprint, Response, jsonify
import os
import time
import io
import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils.yolo_utils import analyze_frame

# ----------------------------------------
# Thiết lập Blueprint
map_bp = Blueprint("map", __name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
frames_dir = os.path.abspath(os.path.join(BASE_DIR, "..", "frames"))
print(f"[DEBUG] frames_dir: {frames_dir}")

if not os.path.exists(frames_dir):
    print(f"[ERROR] Frames directory not found: {frames_dir}")
# ----------------------------------------


@map_bp.route("/frame-map", methods=["GET"])
def stream_topdown_maps():
    def generate():
        try:
            image_files = sorted(
                [f for f in os.listdir(frames_dir) if f.lower().endswith(".jpg")]
            )
        except FileNotFoundError:
            print(f"[ERROR] Cannot find frames directory: {frames_dir}")
            return

        if not image_files:
            print(f"[WARNING] No .jpg files found in {frames_dir}")
            return

        # Duyệt từng frame và gửi biểu đồ tương ứng
        for idx, filename in enumerate(image_files):
            frame_path = os.path.join(frames_dir, filename)

            detections, _ = analyze_frame(frame_path)
            if not detections:
                print(f"[INFO] Frame {idx}: No detections found.")
                continue

            frame = cv2.imread(frame_path)
            if frame is None:
                print(f"[WARNING] Cannot read frame: {frame_path}")
                continue
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            pts_image = np.float32([[129, 91], [299, 96], [30, 200], [401, 216]])
            pts_real = np.float32([[5, 7], [12, 7], [5, 0], [12, 0]])

            M = cv2.getPerspectiveTransform(pts_image, pts_real)

            centers = np.float32([[d["center_x"], d["center_y"]] for d in detections])
            centers = np.expand_dims(centers, axis=0)
            transformed = cv2.perspectiveTransform(centers, M)[0]

            classes = [d["class"] for d in detections]
            X_real, Y_real = transformed[:, 0], transformed[:, 1]

            # ----------------------------------------
            plt.figure(figsize=(6, 5))
            unique_classes = set(classes)
            for cls in unique_classes:
                mask = [c == cls for c in classes]
                plt.scatter(X_real[mask], Y_real[mask], label=cls, alpha=0.8)

            plt.plot(
                [pts_real[0][0], pts_real[1][0], pts_real[3][0], pts_real[2][0], pts_real[0][0]],
                [pts_real[0][1], pts_real[1][1], pts_real[3][1], pts_real[2][1], pts_real[0][1]],
                'k--', label='Vùng thực tế'
            )

            # -------------------------------
            x_min, x_max = X_real.min(), X_real.max()
            y_min, y_max = Y_real.min(), Y_real.max()

            x_min_fixed, x_max_fixed = 0, 15
            y_min_fixed, y_max_fixed = 0, 12

            x_lim = (min(x_min, x_min_fixed), max(x_max, x_max_fixed))
            y_lim = (min(y_min, y_min_fixed), max(y_max, y_max_fixed))

            plt.xlim(x_lim)
            plt.ylim(y_lim)

            # Giữ tỷ lệ 1:1 giữa X và Y
            plt.gca().set_aspect("equal", adjustable="box")

            # -------------------------------
            plt.xlabel("X (m)")
            plt.ylabel("Y (m)")
            plt.title(f"Top-down view - Frame {idx}")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            # ----------------------------------------

            buf = io.BytesIO()
            plt.savefig(buf, format="jpg")
            plt.close()
            buf.seek(0)
            frame_bytes = buf.getvalue()

            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

            time.sleep(1)  # gửi mỗi 1 giây

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")
