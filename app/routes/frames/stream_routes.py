from flask import Blueprint, Response, jsonify
import os
import time
from threading import Event
from utils.yolo_utils import analyze_frame
import cv2

stream_bp = Blueprint("stream", __name__)
frames_dir = "frames"
stop_event = Event()

@stream_bp.route("/frames", methods=["GET"])
def stream_frames():
    stop_event.clear()

    def generate():
        image_files = sorted(
            [f for f in os.listdir(frames_dir) if f.lower().endswith(".jpg")]
        )

        for idx, filename in enumerate(image_files):
            if stop_event.is_set():
                print("[INFO] Stream stopped.")
                break

            file_path = os.path.join(frames_dir, filename)

            # Run YOLO analysis and get frame with bounding boxes
            detections, frame_with_boxes = analyze_frame(file_path)

            # Log detection results
            print(f"\n[FRAME {idx}] {filename}")
            if not detections:
                print("  No objects detected.")
            else:
                for det in detections:
                    print(det)
                    # print(
                    #     f"  class={det['class']:<2} | conf={det['confidence']:.2f} "
                    #     f"| box=({det['x1']},{det['y1']},{det['x2']},{det['y2']})"
                    # )

            # Encode processed frame to JPEG
            ret, buffer = cv2.imencode(".jpg", frame_with_boxes)
            if not ret:
                continue
            frame_bytes = buffer.tobytes()

            # Stream to client
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

            time.sleep(2) 

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@stream_bp.route("/stop", methods=["POST"])
def stop_stream():
    stop_event.set()
    return jsonify({"message": "stream stopped"})
