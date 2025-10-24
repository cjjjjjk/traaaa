import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
ALLOWED_CLASSES = ["car", "motorcycle", "bus", "truck"]

def analyze_frame_cam(frame):
    if frame is None:
        return [], None

    results = model(frame)
    detections = []

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls)
            cls_name = model.names[cls_id]
            if cls_name not in ALLOWED_CLASSES:
                continue

            conf = float(box.conf)
            x1, y1, x2, y2 = map(float, box.xyxy[0])

            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            detections.append({
                "class": cls_name,
                "confidence": conf,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "center_x": center_x,
                "center_y": center_y
            })

            # Vẽ bounding box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"{cls_name} {conf:.2f}", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return detections, frame
