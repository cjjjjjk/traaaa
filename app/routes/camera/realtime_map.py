from flask import Blueprint, Response, request
import cv2
import time
import numpy as np

# Import các hàm tiện ích đã được tách ra
from utils.call_frame_api import get_camera_frame
from utils.detect_util import analyze_frame
from config.base_config import VEHICLE_CLASS_COLORS, BASE_APICALL_VALUE

realtime_map_bp = Blueprint("realtime_map", __name__)

@realtime_map_bp.route("/realtime-map", methods=["GET"])
def realtime_map():
    """
    Cung cấp một map thời gian thực hiển thị các vector phương tiện.
    /realtime-map?url=https://giaothong.hochiminhcity.gov.vn:8007/Render/CameraHandler.ashx?id=587ee2aeb807da0011e33d52
    """
    camera_base = request.args.get("url")
    if not camera_base:
        return "need 'url'", 400

    # Frame size (white map)
    # try:
    #     FRAME_W = int(BASE_APICALL_VALUE["FRAME_W"])
    #     FRAME_H = int(BASE_APICALL_VALUE["FRAME_H"])
    # except:
    #     # Giá trị dự phòng nếu config lỗi
    #     FRAME_W = 400
    #     FRAME_H = 260

    def generate():
        while True:
            try:
                # 1. Gọi API lấy frame
                frame = get_camera_frame(camera_base)
                
                if frame is None:
                    print("[INFO] No frame received, retrying...")
                    time.sleep(2)
                    continue

                # 2. Gọi model detect
                detections = analyze_frame(frame)

                # 3. White map - optional
                # Kích thước map bằng kích thước frame (H, W, C)
                # map_image = np.full((FRAME_H, FRAME_W, 3), 255, dtype=np.uint8)

                # 4. Vẽ vector (mũi tên) lên map
                if detections:
                    for det in detections:
                        bbox = det["bbox"]
                        kpt = det["keypoint"]
                        cls_name = det["class"]
                        
                        color = VEHICLE_CLASS_COLORS.get(cls_name, (0, 0, 0)) 
                        
                        # Tính trung tâm bounding box
                        center_x = int((bbox[0] + bbox[2]) / 2)
                        center_y = int((bbox[1] + bbox[3]) / 2)
                        
                        # Lấy điểm kpt làm đỉnh
                        tip_x = int(kpt[0])
                        tip_y = int(kpt[1])
                        
                        tail_x = center_x*2 - tip_x
                        tail_y = center_y*2- tip_y
                        
                        # Vẽ mũi tên: tail -> tip
                        # cv2.arrowedLine(map_image,   # white map (optional)
                        cv2.arrowedLine(frame, 
                                        (tail_x, tail_y), 
                                        (tip_x, tip_y), 
                                        color, 
                                        thickness=1, 
                                        tipLength=0.2)

                # 5. Encode và yield map_image (ảnh trắng có vector)
                # _, buffer = cv2.imencode(".jpg", map_image)
                _, buffer = cv2.imencode(".jpg", frame)
                frame_bytes = buffer.tobytes()

                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

                time.sleep(10) 

            except Exception as e:
                print(f"[ERROR] in generate loop: {e}")
                time.sleep(2)
                continue

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")