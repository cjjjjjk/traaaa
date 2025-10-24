from flask import Blueprint, Response, request
import cv2
import time
import numpy as np
import requests
from datetime import datetime
from utils.yolo_cam_util import analyze_frame_cam
import warnings
warnings.filterwarnings("ignore")

realtime_detect_bp = Blueprint("realtime_detect", __name__)

# Cookie cố định (giống như script hoạt động)
COOKIE_STR = """ASP.NET_SessionId=2pd400jxoiokdowndnju2zdr; .VDMS=E6D97CFABE16A23E4774230850C06957F36B5B3EA1FF8360A13AFBE6CB680692EF54AF6792F4900295E280829B09309FC1C42F3998D647739B7CDA12A3BBEC339828A6DCD87CE1E69EDE5FED4D54D3E641C6DE126BC213E650AF93CA7D237E3F0BBF1FF20C50762E6AE7BB049B7EE780F66A625A; _frontend=!DXIcY2WO+fUXfJrZrha5HPS1wJuimy99qr25sd/0cSvsQGa2NxADtaW2cwiwYrp8paE1JECWJWCs904=; CurrentLanguage=vi; _ga=GA1.3.4923489.1760691959; _gid=GA1.3.274225042.1760691959; _pk_id.1.2f14=7192ec68653c6b65.1760691959.1.1760691959.1760691959.1760691959.; _pk_ses.1.2f14=*; _ga_JCXT8BPG4E=GS2.3.s1760691960$o1$g0$t1760691960$j60$l0$h0; TS01e7700a=0150c7cfd1dc293c4c03a2ac85fec94a52acc6f241c039622d7f59470ce112b88ed79d54a48de2bf7d912f31e100bf2d5ad97a2cb8a0be33f0f7e148086b7a1d1cf1550652ae0a33c727506fdea7d7a8d673254c321f8ba65848425d90a7cb03f63360c739"""

def parse_cookie_string(s):
    return dict(part.split("=", 1) for part in s.split("; ") if "=" in part)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36",
    "Accept": "image/webp,image/apng,image/*,*/*;q=0.8",
    "Accept-Language": "vi,en;q=0.9",
    "Referer": "https://giaothong.hochiminhcity.gov.vn/",
    "Sec-CH-UA": '"Google Chrome";v="141", "Not?A_Brand";v="8", "Chromium";v="141"',
    "Sec-CH-UA-Platform": '"Windows"',
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Upgrade-Insecure-Requests": "1",
}

COOKIES = parse_cookie_string(COOKIE_STR)

session = requests.Session()
session.headers.update(HEADERS)
session.cookies.update(COOKIES)


@realtime_detect_bp.route("/realtime-detect", methods=["GET"])
def realtime_detect():
    """
    Stream phát hiện đối tượng trực tiếp từ URL camera giao thông (ảnh .jpg có token).
    Ví dụ:
        /realtime-detect?url=https://giaothong.hochiminhcity.gov.vn:8007/Render/CameraHandler.ashx?id=587ee2aeb807da0011e33d52
    """
    camera_base = request.args.get("url")
    if not camera_base:
        return "Thiếu tham số 'url'", 400

    def generate():
        while True:
            try:
                ts = int(time.time() * 1000)
                full_url = f"{camera_base}&bg=black&w=400&h=260&t={ts}"

                r = session.get(full_url, timeout=10, verify=False)
                if r.status_code != 200 or r.content[:4] == b"<!DO":
                    print(f"[WARNING] Không lấy được ảnh ({r.status_code})")
                    time.sleep(1)
                    continue

                img = np.frombuffer(r.content, np.uint8)
                frame = cv2.imdecode(img, cv2.IMREAD_COLOR)

                if frame is None:
                    print("[WARNING] Không decode được frame.")
                    time.sleep(1)
                    continue

                detections, output_frame = analyze_frame_cam(frame)

                pts_image = np.float32([[129, 91], [299, 96], [401, 216], [30, 200]])
                pts_image_int = np.int32(pts_image)
                cv2.polylines(
                    output_frame,
                    [pts_image_int],
                    isClosed=True,
                    color=(84, 247,244),  
                    thickness=2,
                )
                cv2.putText(output_frame, "",
                            (int(pts_image[0][0]), int(pts_image[0][1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                _, buffer = cv2.imencode(".jpg", output_frame)
                frame_bytes = buffer.tobytes()

                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

                time.sleep(15)

            except Exception as e:
                print(f"[ERROR] Lỗi khi xử lý frame: {e}")
                time.sleep(2)
                continue

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")
