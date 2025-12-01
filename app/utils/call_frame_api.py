import requests
import time
import os
from datetime import datetime
import numpy as np
import cv2
import warnings
from config.base_config import BASE_APICALL_VALUE
from utils.segment_util import detect_road_area

warnings.filterwarnings("ignore")

def parse_cookie_string(s):
    return dict(part.split("=", 1) for part in s.split("; ") if "=" in part)

session = requests.Session()
session.headers.update(BASE_APICALL_VALUE['HEADERS'])
session.cookies.update(parse_cookie_string(BASE_APICALL_VALUE['COOKIE_STR']))

def get_camera_frame(camera_base_url: str, timestamp: datetime = None):
    """
    get frame from URL camera.
    return frame OpenCV (np.ndarray) hoặc None nếu thất bại.
    """
    try:
        if timestamp is None:
            timestamp = datetime.now()
            
        ts = int(timestamp.timestamp() * 1000)
        full_url = (
            f"{camera_base_url}"
            f"&bg={BASE_APICALL_VALUE['FRAME_BG']}"
            f"&w={BASE_APICALL_VALUE['FRAME_W']}"
            f"&h={BASE_APICALL_VALUE['FRAME_H']}"
            f"&t={ts}"
        )

        r = session.get(full_url, timeout=10, verify=False)
        if r.status_code != 200 or r.content[:4] == b"<!DO":
            print(f"[WARNING] API call failed ({r.status_code})")
            return None

        img = np.frombuffer(r.content, np.uint8)
        frame = cv2.imdecode(img, cv2.IMREAD_COLOR)
        
        if frame is None:
            print("[WARNING] Frame decode failed.")
            return None
        
        # Save frame to disk
        save_dir = os.path.join("data", "frames")
        os.makedirs(save_dir, exist_ok=True)
        
        timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp_str}.jpg"
        save_path = os.path.join(save_dir, filename)
        
        cv2.imwrite(save_path, frame)
        # print(f"[INFO] Saved frame to {save_path}")

        # Test road detection
        road_pixels = detect_road_area(frame)
        print(f"Road pixels: {road_pixels}")
        
        return frame

    except Exception as e:
        print(f"[ERROR] in get_camera_frame: {e}")
        return None