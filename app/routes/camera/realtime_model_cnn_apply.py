import cv2
from flask import Blueprint, request, jsonify, send_file, render_template_string
import numpy as np
import os
from datetime import datetime
from urllib.parse import urlparse, parse_qs
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageDraw, ImageFont
import io

# import utility functions
from utils.call_frame_api import get_camera_frame

realtime_model_cnn_apply_bp = Blueprint("realtime_model_cnn_apply", __name__)

# define cnn model architecture
class CongestionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet18(weights=None)
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
        
        self.regressor = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        x = self.regressor(x)
        return x.squeeze(1)

# load model and device at startup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', '..'))
MODEL_DIR = os.path.join(APP_DIR, 'utils', 'model', 'cnn')
APPLIED_DIR = os.path.join(APP_DIR, 'data', 'applied_cnn_2')

# ensure applied directory exists
os.makedirs(APPLIED_DIR, exist_ok=True)

model_path = os.path.join(MODEL_DIR, 'congestion_cnn_model_2.pth')

# try to load cnn model
try:    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cnn_model = CongestionCNN().to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    cnn_model.load_state_dict(checkpoint['model_state_dict'])
    cnn_model.eval()
    
    print(f"[INFO] loaded cnn model from {model_path}")
    print(f"[INFO] using device: {device}")
except Exception as e:
    print(f"[ERROR] failed to load cnn model: {e}")
    cnn_model = None
    device = None

# image transform for cnn input
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@realtime_model_cnn_apply_bp.route("/realtime-cnn-score", methods=["GET"])
def realtime_cnn_score():
    """
    route to apply cnn model on realtime camera frame.
    returns image with congestion score displayed on image (cnn model only, no yolo detection).
    
    query params:
        - url: camera url
        - return_json: if 'true', returns json with base64 image. otherwise returns image file.
    """
    full_camera_url = request.args.get("url")
    return_json = request.args.get("return_json", "false").lower() == "true"
    
    if not full_camera_url:
        return jsonify({
            "status": "error",
            "message": "missing parameter 'url'"
        }), 400
    
    # check if model is loaded
    if cnn_model is None or device is None:
        return jsonify({
            "status": "error",
            "message": "cnn model not loaded. please ensure model file exists at utils/model/cnn/congestion_cnn_model.pth"
        }), 500
    
    try:
        # extract camera_id from url
        parsed = urlparse(full_camera_url)
        captured_id = parse_qs(parsed.query).get('id')
        if captured_id:
            camera_id = captured_id[0]
        elif "id=" in full_camera_url:
            camera_id = full_camera_url.split("id=")[-1]
        else:
            camera_id = full_camera_url
    except:
        camera_id = full_camera_url
    
    try:
        current_time = datetime.now()
        
        # 1. get camera frame
        frame = get_camera_frame(full_camera_url, timestamp=current_time)
        
        if frame is None:
            return jsonify({
                "status": "error",
                "message": "failed to get frame from camera"
            }), 500
        
        # 2. apply cnn model to predict congestion score
        # convert frame to pil image (rgb)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        # transform and predict with cnn model
        image_tensor = transform(pil_image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            congestion_score = float(cnn_model(image_tensor).item())
        
        
        # 3. draw congestion score on image using pil for better text rendering
        # convert opencv frame to pil image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_frame = Image.fromarray(frame_rgb)
        draw = ImageDraw.Draw(pil_frame, 'RGBA')
        
        # score text
        score_text = f"{congestion_score:.4f}"
        
        # try to use a larger font, fallback to default if not available
        try:
            font_score = ImageFont.truetype("arial.ttf", 40)
        except:
            try:
                font_score = ImageFont.truetype("Arial.ttf", 40)
            except:
                font_score = ImageFont.load_default()
        
        # get text bounding box
        bbox = draw.textbbox((0, 0), score_text, font=font_score)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # position at top right with padding
        padding = 10
        x = pil_frame.width - text_width - padding
        y = padding
        
        # draw background rectangle with semi-transparency
        draw.rectangle(
            [x - 5, y - 5, x + text_width + 5, y + text_height + 5],
            fill=(0, 0, 0, 180)
        )
        
        # draw text - color based on congestion level
        # green if score < 0.65, red otherwise
        if congestion_score < 0.65:
            text_color = (0, 255, 0)
        else:
            text_color = (255, 0, 0)
        
        draw.text((x, y), score_text, fill=text_color, font=font_score)
        
        # add timestamp at bottom left
        timestamp_text = current_time.strftime("%Y-%m-%d %H:%M:%S")
        try:
            font_timestamp = ImageFont.truetype("arial.ttf", 20)
        except:
            try:
                font_timestamp = ImageFont.truetype("Arial.ttf", 20)
            except:
                font_timestamp = ImageFont.load_default()
        
        # draw timestamp with background
        ts_bbox = draw.textbbox((0, 0), timestamp_text, font=font_timestamp)
        ts_width = ts_bbox[2] - ts_bbox[0]
        ts_height = ts_bbox[3] - ts_bbox[1]
        ts_x = 10
        ts_y = pil_frame.height - ts_height - 10
        
        draw.rectangle(
            [ts_x - 3, ts_y - 3, ts_x + ts_width + 3, ts_y + ts_height + 3],
            fill=(0, 0, 0, 180)
        )
        draw.text((ts_x, ts_y), timestamp_text, fill=(255, 255, 255), font=font_timestamp)
        
        # convert back to opencv format
        frame = cv2.cvtColor(np.array(pil_frame), cv2.COLOR_RGB2BGR)
        
        # 7. save image
        file_name = current_time.strftime("%Y%m%d_%H%M%S.jpg")
        save_path = os.path.join(APPLIED_DIR, file_name)
        cv2.imwrite(save_path, frame)
        print(f"[INFO] saved cnn applied frame to {save_path}")
        
        # 4. prepare response
        model_prediction = {
            "camera_id": camera_id,
            "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
            "congestion_score": congestion_score,
            "model_type": "cnn",
            "architecture": "resnet18"
        }
        
        if return_json:
            # return json with base64 encoded image
            import base64
            _, buffer = cv2.imencode('.jpg', frame)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            
            return jsonify({
                "status": "success",
                "prediction": model_prediction,
                "image_base64": img_base64,
                "image_path": save_path
            }), 200
        else:
            # return image file directly
            _, buffer = cv2.imencode('.jpg', frame)
            img_io = io.BytesIO(buffer.tobytes())
            img_io.seek(0)
            
            return send_file(img_io, 
                           mimetype='image/jpeg',
                           as_attachment=False,
                           download_name=file_name)
        
    except Exception as e:
        print(f"[ERROR] realtime_cnn_score: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@realtime_model_cnn_apply_bp.route("/realtime-cnn-viewer", methods=["GET"])
def realtime_cnn_viewer():
    """
    html viewer route that auto-refreshes every 15 seconds.
    displays cnn prediction image with auto-reload.
    
    query params:
        - url: camera url
    """
    camera_url = request.args.get("url")
    
    if not camera_url:
        return "missing parameter 'url'", 400
    
    # html template with auto-refresh
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>realtime cnn congestion score</title>
        <meta charset="utf-8">
        <style>
            body {
                margin: 0;
                padding: 20px;
                background-color: #1a1a1a;
                font-family: 'Arial', sans-serif;
                color: #ffffff;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
            }
            h1 {
                text-align: center;
                color: #00ff88;
                margin-bottom: 10px;
            }
            .info {
                text-align: center;
                color: #888;
                margin-bottom: 20px;
                font-size: 14px;
            }
            .image-container {
                text-align: center;
                background-color: #2a2a2a;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
            }
            img {
                max-width: 100%;
                height: auto;
                border-radius: 5px;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.5);
            }
            .refresh-info {
                text-align: center;
                margin-top: 15px;
                color: #00ff88;
                font-size: 12px;
            }
            .countdown {
                color: #ffaa00;
                font-weight: bold;
            }
            .loading {
                text-align: center;
                padding: 40px;
                color: #888;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚦 realtime cnn congestion prediction</h1>
            <div class="info">
                auto-refresh every 15 seconds
            </div>
            
            <div class="image-container">
                <img id="cnn-image" src="/realtime/realtime-cnn-score?url={{ camera_url }}&t={{ timestamp }}" 
                     alt="cnn prediction" 
                     onerror="this.src='data:image/svg+xml,%3Csvg xmlns=%22http://www.w3.org/2000/svg%22 width=%22800%22 height=%22600%22%3E%3Crect fill=%22%23333%22 width=%22800%22 height=%22600%22/%3E%3Ctext fill=%22%23fff%22 x=%2250%25%22 y=%2250%25%22 text-anchor=%22middle%22%3Eloading...%3C/text%3E%3C/svg%3E'">
            </div>
            
            <div class="refresh-info">
                next refresh in <span class="countdown" id="countdown">15</span> seconds
            </div>
        </div>
        
        <script>
            let secondsLeft = 15;
            const countdownElement = document.getElementById('countdown');
            const imageElement = document.getElementById('cnn-image');
            const cameraUrl = '{{ camera_url }}';
            
            // function to refresh image
            function refreshImage() {
                const timestamp = new Date().getTime();
                const newSrc = '/realtime/realtime-cnn-score?url=' + encodeURIComponent(cameraUrl) + '&t=' + timestamp;
                console.log('refreshing image at:', new Date().toLocaleTimeString(), 'url:', newSrc);
                imageElement.src = newSrc;
                secondsLeft = 15;
            }
            
            // countdown timer (updates every second)
            setInterval(() => {
                secondsLeft--;
                if (secondsLeft <= 0) {
                    secondsLeft = 15;
                }
                countdownElement.textContent = secondsLeft;
            }, 1000);
            
            // refresh image every 15 seconds
            setInterval(refreshImage, 15000);
            
            // also refresh on page load after 1 second
            setTimeout(() => {
                console.log('initial refresh');
                refreshImage();
            }, 1000);
        </script>
    </body>
    </html>
    """
    
    import time
    current_timestamp = int(time.time() * 1000)
    
    return render_template_string(html_template, camera_url=camera_url, timestamp=current_timestamp)
