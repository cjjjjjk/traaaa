from flask import Blueprint, request, jsonify, Response
from utils.gg_sheet_setup import gg_sheet_config
import gspread 
from datetime import datetime

data_bp = Blueprint('data_bp', __name__)

# --- HÀM HỖ TRỢ GHI DATA TRỰC TIẾP (Dành cho Crawler) ---
def insert_tracking_data(data_dict):
    """
    Hàm này được gọi trực tiếp từ realtime_crawler.py
    data_dict cần có các keys tương ứng với headers trên Google Sheet:
    [camera_id, timestamp, car_count, truck_count, bus_count, motorcycle_count, 
     road_area_pixels, vectors_chao_score, congestion_score]
    """
    try:
        sheet = gg_sheet_config()
        
        # Định nghĩa thứ tự cột cứng để đảm bảo đúng format yêu cầu
        headers = [
            "camera_id", "timestamp", "car_count", "truck_count", 
            "bus_count", "motorcycle_count", "road_area_pixels", 
            "vectors_chao_score", "congestion_score"
        ]
        
        # Tạo row data theo đúng thứ tự header
        # Sử dụng .get(key, 0) để điền 0 nếu thiếu dữ liệu (trừ camera_id và timestamp)
        row = [
            data_dict.get("camera_id", "unknown"),
            data_dict.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            data_dict.get("car_count", 0),
            data_dict.get("truck_count", 0),
            data_dict.get("bus_count", 0),
            data_dict.get("motorcycle_count", 0),
            data_dict.get("road_area_pixels", 0),
            data_dict.get("vectors_chao_score", 0.0),
            data_dict.get("congestion_score", 0)
        ]
        
        sheet.append_row(row, value_input_option='USER_ENTERED')
        print(f"[INFO] Logged data to Sheet for Camera: {data_dict.get('camera_id')}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to insert tracking data to Sheet: {e}")
        return False

# --- CÁC API ROUTE CŨ (GIỮ NGUYÊN) ---
@data_bp.route('/data', methods=['GET'])
def get_all_data():
    try:
        sheet = gg_sheet_config()
        records = sheet.get_all_records()
        return jsonify(records), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@data_bp.route('/data', methods=['POST'])
def add_data():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        sheet = gg_sheet_config()
        headers = sheet.row_values(1)
        new_row = [data.get(header) for header in headers]
        sheet.append_row(new_row, value_input_option='USER_ENTERED')
        
        return jsonify({"success": "Data added successfully"}), 201
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500