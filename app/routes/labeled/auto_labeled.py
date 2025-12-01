from flask import Blueprint, jsonify
from utils.sheet_updater import update_congestion_score

auto_labeled_bp = Blueprint('auto_labeled', __name__)

@auto_labeled_bp.route('/auto-label', methods=['POST', 'GET'])
def auto_label_data():
    """
    API to automatically label data on Google Sheets.
    Updates 'congestion_score' based on 'road_area_pixels'.
    """
    try:
        result = update_congestion_score()
        status_code = 200 if result.get("status") == "success" else 500
        return jsonify(result), status_code
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
