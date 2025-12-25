from flask import Blueprint, jsonify
from utils.sheet_updater import update_congestion_score

auto_labeled_bp = Blueprint('auto_labeled', __name__)

@auto_labeled_bp.route('/auto-label', methods=['POST', 'GET'])
def auto_label_data():
    """
    api to automatically label data on google sheets
    
    logic:
    - if manual scores exist at min/max area in each hour: use interpolation
    - otherwise: use formula gate * (0.6 * density + 0.4 * chaos) with baseline 0.1
    """
    try:
        result = update_congestion_score()
        status_code = 200 if result.get("status") == "success" else 500
        return jsonify(result), status_code
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
