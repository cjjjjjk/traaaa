from flask import Blueprint, jsonify
import pandas as pd
import os
from utils.gg_sheet_setup import gg_sheet_config

sync_sheet_bp = Blueprint('sync_sheet', __name__)

@sync_sheet_bp.route('/sync-csv', methods=['POST', 'GET'])
def sync_sheet_to_csv():
    try:
        sheet = gg_sheet_config()
        # Get all values as strings
        data = sheet.get_all_values()
        
        if not data:
             return jsonify({"status": "warning", "message": "No data found in sheet"}), 404

        headers = data[0]
        rows = data[1:]
        
        if not rows:
             return jsonify({"status": "warning", "message": "No data rows found"}), 404

        df = pd.DataFrame(rows, columns=headers)
        
        # Determine root directory
        # Current file: app/routes/data/sync_sheet.py
        # Target: app/data/raw/database.csv
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Go up 2 levels to get to 'app' directory (app/routes/data -> app/routes -> app)
        app_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
        
        output_dir = os.path.join(app_dir, 'data', 'raw')
        os.makedirs(output_dir, exist_ok=True)
        
        output_file = os.path.join(output_dir, 'database.csv')
        
        df.to_csv(output_file, index=False)
        
        return jsonify({
            "status": "success", 
            "message": f"Saved {len(df)} rows to {output_file}",
            "path": output_file
        }), 200

    except Exception as e:
        print(f"[ERROR] sync_sheet_to_csv: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500
