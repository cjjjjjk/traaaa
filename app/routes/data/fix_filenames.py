from flask import Blueprint, jsonify
import os
import pandas as pd
from datetime import datetime
from utils.gg_sheet_setup import gg_sheet_config

fix_filenames_bp = Blueprint('fix_filenames', __name__)

@fix_filenames_bp.route('/fix-filenames', methods=['GET'])
def fix_filenames():
    try:
        # 1. Get data from Sheet
        sheet = gg_sheet_config()
        data = sheet.get_all_records()
        df = pd.DataFrame(data)
        
        # 2. Get all files
        # Assuming app is running from 'app' folder
        frames_dir = os.path.join(os.getcwd(), 'data', 'frames')
        if not os.path.exists(frames_dir):
             return jsonify({"status": "error", "message": f"Directory not found: {frames_dir}"}), 404
             
        files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])
        
        if not files:
            return jsonify({"status": "warning", "message": "No jpg files found"}), 200

        limit = min(len(df), len(files))
        renamed_count = 0
        errors = []
        
        # 3. Rename to temp first to avoid collisions
        temp_files = []
        for i in range(limit):
            old_path = os.path.join(frames_dir, files[i])
            temp_name = f"temp_fix_{i}.jpg"
            temp_path = os.path.join(frames_dir, temp_name)
            try:
                os.rename(old_path, temp_path)
                temp_files.append((i, temp_path))
            except Exception as e:
                errors.append(f"Error renaming to temp {files[i]}: {e}")

        # 4. Rename temp to final
        for i, temp_path in temp_files:
            try:
                row = df.iloc[i]
                sheet_time_str = str(row['timestamp']) # Expected YYYY-MM-DD HH:MM:SS
                
                # Parse timestamp
                # Handle potential format variations if needed, but assuming standard format
                dt = datetime.strptime(sheet_time_str, "%Y-%m-%d %H:%M:%S")
                new_name = dt.strftime("%Y%m%d_%H%M%S.jpg")
                new_path = os.path.join(frames_dir, new_name)
                
                os.rename(temp_path, new_path)
                renamed_count += 1
            except Exception as e:
                errors.append(f"Error renaming temp {i} to final: {e}")
                # Try to revert or leave as temp?
                # If we fail here, the file is stuck as temp_fix_i.jpg
                
        return jsonify({
            "status": "success", 
            "renamed_count": renamed_count, 
            "total_files_found": len(files),
            "total_rows_sheet": len(df),
            "errors": errors
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
