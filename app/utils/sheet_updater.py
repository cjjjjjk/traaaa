import pandas as pd
import gspread
import numpy as np
from utils.gg_sheet_setup import gg_sheet_config

def update_congestion_score():
    try:
        sheet = gg_sheet_config()
        # Get all values as strings (list of lists) to avoid auto-conversion issues
        raw_data = sheet.get_all_values()
        
        if not raw_data:
            return {"status": "warning", "message": "No data found in sheet"}

        headers = raw_data[0]
        rows = raw_data[1:]
        
        if not rows:
             return {"status": "warning", "message": "No data rows found"}

        df = pd.DataFrame(rows, columns=headers)
        
        # Check if required columns exist
        required_columns = ['timestamp', 'road_area_pixels', 'camera_id']
        for col in required_columns:
            if col not in df.columns:
                return {"status": "error", "message": f"Column '{col}' not found"}

        # Helper to safely parse numbers and normalize
        def clean_and_parse(x, is_score=False):
            try:
                if pd.isna(x) or str(x).strip() == "": return np.nan
                s = str(x).strip()
                
                # Handle European format: 1.000,56 -> 1000.56; 0,56 -> 0.56
                if ',' in s and '.' in s:
                    if s.rfind(',') > s.rfind('.'): # Euro: 1.000,56
                        s = s.replace('.', '').replace(',', '.')
                    else: # US: 1,000.56
                        s = s.replace(',', '')
                elif ',' in s: # 0,56 or 1000,56
                    s = s.replace(',', '.')
                
                val = float(s)
                
                # Normalize if it's a score and > 1.0 (likely parsed 56.0 from 0,56 incorrectly before)
                if is_score and val > 1.0:
                    # Heuristic: if score is > 1 and <= 100, it might be scaled by 100
                    if val <= 100.0:
                        val = val / 100.0
                    
                return val
            except:
                return np.nan

        # Parse road_area_pixels
        df['road_area_pixels'] = df['road_area_pixels'].apply(lambda x: clean_and_parse(x))
        
        # Ensure congestion_score exists
        if 'congestion_score' not in df.columns:
            df['congestion_score'] = ""
        
        # Parse congestion_score
        df['score_numeric'] = df['congestion_score'].apply(lambda x: clean_and_parse(x, is_score=True))

        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Create grouping key (camera_id, hour)
        df['hour_group'] = df['timestamp'].dt.floor('H')
        
        # Initialize new scores list
        new_scores = [None] * len(df)
        
        # Group by camera_id and hour
        grouped = df.groupby(['camera_id', 'hour_group'])
        
        for name, group in grouped:
            if group.empty:
                continue
                
            # 1. Find min/max road_area_pixels
            min_area = group['road_area_pixels'].min()
            max_area = group['road_area_pixels'].max()
            
            # 2. Determine Anchors (score_at_min_area, score_at_max_area)
            score_at_min_area = 1.0 # Congested (Small area)
            score_at_max_area = 0.0 # Free (Large area)
            
            # Strategy: Look for manual updates specifically at the min/max area rows.
            scores_at_min = group[group['road_area_pixels'] == min_area]['score_numeric'].dropna()
            scores_at_max = group[group['road_area_pixels'] == max_area]['score_numeric'].dropna()
            
            # Filter out abnormal values
            scores_at_min = scores_at_min[scores_at_min.abs() < 1000]
            scores_at_max = scores_at_max[scores_at_max.abs() < 1000]

            if not scores_at_min.empty and not scores_at_max.empty:
                temp_min = scores_at_min.mean()
                temp_max = scores_at_max.mean()
                
                if temp_min != temp_max:
                    score_at_min_area = temp_min
                    score_at_max_area = temp_max
            
            # 3. Calculate Score for each row
            for idx, row in group.iterrows():
                area = row['road_area_pixels']
                
                if pd.isna(area):
                    new_scores[idx] = 0.0
                    continue
                
                if max_area == min_area:
                    new_scores[idx] = score_at_min_area
                else:
                    numerator = (area - min_area) * (score_at_max_area - score_at_min_area)
                    denominator = max_area - min_area
                    score = score_at_min_area + (numerator / denominator)
                    new_scores[idx] = float(score)

        # Update the sheet
        if 'congestion_score' not in headers:
            sheet.update_cell(1, len(headers) + 1, 'congestion_score')
            col_index = len(headers) + 1
        else:
            col_index = headers.index('congestion_score') + 1
            
        # Prepare column data
        # Write back as floats. 
        cell_values = [[float(s) if s is not None else 0.0] for s in new_scores]
        
        start_cell = gspread.utils.rowcol_to_a1(2, col_index)
        end_cell = gspread.utils.rowcol_to_a1(len(new_scores) + 1, col_index)
        range_name = f"{start_cell}:{end_cell}"
        
        sheet.update(range_name, cell_values)
        
        return {"status": "success", "message": f"Updated {len(new_scores)} rows"}
        
    except Exception as e:
        print(f"[ERROR] update_congestion_score: {e}")
        return {"status": "error", "message": str(e)}
