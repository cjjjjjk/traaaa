import pandas as pd
import gspread
import numpy as np
from utils.gg_sheet_setup import gg_sheet_config
from config.base_config import DATABASE_UPDATE_RULE

def compute_congestion_score(
    road_area_pixels,
    vector_chaos_score,
    area_min,
    area_max,
    baseline=0.1
):
    """
    compute congestion score using density and chaos
    
    formula:
    - area_norm: normalize road area by hour/camera min-max
    - density_score: 1 - area_norm (higher when road is crowded)
    - gate: prevent high chaos when road is empty
    - congestion_score: weighted combination of density and chaos
    - baseline: minimum score to avoid zero values
    
    returns: congestion score in [baseline, 1]
    """
    if area_max == area_min:
        area_norm = 0.0
    else:
        area_norm = (road_area_pixels - area_min) / (area_max - area_min)
    
    area_norm = min(max(area_norm, 0.0), 1.0)
    
    density_score = 1.0 - area_norm
    
    gate = (density_score - 0.3) / 0.4
    gate = min(max(gate, 0.0), 1.0)
    
    congestion_score = gate * (
        0.6 * density_score +
        0.4 * vector_chaos_score
    )
    
    congestion_score = min(max(congestion_score, 0.0), 1.0)
    
    congestion_score = baseline + congestion_score * (1.0 - baseline)
    
    return min(max(congestion_score, baseline), 1.0)


def update_congestion_score():
    try:
        sheet = gg_sheet_config()
        raw_data = sheet.get_all_values()
        
        if not raw_data:
            return {"status": "warning", "message": "no data found in sheet"}

        headers = raw_data[0]
        rows = raw_data[1:]
        
        if not rows:
             return {"status": "warning", "message": "no data rows found"}

        df = pd.DataFrame(rows, columns=headers)
        
        required_columns = ['timestamp', 'road_area_pixels', 'vectors_chao_score', 'camera_id']
        for col in required_columns:
            if col not in df.columns:
                return {"status": "error", "message": f"column '{col}' not found"}

        def clean_and_parse(x, is_score=False):
            try:
                if pd.isna(x) or str(x).strip() == "": 
                    return np.nan
                s = str(x).strip()
                
                if ',' in s and '.' in s:
                    if s.rfind(',') > s.rfind('.'):
                        s = s.replace('.', '').replace(',', '.')
                    else:
                        s = s.replace(',', '')
                elif ',' in s:
                    s = s.replace(',', '.')
                
                val = float(s)
                
                if is_score and val > 1.0:
                    if val <= 100.0:
                        val = val / 100.0
                    
                return val
            except:
                return np.nan

        df['road_area_pixels'] = df['road_area_pixels'].apply(lambda x: clean_and_parse(x))
        df['vectors_chao_score'] = df['vectors_chao_score'].apply(lambda x: clean_and_parse(x, is_score=True))
        
        if 'congestion_score' not in df.columns:
            df['congestion_score'] = ""
        
        df['score_numeric'] = df['congestion_score'].apply(lambda x: clean_and_parse(x, is_score=True))

        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour_group'] = df['timestamp'].dt.floor('H')
        
        new_scores = [None] * len(df)
        
        grouped = df.groupby(['camera_id', 'hour_group'])
        
        for name, group in grouped:
            if group.empty:
                continue
                
            min_area = group['road_area_pixels'].min()
            max_area = group['road_area_pixels'].max()
            
            scores_at_min = group[group['road_area_pixels'] == min_area]['score_numeric'].dropna()
            scores_at_max = group[group['road_area_pixels'] == max_area]['score_numeric'].dropna()
            
            scores_at_min = scores_at_min[scores_at_min.abs() < 1000]
            scores_at_max = scores_at_max[scores_at_max.abs() < 1000]
            
            use_manual_anchors = False
            score_at_min_area = None
            score_at_max_area = None
            
            if not scores_at_min.empty and not scores_at_max.empty:
                temp_min = scores_at_min.mean()
                temp_max = scores_at_max.mean()
                
                if abs(temp_min - temp_max) > 0.01:
                    use_manual_anchors = True
                    score_at_min_area = temp_min
                    score_at_max_area = temp_max
            
            for idx, row in group.iterrows():
                area = row['road_area_pixels']
                chaos = row['vectors_chao_score']
                
                if pd.isna(area):
                    new_scores[idx] = 0.1
                    continue
                
                if pd.isna(chaos):
                    chaos = 0.0
                
                if use_manual_anchors:
                    if max_area == min_area:
                        score = score_at_min_area
                    else:
                        numerator = (area - min_area) * (score_at_max_area - score_at_min_area)
                        denominator = max_area - min_area
                        score = score_at_min_area + (numerator / denominator)
                    new_scores[idx] = float(score)
                else:
                    score = compute_congestion_score(
                        road_area_pixels=area,
                        vector_chaos_score=chaos,
                        area_min=min_area,
                        area_max=max_area,
                        baseline=0.1
                    )
                    new_scores[idx] = float(score)


        if 'congestion_score' not in headers:
            sheet.update_cell(1, len(headers) + 1, 'congestion_score')
            col_index = len(headers) + 1
        else:
            col_index = headers.index('congestion_score') + 1
        
        start_line = DATABASE_UPDATE_RULE.get('update_from_line', 2)
        start_line = max(2, start_line)
        start_index = start_line - 2
        
        if start_index < len(new_scores):
            cell_values = [[round(float(s), 4) if s is not None else 0.0] for s in new_scores[start_index:]]
            
            start_cell = gspread.utils.rowcol_to_a1(start_line, col_index)
            end_cell = gspread.utils.rowcol_to_a1(start_line + len(cell_values) - 1, col_index)
            range_name = f"{start_cell}:{end_cell}"
            
            sheet.update(range_name, cell_values)
            
            return {"status": "success", "message": f"updated {len(cell_values)} rows starting from line {start_line}"}
        else:
            return {"status": "success", "message": f"no rows to update (start_line {start_line} beyond data length {len(new_scores)})"}
        
    except Exception as e:
        print(f"[ERROR] update_congestion_score: {e}")
        return {"status": "error", "message": str(e)}
