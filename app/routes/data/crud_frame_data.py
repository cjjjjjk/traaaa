from flask import Blueprint, request, jsonify, Response
from utils.gg_sheet_setup import gg_sheet_config
import gspread 

data_bp = Blueprint('data_bp', __name__)

@data_bp.route('/data', methods=['GET'])
def get_all_data():
    try:
        sheet = gg_sheet_config()
        # get_all_records() trả về một danh sách các dictionary
        records = sheet.get_all_records()
        return jsonify(records), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# @data_bp.route('/data/<int:row_num>', methods=['GET'])
# def get_data_by_row(row_num):
#     try:
#         sheet = gg_sheet_config()
        
#         # Lấy header (dòng 1)
#         headers = sheet.row_values(1)
        
#         # Lấy dữ liệu của dòng cụ thể
#         values = sheet.row_values(row_num)
        
#         if not values:
#             return jsonify({"error": "Row not found"}), 404
            
#         # Kết hợp header và values thành dictionary
#         record = dict(zip(headers, values))
        
#         return jsonify(record), 200
#     except gspread.exceptions.APIError as e:
#         return jsonify({"error": "Row not found or API error", "details": str(e)}), 404
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

@data_bp.route('/data', methods=['POST'])
def add_data():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        sheet = gg_sheet_config()
        
        # header(line 1) để đảm bảo thứ tự cột là đúng
        headers = sheet.row_values(1)
        
        # list giá trị mới theo  header
        new_row = [data.get(header) for header in headers]
        
        # Thêm dòng mới vào cuối sheet
        # value_input_option='USER_ENTERED' để giữ nguyên định dạng (vd: số không bị coi là ngày)
        sheet.append_row(new_row, value_input_option='USER_ENTERED')
        
        return jsonify({"success": "Data added successfully"}), 201
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Route để cập nhật dữ liệu (UPDATE)
# @data_bp.route('/data/<int:row_num>', methods=['PUT'])
# def update_data(row_num):
#     try:
#         data = request.get_json()
#         if not data:
#             return jsonify({"error": "No data provided"}), 400

#         sheet = gg_sheet_config()
        
#         # Lấy header để tìm đúng cột cần cập nhật
#         headers = sheet.row_values(1)
        
#         # Chuẩn bị danh sách các cell cần cập nhật
#         cells_to_update = []
#         for key, value in data.items():
#             try:
#                 # Tìm vị trí cột (1-indexed)
#                 col_num = headers.index(key) + 1
#                 # Tạo một Cell object
#                 cell = gspread.cell.Cell(row_num, col_num, value)
#                 cells_to_update.append(cell)
#             except ValueError:
#                 # Bỏ qua nếu key từ JSON không có trong header
#                 print(f"Warning: Column '{key}' not found in headers.")
        
#         if not cells_to_update:
#             return jsonify({"error": "No valid columns to update"}), 400
            
#         # Cập nhật tất cả các cell cùng lúc
#         sheet.update_cells(cells_to_update, value_input_option='USER_ENTERED')
        
#         return jsonify({"success": f"Row {row_num} updated successfully"}), 200
        
#     except gspread.exceptions.APIError as e:
#         return jsonify({"error": "Row not found or API error", "details": str(e)}), 404
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# # Route để xoá dữ liệu (DELETE)
# @data_bp.route('/data/<int:row_num>', methods=['DELETE'])
# def delete_data(row_num):
#     try:
#         sheet = gg_sheet_config()
        
#         # Xoá dòng dựa trên số dòng
#         sheet.delete_rows(row_num)
        
#         return jsonify({"success": f"Row {row_num} deleted successfully"}), 200
#     except gspread.exceptions.APIError as e:
#         return jsonify({"error": "Row not found or API error", "details": str(e)}), 404
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500