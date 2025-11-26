import gspread
import os
from google.oauth2.service_account import Credentials # Dùng thư viện mới này
from dotenv import load_dotenv

load_dotenv()

# Cấu hình credentials
google_creds = {
    "type": os.getenv("GOOGLE_TYPE"),
    "project_id": os.getenv("GOOGLE_PROJECT_ID"),
    "private_key_id": os.getenv("GOOGLE_PRIVATE_KEY_ID"),
    # Xử lý ký tự xuống dòng cho private key
    "private_key": os.getenv("GOOGLE_PRIVATE_KEY").replace('\\n', '\n') if os.getenv("GOOGLE_PRIVATE_KEY") else None,
    "client_email": os.getenv("GOOGLE_CLIENT_EMAIL"),
    "client_id": os.getenv("GOOGLE_CLIENT_ID"),
    "auth_uri": os.getenv("GOOGLE_AUTH_URI"),
    "token_uri": os.getenv("GOOGLE_TOKEN_URI"),
    "auth_provider_x509_cert_url": os.getenv("GOOGLE_AUTH_PROVIDER_CERT_URL"),
    "client_x509_cert_url": os.getenv("GOOGLE_CLIENT_CERT_URL")
}

# Scope mới chuẩn hơn
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]

def gg_sheet_config(sheet_number=1): 
    """
    Hàm này tạo kết nối mới mỗi khi được gọi để đảm bảo token luôn 'tươi',
    tránh lỗi hết hạn sau 60 phút.
    """
    try:
        # Tạo credentials từ dict
        creds = Credentials.from_service_account_info(google_creds, scopes=SCOPES)
        
        # Authorize client
        client = gspread.authorize(creds)
        
        SHEET_ID = os.getenv("SHEET_ID")
        
        # Mở spreadsheet và lấy sheet đầu tiên
        sheet = client.open_by_key(SHEET_ID).sheet1
        
        return sheet
        
    except Exception as e:
        print(f"[ERROR] Google Sheet Connection Error: {e}")
        raise e