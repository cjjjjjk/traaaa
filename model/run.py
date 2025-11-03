from ultralytics import YOLO
import cv2
import os

# --- Cấu hình ---
MODEL_PATH = 'best.pt'
IMAGE_PATH = '20251023_073227.jpg'
# -----------------

def run_prediction():
    # 1. Kiểm tra file tồn tại
    if not os.path.exists(MODEL_PATH):
        print(f"Loi: Khong tim thay file model tai '{MODEL_PATH}'")
        return
    if not os.path.exists(IMAGE_PATH):
        print(f"Loi: Khong tim thay file anh tai '{IMAGE_PATH}'")
        return

    # 2. Tải mô hình YOLOv8-Pose đã huấn luyện
    # (Vì 'best.pt' nằm cùng thư mục, chỉ cần ghi tên file là đủ)
    print(f"Dang tai mo hinh tu {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)

    # 3. Đọc ảnh bằng OpenCV (để lấy kích thước gốc)
    img = cv2.imread(IMAGE_PATH)
    H, W, _ = img.shape
    print(f"Da tai anh: {IMAGE_PATH} (Kich thuoc: {W}x{H})")

    # 4. Chạy dự đoán (Inference)
    # 'conf=0.25' -> Chỉ giữ lại các phát hiện có độ tin cậy > 25%
    # 'imgsz=640' (Tùy chọn) -> Có thể chỉ định kích thước ảnh để dự đoán
    print("Dang chay du doan...")
    results = model(IMAGE_PATH, conf=0.25)

    # 5. Lấy ảnh đã được vẽ kết quả (annotated image)
    # results là một danh sách, ta lấy kết quả đầu tiên (index [0])
    # .plot() sẽ tự động vẽ:
    #   - Bounding boxes (hộp bao)
    #   - Class labels (tên lớp: car, bus,...)
    #   - Keypoints (điểm "đầu xe")
    annotated_frame = results[0].plot()

    # 6. Hiển thị kết quả bằng OpenCV
    print("Hien thi ket qua... (Nhan phim bat ky de dong)")
    cv2.imshow("YOLOv8 Pose Prediction", annotated_frame)
    
    # Đợi người dùng nhấn một phím bất kỳ để đóng cửa sổ
    cv2.waitKey(0)
    
    # Đóng tất cả cửa sổ OpenCV
    cv2.destroyAllWindows()
    print("Da dong chuong trinh.")

if __name__ == "__main__":
    run_prediction()