import torch
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
from PIL import Image
import numpy as np
import os

def create_road_masks():
    """
    Hàm này đọc ảnh từ thư mục 'images', dùng AI để tạo mask lòng đường,
    và lưu kết quả vào thư mục 'masks'.
    """
    
    # --- 1. Cấu hình ---
    
    # Kiểm tra xem có GPU không (CUDA), nếu không thì dùng CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Đang sử dụng thiết bị: {device}")

    # Tên của mô hình AI từ Hugging Face
    # Mô hình này được huấn luyện trên Cityscapes, rất tốt cho ảnh giao thông
    model_name = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
    
    # Xác định các thư mục (dựa trên cấu trúc bạn cung cấp)
    base_dir = "." # Thư mục hiện tại
    img_dir = os.path.join(base_dir, "data_set", "images")
    mask_dir = os.path.join(base_dir, "data_set", "masks")

    # Đảm bảo thư mục "masks" tồn tại
    os.makedirs(mask_dir, exist_ok=True)

    print(f"Thư mục ảnh nguồn: {img_dir}")
    print(f"Thư mục mask đích:  {mask_dir}")

    # --- 2. Tải mô hình AI ---
    # (Việc này chỉ làm 1 lần, có thể mất vài phút để tải về)
    try:
        print("Đang tải mô hình AI. Vui lòng chờ...")
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = AutoModelForSemanticSegmentation.from_pretrained(model_name).to(device)
        print("Tải mô hình thành công.")
    except Exception as e:
        print(f"Lỗi khi tải mô hình: {e}")
        print("Vui lòng kiểm tra kết nối mạng và tên mô hình.")
        return

    # --- 3. Xử lý ảnh ---
    
    # Lấy danh sách các file ảnh
    image_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print(f"Không tìm thấy ảnh nào trong thư mục: {img_dir}")
        return

    print(f"\nTìm thấy {len(image_files)} ảnh. Bắt đầu xử lý...")

    for i, filename in enumerate(image_files):
        img_path = os.path.join(img_dir, filename)
        
        try:
            # Mở ảnh
            image = Image.open(img_path).convert("RGB")
            original_size = image.size # (width, height)

            # Đưa ảnh vào mô hình
            # 1. Chuẩn bị ảnh cho mô hình (resize, normalize,...)
            inputs = processor(images=image, return_tensors="pt").to(device)

            # 2. Chạy suy luận (inference)
            with torch.no_grad(): # Tắt tính toán gradient để tiết kiệm bộ nhớ
                outputs = model(**inputs)
            
            # 3. Lấy kết quả
            logits = outputs.logits # Đây là kết quả thô
            
            # 4. Upscale kết quả về kích thước ảnh gốc
            # Mô hình thường xử lý ở kích thước nhỏ hơn, ta cần phóng to mask
            upscaled_logits = torch.nn.functional.interpolate(
                logits,
                size=original_size[::-1], # (height, width) cho Pytorch
                mode="bilinear",
                align_corners=False
            )

            # 5. Tìm lớp (class) có xác suất cao nhất cho mỗi pixel
            # Kết quả là 1 tensor 2D, mỗi pixel có 1 ID (ví dụ: 0=road, 1=building,...)
            pred_seg = upscaled_logits.argmax(dim=1)[0].cpu()

            # --- 4. Tạo và lưu mask ---
            
            # Trong bộ dữ liệu Cityscapes, 'road' (lòng đường) có ID = 0
            # Tạo mask: Pixel nào là 'road' (ID=0) thì có giá trị 1, còn lại là 0
            road_mask = (pred_seg == 0).byte() 
            
            # Chuyển mask (0, 1) thành ảnh (0, 255)
            # 0 = màu đen (nền)
            # 255 = màu trắng (lòng đường)
            mask_np = road_mask.numpy() * 255
            
            # Chuyển mảng numpy thành ảnh PIL
            mask_image = Image.fromarray(mask_np.astype(np.uint8), mode='L') # 'L' = 8-bit grayscale
            
            # Tạo tên file .png cho mask
            mask_filename = os.path.splitext(filename)[0] + ".png"
            save_path = os.path.join(mask_dir, mask_filename)
            
            # Lưu ảnh
            mask_image.save(save_path)
            
            print(f"  [{i+1}/{len(image_files)}] Đã xử lý: {filename} -> Đã lưu: {mask_filename}")

        except Exception as e:
            print(f"  [LỖI] Xảy ra lỗi với ảnh {filename}: {e}")

    print("\nHoàn tất quá trình tạo mask!")

# --- Chạy hàm chính ---
if __name__ == "__main__":
    create_road_masks()