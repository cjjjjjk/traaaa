import torch
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
from PIL import Image
import numpy as np
import os

def count_road_pixels():
    # Kiểm tra xem có GPU không (CUDA), nếu không thì dùng CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Đang sử dụng thiết bị: {device}")

    # Mô hình segmentation
    model_name = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
    
    # Thư mục chứa ảnh
    base_dir = "."
    img_dir = os.path.join(base_dir, "data_set", "images")

    # Tải mô hình
    try:
        print("Đang tải mô hình AI. Vui lòng chờ...")
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = AutoModelForSemanticSegmentation.from_pretrained(model_name).to(device)
        print("Tải mô hình thành công.")
    except Exception as e:
        print(f"Lỗi khi tải mô hình: {e}")
        return

    # Lấy danh sách ảnh
    image_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print(f"Không tìm thấy ảnh nào trong thư mục: {img_dir}")
        return

    print(f"\nTìm thấy {len(image_files)} ảnh. Bắt đầu xử lý...\n")

    road_pixel_counts = {}  # Lưu kết quả cho từng ảnh

    for i, filename in enumerate(image_files):
        img_path = os.path.join(img_dir, filename)
        
        try:
            # Mở ảnh
            image = Image.open(img_path).convert("RGB")
            original_size = image.size  

            # Chuẩn bị ảnh
            inputs = processor(images=image, return_tensors="pt").to(device)

            # Suy luận
            with torch.no_grad():
                outputs = model(**inputs)

            logits = outputs.logits
            upscaled_logits = torch.nn.functional.interpolate(
                logits,
                size=original_size[::-1],  # (height, width)
                mode="bilinear",
                align_corners=False
            )

            pred_seg = upscaled_logits.argmax(dim=1)[0].cpu()

            # Tạo mask cho lớp "road" (Cityscapes: class_id = 0)
            road_mask = (pred_seg == 0).byte()
            
            # Đếm số pixel thuộc lớp "road"
            road_pixel_count = int(road_mask.sum().item())

            # Lưu kết quả
            road_pixel_counts[filename] = {
                "road_pixels": road_pixel_count
            }

            print(f"[{i+1}/{len(image_files)}] {filename}: {road_pixel_count} pixel đường)")

        except Exception as e:
            print(f"[LỖI] Xảy ra lỗi với ảnh {filename}: {e}")
    return road_pixel_counts


# --- Chạy hàm chính ---
if __name__ == "__main__":
    results = count_road_pixels()
