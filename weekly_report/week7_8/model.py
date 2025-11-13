from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

model_name = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"

processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModelForSemanticSegmentation.from_pretrained(model_name)

save_path = "./models/segformer-cityscapes"
processor.save_pretrained(save_path)
model.save_pretrained(save_path)

print(f"Đã lưu mô hình vào: {save_path}")
