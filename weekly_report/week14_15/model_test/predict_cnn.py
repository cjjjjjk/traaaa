import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# define the same model architecture
class CongestionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet18(weights=None)  # no pretrained weights needed for inference
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
        
        self.regressor = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        x = self.regressor(x)
        return x.squeeze(1)

# load trained model
def load_model(model_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CongestionCNN().to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded from {model_path}")
    print(f"Architecture: {checkpoint.get('model_architecture', 'N/A')}")
    print(f"Input size: {checkpoint.get('input_size', 'N/A')}")
    print(f"Output range: {checkpoint.get('output_range', 'N/A')}")
    
    return model, device

# predict congestion score from image
def predict_image(model, image_path, device):
    # same transform as validation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # load and transform image
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # predict
    with torch.no_grad():
        score = model(image_tensor).item()
    
    return score

# example usage
if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, 'congestion_cnn_model.pth')
    
    # load model
    model, device = load_model(MODEL_PATH)
    
    # example: predict on a test image
    # replace with actual image path
    test_image = os.path.join(BASE_DIR,"images", 'test_2.png')
    
    if os.path.exists(test_image):
        score = predict_image(model, test_image, device)
        print(f"\nTest image: {os.path.basename(test_image)}")
        print(f"Predicted congestion score: {score:.4f}")
    else:
        print(f"Test image not found: {test_image}")
        print("Please update the test_image path to an actual image file.")
