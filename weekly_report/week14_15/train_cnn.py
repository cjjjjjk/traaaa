import os
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models

# 1. Định nghĩa Dataset
class TrafficDataset(Dataset):
    def __init__(self, csv_path, image_dir, transform=None):
        df = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.transform = transform
        
        # Filter out images that don't exist
        valid_indices = []
        print(f"Checking {len(df)} images...")
        existing_count = 0
        for idx, row in df.iterrows():
            img_path = os.path.join(self.image_dir, row["image_name"])
            if os.path.exists(img_path):
                valid_indices.append(idx)
                existing_count += 1
        
        self.data = df.iloc[valid_indices].reset_index(drop=True)
        print(f"Filtered dataset: {existing_count}/{len(df)} images found.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx]["image_name"]
        # Sửa lỗi: Thay thế dấu phẩy bằng dấu chấm để chuyển sang float
        score_str = str(self.data.iloc[idx]["congestion_score"])
        try:
            y = float(score_str.replace(',', '.'))
        except ValueError:
            y = 0.0 # Handle potential parsing errors

        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(y, dtype=torch.float32)

# 2. Transforms
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 3. Chuẩn bị DataLoader
# Đường dẫn tương đối từ thư mục chạy script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, 'dataset', 'label.csv')
IMG_DIR = os.path.join(BASE_DIR, 'dataset', 'images')

print(f"Loading data from {CSV_PATH}")

# Tạo dataset với transform None trước để lọc ảnh
temp_dataset = TrafficDataset(CSV_PATH, IMG_DIR, transform=None)
n_samples = len(temp_dataset)  # Use filtered dataset length
n_train = int(0.8 * n_samples)
n_val = n_samples - n_train

# Tạo generator để split cố định
generator = torch.Generator().manual_seed(42)
train_indices, val_indices = random_split(range(n_samples), [n_train, n_val], generator=generator)

# Tạo 2 instance dataset riêng biệt để áp dụng transform khác nhau
train_ds_full = TrafficDataset(CSV_PATH, IMG_DIR, transform=train_transform)
val_ds_full = TrafficDataset(CSV_PATH, IMG_DIR, transform=val_transform)

train_dataset = torch.utils.data.Subset(train_ds_full, train_indices.indices)
val_dataset = torch.utils.data.Subset(val_ds_full, val_indices.indices)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

print(f"Data loaded: {len(train_dataset)} train images, {len(val_dataset)} val images")

# 4. Định nghĩa Model
class CongestionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Load pretrained ResNet18
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
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

# 5. Setup Training
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model = CongestionCNN().to(device)

# Freeze feature extractor initially
for param in model.feature_extractor.parameters():
    param.requires_grad = False

criterion = nn.SmoothL1Loss()
optimizer = torch.optim.Adam(model.regressor.parameters(), lr=1e-3)

# 6. Training Functions
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    return total_loss / len(loader)

# 7. Run Training
print("Starting initial training (Head Only)...")
for epoch in range(10): 
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
    val_loss = validate(model, val_loader, criterion, device)
    print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

# 8. Fine-tuning
print("\nStarting fine-tuning (Last 2 layers + Head)...")
for param in model.feature_extractor[-2:].parameters():
    param.requires_grad = True

optimizer = torch.optim.Adam([
    {"params": model.feature_extractor.parameters(), "lr": 1e-5},
    {"params": model.regressor.parameters(), "lr": 1e-4}
])

for epoch in range(10): 
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
    val_loss = validate(model, val_loader, criterion, device)
    print(f"Fine-tune Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

print("\nTraining complete.")

# 9. Save model
MODEL_SAVE_PATH = os.path.join(BASE_DIR,"model", 'congestion_cnn_model.pth')
torch.save({
    'model_state_dict': model.state_dict(),
    'model_architecture': 'ResNet18-based CNN',
    'input_size': (224, 224),
    'output_range': (0, 1),
}, MODEL_SAVE_PATH)
print(f"Model saved to: {MODEL_SAVE_PATH}")
