from fastapi import APIRouter, UploadFile, File
import torch
from PIL import Image
import torchvision.transforms as T
import torch.nn as nn
from pathlib import Path
from torchvision import models

router = APIRouter(prefix="/resnet18", tags=["ResNet18"])

class CKPlus_ResNet18(nn.Module):
    def __init__(self, num_class=8, freeze_backbone=False):
        super().__init__()

        self.backbone = models.resnet18(weights="IMAGENET1K_V1")

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        in_features = self.backbone.fc.in_features

        self.backbone.fc = nn.Linear(in_features, num_class)

    def forward(self, x):
        return self.backbone(x)
    
label_dataset = {
    0: "Neutral",
    1: "Anger",
    2: "Contempt",
    3: "Disgust",
    4: "Fear",
    5: "Happiness",
    6: "Sadness",
    7: "Surprise"
}

height_width = 224

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CKPlus_ResNet18()

BASE_DIR = Path(__file__).resolve().parent
model_path = BASE_DIR.parent.parent / "model" / "resnet18" / "CKPlus_ResNet18_best_weights.pth"
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)

model.eval()

transform_val = T.Compose([
    T.Resize((height_width, height_width)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

@router.post("/predict")
async def predict(file: UploadFile = File(...)):

    img = Image.open(file.file).convert("L").convert("RGB")
    img = transform_val(img).unsqueeze(0)
    img = img.to(device)

    with torch.no_grad():
        output = model(img)
        pred = torch.softmax(output, dim=1)
        pred = pred.squeeze().cpu().numpy()
    
    pred_result = {}

    for i in range(len(pred)):
        pred_result[label_dataset[i]] = round(float(pred[i]) * 100, 2)
    
    pred_result = dict(sorted(pred_result.items(), key= lambda x: x[1], reverse=True))

    return {"Emotions": pred_result}