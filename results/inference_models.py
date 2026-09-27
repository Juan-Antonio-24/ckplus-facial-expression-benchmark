import torch
from PIL import Image
import torchvision.transforms as T
import torch.nn as nn
from torchvision import models
from pathlib import Path
from PIL import Image

# ======================================== ResNet18 ========================================

def ResNet18(image_path):
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
    BASE_DIR = Path.cwd()

    model_path = (BASE_DIR.parent / "model" / "resnet18" / "CKPlus_ResNet18_best_weights.pth")
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

    img = Image.open(image_path)
    display_img = img.copy()
    img = img.convert("L").convert("RGB")
    img = transform_val(img).unsqueeze(0)
    img = img.to(device)

    with torch.no_grad():
        output = model(img)
        pred = torch.softmax(output, dim=1)
        pred = pred.squeeze().cpu().numpy()

    pred_index = pred.argmax()

    pred_emotion = label_dataset[pred_index]

    architectural_name = "ResNet18"

    return display_img, pred_emotion, architectural_name

# ======================================== ResNet34 ========================================

def ResNet34(image_path):
    class CKPlus_ResNet34(nn.Module):
        def __init__(self, num_class=8, freeze_backbone=False):
            super().__init__()

            self.backbone = models.resnet34(weights="IMAGENET1K_V1")

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
    model = CKPlus_ResNet34()
    BASE_DIR = Path.cwd()

    model_path = (BASE_DIR.parent / "model" / "resnet34" / "CKPlus_ResNet34_best_weights.pth")
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

    img = Image.open(image_path)
    display_img = img.copy()
    img = img.convert("L").convert("RGB")
    img = transform_val(img).unsqueeze(0)
    img = img.to(device)

    with torch.no_grad():
        output = model(img)
        pred = torch.softmax(output, dim=1)
        pred = pred.squeeze().cpu().numpy()

    pred_index = pred.argmax()

    pred_emotion = label_dataset[pred_index]

    architectural_name = "ResNet34"

    return display_img, pred_emotion, architectural_name

# ======================================== ResNet50 ========================================

def ResNet50(image_path):
    class CKPlus_ResNet50(nn.Module):
        def __init__(self, num_class=8, freeze_backbone=False):
            super().__init__()

            self.backbone = models.resnet50(weights="IMAGENET1K_V2")

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
    model = CKPlus_ResNet50()
    BASE_DIR = Path.cwd()

    model_path = (BASE_DIR.parent / "model" / "resnet50" / "CKPlus_ResNet50_best_weights.pth")
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

    img = Image.open(image_path)
    display_img = img.copy()
    img = img.convert("L").convert("RGB")
    img = transform_val(img).unsqueeze(0)
    img = img.to(device)

    with torch.no_grad():
        output = model(img)
        pred = torch.softmax(output, dim=1)
        pred = pred.squeeze().cpu().numpy()

    pred_index = pred.argmax()

    pred_emotion = label_dataset[pred_index]

    architectural_name = "ResNet50"

    return display_img, pred_emotion, architectural_name

# ======================================== ResNet101 ========================================

def ResNet101(image_path):
    class CKPlus_ResNet101(nn.Module):
        def __init__(self, num_class=8, freeze_backbone=False):
            super().__init__()

            self.backbone = models.resnet101(weights="IMAGENET1K_V2")

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
    model = CKPlus_ResNet101()
    BASE_DIR = Path.cwd()

    model_path = BASE_DIR.parent / "model" / "resnet101" / "CKPlus_ResNet101_best_weights.pth"
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

    img = Image.open(image_path)
    display_img = img.copy()
    img = img.convert("L").convert("RGB")
    img = transform_val(img).unsqueeze(0)
    img = img.to(device)

    with torch.no_grad():
        output = model(img)
        pred = torch.softmax(output, dim=1)
        pred = pred.squeeze().cpu().numpy()

    pred_index = pred.argmax()

    pred_emotion = label_dataset[pred_index]

    architectural_name = "ResNet101"

    return display_img, pred_emotion, architectural_name

# ======================================== MobilenetV2 ========================================

def MobilenetV2(image_path):
    class CKPlus_MobileNetV2(nn.Module):
        def __init__(self, num_class=8, freeze_backbone=False):
            super().__init__()

            self.backbone = models.mobilenet_v2(weights="IMAGENET1K_V2")

            if freeze_backbone:
                for param in self.backbone.parameters():
                    param.requires_grad = False

            in_features = self.backbone.classifier[-1].in_features

            self.backbone.classifier[-1] = nn.Linear(in_features, num_class)

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
    model = CKPlus_MobileNetV2()
    BASE_DIR = Path.cwd()

    model_path = BASE_DIR.parent / "model" / "mobilenet_v2" / "CKPlus_MobileNetV2_best_weights.pth"
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

    img = Image.open(image_path)
    display_img = img.copy()
    img = img.convert("L").convert("RGB")
    img = transform_val(img).unsqueeze(0)
    img = img.to(device)

    with torch.no_grad():
        output = model(img)
        pred = torch.softmax(output, dim=1)
        pred = pred.squeeze().cpu().numpy()

    pred_index = pred.argmax()

    pred_emotion = label_dataset[pred_index]

    architectural_name = "MobilenetV2"

    return display_img, pred_emotion, architectural_name

# ======================================== MobileNetV3 Large ========================================

def MobilenetV3Large(image_path):
    class CKPlus_MobileNetV3Large(nn.Module):
        def __init__(self, num_class=8, freeze_backbone=False):
            super().__init__()

            self.backbone = models.mobilenet_v3_large(weights="IMAGENET1K_V2")

            if freeze_backbone:
                for param in self.backbone.parameters():
                    param.requires_grad = False

            in_features = self.backbone.classifier[-1].in_features

            self.backbone.classifier[-1] = nn.Linear(in_features, num_class)

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
    model = CKPlus_MobileNetV3Large()
    BASE_DIR = Path.cwd()

    model_path = BASE_DIR.parent / "model" / "mobilenet_v3_large" / "CKPlus_MobileNetV3Large_best_weights.pth"
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

    img = Image.open(image_path)
    display_img = img.copy()
    img = img.convert("L").convert("RGB")
    img = transform_val(img).unsqueeze(0)
    img = img.to(device)

    with torch.no_grad():
        output = model(img)
        pred = torch.softmax(output, dim=1)
        pred = pred.squeeze().cpu().numpy()

    pred_index = pred.argmax()

    pred_emotion = label_dataset[pred_index]

    architectural_name = "MobilenetV3 Large"

    return display_img, pred_emotion, architectural_name

# ======================================== MobileNetV3 Large ========================================

def MobilenetV3Small(image_path):
    class CKPlus_MobileNetV3Small(nn.Module):
        def __init__(self, num_class=8, freeze_backbone=False):
            super().__init__()

            self.backbone = models.mobilenet_v3_small(weights="IMAGENET1K_V1")

            if freeze_backbone:
                for param in self.backbone.parameters():
                    param.requires_grad = False

            in_features = self.backbone.classifier[-1].in_features

            self.backbone.classifier[-1] = nn.Linear(in_features, num_class)

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
    model = CKPlus_MobileNetV3Small()
    BASE_DIR = Path.cwd()

    model_path = BASE_DIR.parent / "model" / "mobilenet_v3_small" / "CKPlus_MobileNetV3Small_best_weights.pth"
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

    img = Image.open(image_path)
    display_img = img.copy()
    img = img.convert("L").convert("RGB")
    img = transform_val(img).unsqueeze(0)
    img = img.to(device)

    with torch.no_grad():
        output = model(img)
        pred = torch.softmax(output, dim=1)
        pred = pred.squeeze().cpu().numpy()

    pred_index = pred.argmax()

    pred_emotion = label_dataset[pred_index]

    architectural_name = "MobilenetV3 Small"

    return display_img, pred_emotion, architectural_name

# ======================================== EfficientNet B0 ========================================

def EfficientNetB0(image_path):
    class CKPlus_EfficientNetB0(nn.Module):
        def __init__(self, num_class=8, freeze_backbone=False):
            super().__init__()

            self.backbone = models.efficientnet_b0(weights="IMAGENET1K_V1")

            if freeze_backbone:
                for param in self.backbone.parameters():
                    param.requires_grad = False

            in_features = self.backbone.classifier[-1].in_features

            self.backbone.classifier[-1] = nn.Linear(in_features, num_class)

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
    model = CKPlus_EfficientNetB0()
    BASE_DIR = Path.cwd()

    model_path = BASE_DIR.parent / "model" / "efficientnet_b0" / "CKPlus_EfficientNetB0_best_weights.pth"
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

    img = Image.open(image_path)
    display_img = img.copy()
    img = img.convert("L").convert("RGB")
    img = transform_val(img).unsqueeze(0)
    img = img.to(device)

    with torch.no_grad():
        output = model(img)
        pred = torch.softmax(output, dim=1)
        pred = pred.squeeze().cpu().numpy()

    pred_index = pred.argmax()

    pred_emotion = label_dataset[pred_index]

    architectural_name = "EfficientNetB0"

    return display_img, pred_emotion, architectural_name

# ======================================== EfficientNet B1 ========================================

def EfficientNetB1(image_path):
    class CKPlus_EfficientNetB1(nn.Module):
        def __init__(self, num_class=8, freeze_backbone=False):
            super().__init__()

            self.backbone = models.efficientnet_b1(weights="IMAGENET1K_V2")

            if freeze_backbone:
                for param in self.backbone.parameters():
                    param.requires_grad = False

            in_features = self.backbone.classifier[-1].in_features

            self.backbone.classifier[-1] = nn.Linear(in_features, num_class)

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

    height_width = 240

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CKPlus_EfficientNetB1()
    BASE_DIR = Path.cwd()

    model_path = BASE_DIR.parent / "model" / "efficientnet_b1" / "CKPlus_EfficientNetB1_best_weights.pth"
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

    img = Image.open(image_path)
    display_img = img.copy()
    img = img.convert("L").convert("RGB")
    img = transform_val(img).unsqueeze(0)
    img = img.to(device)

    with torch.no_grad():
        output = model(img)
        pred = torch.softmax(output, dim=1)
        pred = pred.squeeze().cpu().numpy()

    pred_index = pred.argmax()

    pred_emotion = label_dataset[pred_index]

    architectural_name = "EfficientNetB1"

    return display_img, pred_emotion, architectural_name

# ======================================== EfficientNet B2 ========================================

def EfficientNetB2(image_path):
    class CKPlus_EfficientNetB2(nn.Module):
        def __init__(self, num_class=8, freeze_backbone=False):
            super().__init__()

            self.backbone = models.efficientnet_b2(weights="IMAGENET1K_V1")

            if freeze_backbone:
                for param in self.backbone.parameters():
                    param.requires_grad = False

            in_features = self.backbone.classifier[-1].in_features

            self.backbone.classifier[-1] = nn.Linear(in_features, num_class)

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

    height_width = 288

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CKPlus_EfficientNetB2()
    BASE_DIR = Path.cwd()

    model_path = BASE_DIR.parent / "model" / "efficientnet_b2" / "CKPlus_EfficientNetB2_best_weights.pth"
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

    img = Image.open(image_path)
    display_img = img.copy()
    img = img.convert("L").convert("RGB")
    img = transform_val(img).unsqueeze(0)
    img = img.to(device)

    with torch.no_grad():
        output = model(img)
        pred = torch.softmax(output, dim=1)
        pred = pred.squeeze().cpu().numpy()

    pred_index = pred.argmax()

    pred_emotion = label_dataset[pred_index]

    architectural_name = "EfficientNetB2"

    return display_img, pred_emotion, architectural_name

# ======================================== EfficientNet B3 ========================================

def EfficientNetB3(image_path):
    class CKPlus_EfficientNetB3(nn.Module):
        def __init__(self, num_class=8, freeze_backbone=False):
            super().__init__()

            self.backbone = models.efficientnet_b3(weights="IMAGENET1K_V1")

            if freeze_backbone:
                for param in self.backbone.parameters():
                    param.requires_grad = False

            in_features = self.backbone.classifier[-1].in_features

            self.backbone.classifier[-1] = nn.Linear(in_features, num_class)

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

    height_width = 300

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CKPlus_EfficientNetB3()
    BASE_DIR = Path.cwd()

    model_path = BASE_DIR.parent / "model" / "efficientnet_b3" / "CKPlus_EfficientNetB3_best_weights.pth"
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

    img = Image.open(image_path)
    display_img = img.copy()
    img = img.convert("L").convert("RGB")
    img = transform_val(img).unsqueeze(0)
    img = img.to(device)

    with torch.no_grad():
        output = model(img)
        pred = torch.softmax(output, dim=1)
        pred = pred.squeeze().cpu().numpy()

    pred_index = pred.argmax()

    pred_emotion = label_dataset[pred_index]

    architectural_name = "EfficientNetB3"

    return display_img, pred_emotion, architectural_name

# ======================================== EfficientNet B4 ========================================

def EfficientNetB4(image_path):
    class CKPlus_EfficientNetB4(nn.Module):
        def __init__(self, num_class=8, freeze_backbone=False):
            super().__init__()

            self.backbone = models.efficientnet_b4(weights="IMAGENET1K_V1")

            if freeze_backbone:
                for param in self.backbone.parameters():
                    param.requires_grad = False

            in_features = self.backbone.classifier[-1].in_features

            self.backbone.classifier[-1] = nn.Linear(in_features, num_class)

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

    height_width = 380

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CKPlus_EfficientNetB4()
    BASE_DIR = Path.cwd()

    model_path = BASE_DIR.parent / "model" / "efficientnet_b4" / "CKPlus_EfficientNetB4_best_weights.pth"
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

    img = Image.open(image_path)
    display_img = img.copy()
    img = img.convert("L").convert("RGB")
    img = transform_val(img).unsqueeze(0)
    img = img.to(device)

    with torch.no_grad():
        output = model(img)
        pred = torch.softmax(output, dim=1)
        pred = pred.squeeze().cpu().numpy()

    pred_index = pred.argmax()

    pred_emotion = label_dataset[pred_index]

    architectural_name = "EfficientNetB4"

    return display_img, pred_emotion, architectural_name