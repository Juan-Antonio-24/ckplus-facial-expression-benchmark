from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
from api.routers.resnet18 import router as resnet18_router
from api.routers.resnet34 import router as resnet34_router
from api.routers.resnet50 import router as resnet50_router
from api.routers.resnet101 import router as resnet101_router
from api.routers.mobilenet_v2 import router as mobilenet_v2_router
from api.routers.mobilenet_v3_small import router as mobilenet_v3_small_router
from api.routers.mobilenet_v3_large import router as mobilenet_v3_large_router
from api.routers.efficientnet_b0 import router as efficientnet_b0_router
from api.routers.efficientnet_b1 import router as efficientnet_b1_router
load_dotenv()

FRONTEND_URL = os.getenv("FRONTEND_URL")

app = FastAPI(
    title="CK+ Facial Expression Benchmark API",
    description="API for facial expression recognition using MobileNet, ResNet and EfficientNet architectures trained on the CK+ dataset.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"])

app.include_router(resnet18_router)
app.include_router(resnet34_router)
app.include_router(resnet50_router)
app.include_router(resnet101_router)
app.include_router(mobilenet_v2_router)
app.include_router(mobilenet_v3_small_router)
app.include_router(mobilenet_v3_large_router)
app.include_router(efficientnet_b0_router)
app.include_router(efficientnet_b1_router)