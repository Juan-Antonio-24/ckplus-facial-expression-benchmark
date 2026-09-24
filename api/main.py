from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
from api.routers.resnet18 import router as resnet18_router

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