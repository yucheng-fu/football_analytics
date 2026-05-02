from fastapi import APIRouter
from .routes import inference

api_router = APIRouter()

api_router.include_router(inference.router)
