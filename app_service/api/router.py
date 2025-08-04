from fastapi import APIRouter

from . import chatbot

api_router = APIRouter()

api_router.include_router(chatbot)