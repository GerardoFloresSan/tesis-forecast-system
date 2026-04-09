from fastapi import APIRouter
from app.schemas.auth import LoginRequest, TokenResponse

router = APIRouter(prefix="/auth", tags=["Auth"])

@router.post("/login", response_model=TokenResponse)
def login(data: LoginRequest):
    return {"access_token": "fake-token-for-now", "token_type": "bearer"}