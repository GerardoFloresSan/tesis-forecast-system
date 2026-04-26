from pydantic import BaseModel, Field


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=100)
    password: str = Field(..., min_length=6, max_length=100)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    username: str
    full_name: str
    role: str


class CurrentUserResponse(BaseModel):
    id: int
    username: str
    full_name: str
    role: str
    is_active: bool