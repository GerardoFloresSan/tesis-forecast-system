from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.security import create_access_token, get_password_hash, verify_password
from app.models.user import User


DEFAULT_ROLE = "analista_operativo"
DEFAULT_USERNAME = "analista"
DEFAULT_PASSWORD = "Analista123*"
DEFAULT_FULL_NAME = "Analista Operativo"


def ensure_default_user(db: Session) -> User:
    user = db.query(User).filter(User.username == DEFAULT_USERNAME).first()

    if user:
        return user

    user = User(
        username=DEFAULT_USERNAME,
        full_name=DEFAULT_FULL_NAME,
        role=DEFAULT_ROLE,
        hashed_password=get_password_hash(DEFAULT_PASSWORD),
        is_active=True,
    )

    db.add(user)
    db.commit()
    db.refresh(user)

    return user


def authenticate_user(db: Session, username: str, password: str) -> User | None:
    user = db.query(User).filter(User.username == username).first()

    if user is None:
        return None

    if not user.is_active:
        return None

    if not verify_password(password, user.hashed_password):
        return None

    return user


def build_token_response(user: User) -> dict:
    access_token = create_access_token(
        subject=user.username,
        role=user.role,
    )

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "expires_in": settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        "username": user.username,
        "full_name": user.full_name,
        "role": user.role,
    }