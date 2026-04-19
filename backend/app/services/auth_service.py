import os

from sqlalchemy.orm import Session

from app.core.security import create_access_token, get_password_hash, verify_password
from app.models.user import User


def get_default_admin_username() -> str:
    return os.getenv("DEFAULT_ADMIN_USERNAME", "admin")


def get_default_admin_password() -> str:
    return os.getenv("DEFAULT_ADMIN_PASSWORD", "Admin123*")


def ensure_default_user(db: Session) -> User:
    username = get_default_admin_username()
    password = get_default_admin_password()

    user = db.query(User).filter(User.username == username).first()
    if user:
        return user

    user = User(
        username=username,
        password=get_password_hash(password),
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def authenticate_user(db: Session, username: str, password: str) -> User | None:
    ensure_default_user(db)

    user = db.query(User).filter(User.username == username).first()
    if user is None:
        return None

    if not verify_password(password, user.password):
        return None

    return user


def build_token_response(user: User) -> dict:
    access_token = create_access_token(subject=user.username)
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "expires_in": 60 * 60,
        "username": user.username,
    }