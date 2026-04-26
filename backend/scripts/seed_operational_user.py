import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BACKEND_DIR))

from app.core.database import SessionLocal
from app.services.auth_service import ensure_default_user


def main():
    db = SessionLocal()

    try:
        user = ensure_default_user(db)
        print("Usuario inicial creado o validado correctamente:")
        print(f"ID: {user.id}")
        print(f"Username: {user.username}")
        print(f"Full name: {user.full_name}")
        print(f"Role: {user.role}")
        print(f"Activo: {user.is_active}")

    finally:
        db.close()


if __name__ == "__main__":
    main()