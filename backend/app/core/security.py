import base64
import hashlib
import hmac
import json
import time
from typing import Any

from app.core.config import settings


class TokenValidationError(Exception):
    pass


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("utf-8")


def _b64url_decode(data: str) -> bytes:
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + padding)


def get_password_hash(password: str) -> str:
    digest = hashlib.sha256(password.encode("utf-8")).hexdigest()
    return f"sha256${digest}"


def verify_password(plain_password: str, stored_password: str) -> bool:
    if not stored_password:
        return False

    if stored_password.startswith("sha256$"):
        expected = get_password_hash(plain_password)
        return hmac.compare_digest(expected, stored_password)

    return hmac.compare_digest(plain_password, stored_password)


def create_access_token(subject: str, expires_minutes: int | None = None) -> str:
    now = int(time.time())
    exp_minutes = expires_minutes or settings.ACCESS_TOKEN_EXPIRE_MINUTES
    exp = now + int(exp_minutes * 60)

    header = {
        "alg": "HS256",
        "typ": "JWT",
    }
    payload = {
        "sub": subject,
        "iat": now,
        "exp": exp,
    }

    header_segment = _b64url_encode(
        json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    )
    payload_segment = _b64url_encode(
        json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    )

    signing_input = f"{header_segment}.{payload_segment}".encode("utf-8")
    signature = hmac.new(
        settings.SECRET_KEY.encode("utf-8"),
        signing_input,
        hashlib.sha256,
    ).digest()
    signature_segment = _b64url_encode(signature)

    return f"{header_segment}.{payload_segment}.{signature_segment}"


def decode_access_token(token: str) -> dict[str, Any]:
    try:
        header_segment, payload_segment, signature_segment = token.split(".")
    except ValueError as exc:
        raise TokenValidationError("Token inválido.") from exc

    signing_input = f"{header_segment}.{payload_segment}".encode("utf-8")
    expected_signature = hmac.new(
        settings.SECRET_KEY.encode("utf-8"),
        signing_input,
        hashlib.sha256,
    ).digest()
    provided_signature = _b64url_decode(signature_segment)

    if not hmac.compare_digest(expected_signature, provided_signature):
        raise TokenValidationError("Firma del token inválida.")

    try:
        payload = json.loads(_b64url_decode(payload_segment).decode("utf-8"))
    except Exception as exc:
        raise TokenValidationError("Payload del token inválido.") from exc

    exp = payload.get("exp")
    sub = payload.get("sub")

    if not sub:
        raise TokenValidationError("Token sin sujeto válido.")

    if exp is None or int(exp) < int(time.time()):
        raise TokenValidationError("Token expirado.")

    return payload