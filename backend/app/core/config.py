from dotenv import load_dotenv
import os

load_dotenv()


def _to_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


class Settings:
    DATABASE_URL: str = os.getenv("DATABASE_URL", "")
    SECRET_KEY: str = os.getenv("SECRET_KEY", "default_secret")
    ALGORITHM: str = os.getenv("ALGORITHM", "HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 60))

    AUTO_RETRAIN_ENABLED: bool = _to_bool(os.getenv("AUTO_RETRAIN_ENABLED"), True)
    AUTO_RETRAIN_INTERVAL_MINUTES: int = int(os.getenv("AUTO_RETRAIN_INTERVAL_MINUTES", 60))
    AUTO_RETRAIN_THRESHOLD_MAPE: float = float(os.getenv("AUTO_RETRAIN_THRESHOLD_MAPE", 15.0))
    AUTO_RETRAIN_CHANNEL: str = os.getenv("AUTO_RETRAIN_CHANNEL", "Choice")

    AUTO_FORECAST_ENABLED: bool = _to_bool(os.getenv("AUTO_FORECAST_ENABLED"), True)
    AUTO_FORECAST_INTERVAL_MINUTES: int = int(os.getenv("AUTO_FORECAST_INTERVAL_MINUTES", 1440))
    AUTO_FORECAST_CHANNEL: str = os.getenv("AUTO_FORECAST_CHANNEL", "Choice")


settings = Settings()