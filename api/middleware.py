"""
Middleware для FastAPI:
- JWT-аутентификация
- Rate limiting (sliding window)
- Request/Response логирование
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt

from utils.logger import setup_logger

logger = setup_logger("hybridqa.api.middleware")


# ============================================================= Конфигурация
SECRET_KEY = "CHANGE_ME_IN_PRODUCTION"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

# Простые credentials для демо (в проде использовать БД)
DEMO_USERS = {
    "admin": "password123",
    "user": "user_password",
    "readonly": "readonly_pass",
}


# ============================================================= JWT utils
def create_access_token(
    data: dict,
    expires_delta: Optional[timedelta] = None,
    secret_key: str = SECRET_KEY,
) -> str:
    """Создать JWT access token."""
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, secret_key, algorithm=ALGORITHM)


def verify_token(token: str, secret_key: str = SECRET_KEY) -> dict:
    """
    Верифицировать JWT токен.

    Raises:
        HTTPException 401 при невалидном токене.
    """
    try:
        payload = jwt.decode(token, secret_key, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Невалидный токен",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return payload
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Не удалось верифицировать токен",
            headers={"WWW-Authenticate": "Bearer"},
        )


def authenticate_user(username: str, password: str) -> Optional[str]:
    """
    Проверить учётные данные пользователя.

    Returns:
        username при успехе, None при неудаче.
    """
    expected_password = DEMO_USERS.get(username)
    if expected_password and expected_password == password:
        return username
    return None


# ============================================================= Auth Dependency
security = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> dict:
    """
    FastAPI dependency: извлечь текущего пользователя из Bearer токена.

    Raises:
        HTTPException 401 если токен отсутствует или невалиден.
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Требуется аутентификация",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return verify_token(credentials.credentials)


async def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> Optional[dict]:
    """
    Опциональная аутентификация (для публичных эндпоинтов).
    """
    if credentials is None:
        return None
    try:
        return verify_token(credentials.credentials)
    except HTTPException:
        return None


# ============================================================= Rate Limiter
class SlidingWindowRateLimiter:
    """
    Rate limiter на основе скользящего окна.

    Хранит временные метки запросов для каждого клиента в памяти.
    Для продакшна рекомендуется использовать Redis.
    """

    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        """
        Args:
            max_requests: Максимальное число запросов за окно.
            window_seconds: Размер скользящего окна в секундах.
        """
        self.max_requests = max_requests
        self.window = window_seconds
        self._clients: dict[str, deque[float]] = defaultdict(deque)

    def is_allowed(self, client_id: str) -> tuple[bool, int]:
        """
        Проверить, разрешён ли запрос для данного клиента.

        Args:
            client_id: Идентификатор клиента (IP или user ID).

        Returns:
            (allowed: bool, remaining: int) — разрешён ли запрос и сколько осталось.
        """
        now = time.time()
        window_start = now - self.window
        timestamps = self._clients[client_id]

        # Убрать устаревшие метки
        while timestamps and timestamps[0] < window_start:
            timestamps.popleft()

        if len(timestamps) >= self.max_requests:
            return False, 0

        timestamps.append(now)
        remaining = self.max_requests - len(timestamps)
        return True, remaining


# Глобальный экземпляр rate limiter
_rate_limiter = SlidingWindowRateLimiter(max_requests=100, window_seconds=60)


def get_client_id(request: Request) -> str:
    """Определить идентификатор клиента по IP."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


async def rate_limit_middleware(
    request: Request,
    max_requests: int = 100,
) -> None:
    """
    FastAPI dependency для rate limiting.

    Raises:
        HTTPException 429 при превышении лимита.
    """
    client_id = get_client_id(request)
    allowed, remaining = _rate_limiter.is_allowed(client_id)

    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Превышен лимит запросов. Попробуйте через {_rate_limiter.window} секунд.",
            headers={"Retry-After": str(_rate_limiter.window)},
        )

    # Добавляем заголовок с оставшимися запросами
    request.state.rate_limit_remaining = remaining
