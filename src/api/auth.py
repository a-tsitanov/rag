"""X-API-Key header auth.

Keys are loaded from ``settings.api_keys`` (comma-separated in ``.env``).
Apply via ``Depends(require_api_key)`` on any protected route or include
in the ``dependencies=[...]`` of a router.
"""

from fastapi import HTTPException, Security, status
from fastapi.security.api_key import APIKeyHeader

from src.config import settings

_api_key_header = APIKeyHeader(
    name="X-API-Key",
    auto_error=False,
    description="API key issued by the Enterprise KB admin.",
)


async def require_api_key(key: str | None = Security(_api_key_header)) -> str:
    if not key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-API-Key header",
        )
    if key not in settings.api_keys_set:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key",
        )
    return key
