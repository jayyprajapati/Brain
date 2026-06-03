"""Bearer API-key auth. A single shared secret guards every route except /health."""
import secrets

from fastapi import Header, HTTPException, status

from .config import settings


async def require_api_key(authorization: str = Header(default="")) -> None:
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or malformed Authorization header",
        )
    if not secrets.compare_digest(token, settings.brain_api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )
