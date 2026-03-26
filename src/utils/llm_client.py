import httpx
from openai import AsyncOpenAI

from src.utils.config import settings

llm_client = AsyncOpenAI(
    api_key=settings.openai_api_key,
    http_client=httpx.AsyncClient(
        limits=httpx.Limits(
            max_connections=20,
            max_keepalive_connections=10,
        ),
        timeout=httpx.Timeout(30.0, connect=5.0),
    ),
)
