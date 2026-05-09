import asyncio
import random
from typing import Any, Iterable, Optional, Tuple, Type

import httpx


_DEFAULT_RETRY_STATUS_CODES = {429, 500, 502, 503, 504}


def _compute_backoff_seconds(
    attempt: int,
    *,
    initial_backoff: float,
    max_backoff: float,
    jitter: float,
) -> float:
    """
    Exponential backoff with jitter.
    attempt=0 means first retry wait, attempt=1 means second retry wait, ...
    """
    base = min(max_backoff, initial_backoff * (2**attempt))
    if jitter <= 0:
        return base
    lo = max(0.0, 1.0 - jitter)
    hi = 1.0 + jitter
    return base * random.uniform(lo, hi)


def _parse_retry_after_seconds(headers: httpx.Headers) -> Optional[float]:
    # Retry-After can be seconds or an HTTP date; we only support seconds here.
    value = headers.get("Retry-After")
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


async def httpx_request_with_retry(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    *,
    semaphore: Optional[asyncio.Semaphore] = None,
    max_retries: int = 3,
    retry_status_codes: Iterable[int] = _DEFAULT_RETRY_STATUS_CODES,
    retry_exceptions: Tuple[Type[BaseException], ...] = (httpx.RequestError, httpx.TimeoutException),
    initial_backoff: float = 0.5,
    max_backoff: float = 8.0,
    jitter: float = 0.2,
    **kwargs: Any,
) -> httpx.Response:
    """
    Wrap httpx.AsyncClient.request with retries.

    Fix: Always release the connection on success by reading the full body and closing the response.
    This prevents connection pool exhaustion caused by unclosed responses.

    Notes:
    - Retries on network/timeout exceptions, and on retry_status_codes (e.g. 429/5xx).
    - max_retries means "extra tries" (total attempts = max_retries + 1).
    - If server returns Retry-After (seconds), it takes precedence.
    - If caller passes stream=True, we will NOT auto-read/close; caller must manage the response.
    """
    retry_status_set = set(retry_status_codes)
    last_exc: Optional[BaseException] = None
    stream = bool(kwargs.get("stream", False))

    for attempt in range(max_retries + 1):
        try:
            if semaphore is not None:
                await semaphore.acquire()
            try:
                resp = await client.request(method, url, **kwargs)
            finally:
                if semaphore is not None:
                    semaphore.release()

            if resp.status_code in retry_status_set and attempt < max_retries:
                try:
                    await resp.aread()
                finally:
                    await resp.aclose()

                retry_after = _parse_retry_after_seconds(resp.headers)
                delay = retry_after if retry_after is not None else _compute_backoff_seconds(
                    attempt,
                    initial_backoff=initial_backoff,
                    max_backoff=max_backoff,
                    jitter=jitter,
                )
                await asyncio.sleep(delay)
                continue

            if stream:
                return resp

            try:
                await resp.aread()
            finally:
                await resp.aclose()
            return resp
        except retry_exceptions as e:
            last_exc = e
            if attempt >= max_retries:
                raise
            delay = _compute_backoff_seconds(
                attempt,
                initial_backoff=initial_backoff,
                max_backoff=max_backoff,
                jitter=jitter,
            )
            await asyncio.sleep(delay)

    assert last_exc is not None
    raise last_exc
