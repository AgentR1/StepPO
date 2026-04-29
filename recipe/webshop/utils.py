from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import httpx


def format_history_actions(actions: list[str]) -> str:
    if not actions:
        return "None"
    return "\n".join(f"[Action {i + 1}] {action}" for i, action in enumerate(actions))


@dataclass
class WebShopEnvClient:
    base_url: str | None = None
    timeout: float = 30.0

    def __post_init__(self) -> None:
        self.base_url = (self.base_url or os.getenv("WEBSHOP_ENV_BASE_URL") or "http://127.0.0.1:4100").rstrip("/")
        self.client = httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout)

    async def reset(self, goal_index: int) -> dict[str, Any]:
        resp = await self.client.post("/reset", json={"goal_index": int(goal_index)})
        resp.raise_for_status()
        return resp.json()

    async def step(self, goal_index: int, env_state: dict[str, Any], action: str) -> dict[str, Any]:
        resp = await self.client.post(
            "/step",
            json={"goal_index": int(goal_index), "env_state": env_state, "action": action},
        )
        resp.raise_for_status()
        return resp.json()

    async def close(self) -> None:
        await self.client.aclose()

