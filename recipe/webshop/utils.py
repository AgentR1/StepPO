from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import httpx

from recipe.webshop.prompts import WEBSHOP_SYSTEM_PROMPT, WEBSHOP_USER_PROMPT


def format_history_actions(actions: list[str]) -> str:
    if not actions:
        return "None"
    return "\n".join(f"[Action {i + 1}] {action}" for i, action in enumerate(actions))


def format_available_actions(actions: list[str] | None) -> str:
    if not isinstance(actions, list) or not actions:
        return "None"
    return "\n".join(f"- {action}" for action in actions)


def build_webshop_messages(
    *,
    instruction: str,
    observation: str,
    history_actions: list[str],
    available_actions: list[str] | None,
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": WEBSHOP_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": WEBSHOP_USER_PROMPT.format(
                instruction=instruction,
                observation=observation,
                history_actions=format_history_actions(history_actions),
                available_actions=format_available_actions(available_actions),
            ),
        },
    ]


def build_invalid_tool_call_observation(previous_observation: str, reason: str) -> str:
    return (
        "Invalid tool call. You must call the `env_step` tool exactly once with JSON arguments "
        'like {"command": "search[wireless headphones]"} or {"command": "click[Buy Now]"}. '
        f"Reason: {reason}\n\n"
        "The environment state did not change. Current Observation:\n"
        f"{previous_observation}"
    )


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
