from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from recipe.alfworld.env.alfworld_wrapper import AlfworldTextworldEnv


@dataclass
class AlfworldToolExecutor:
    max_episode_steps: int = 50
    _env: AlfworldTextworldEnv = field(init=False)
    _history_actions: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._env = AlfworldTextworldEnv(max_episode_steps=self.max_episode_steps)

    def reset(self, game_relative_path: str, task_id: str | None = None) -> str:
        self._history_actions.clear()
        return self._env.reset(game_relative_path=game_relative_path, task_id=task_id)

    def step(self, command: str) -> dict[str, Any]:
        self._history_actions.append(command)
        observation, reward, done, info = self._env.step(command)
        return {
            "observation": str(observation),
            "reward": float(reward),
            "done": bool(done),
            "info": info,
            "history_actions": list(self._history_actions),
        }


def format_history_actions(actions: list[str]) -> str:
    if not actions:
        return "None"
    return "\n".join(f"[Action {i + 1}] {action}" for i, action in enumerate(actions))
