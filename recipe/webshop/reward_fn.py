from __future__ import annotations

from typing import Any

from verl.utils.reward_score import default_compute_score


def _metadata_str(value: Any) -> str:
    return "" if value is None else str(value)


def _metadata_int(value: Any) -> int:
    if value is None:
        return -1
    try:
        return int(value)
    except Exception:
        return -1


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict | None = None,
    **kwargs,
) -> float | dict[str, Any]:
    if not str(data_source).startswith("webshop"):
        return default_compute_score(data_source, solution_str, ground_truth, extra_info, **kwargs)

    extra_info = extra_info or {}
    runtime_info = extra_info.get("reward_extra_info", {}) if isinstance(extra_info, dict) else {}
    if not isinstance(runtime_info, dict):
        runtime_info = {}

    score = float(runtime_info.get("final_reward") or runtime_info.get("step_env_reward") or 0.0)
    return {
        "score": score,
        "acc": score,
        "success": bool(runtime_info.get("success", score >= 0.999)),
        "final_reward": score,
        "split": _metadata_str(runtime_info.get("split", extra_info.get("split"))),
        "goal_index": _metadata_int(runtime_info.get("goal_index", extra_info.get("goal_index"))),
        "asin": _metadata_str(runtime_info.get("asin", extra_info.get("asin"))),
        "selected_asin": _metadata_str(runtime_info.get("selected_asin")),
        "num_steps": _metadata_int(runtime_info.get("num_steps")),
    }
