#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from recipe.webshop.prompts import WEBSHOP_TOOL_SCHEMAS
from recipe.webshop.utils import build_invalid_tool_call_observation, build_webshop_messages


TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


def to_jsonable(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


def apply_chat_template(tokenizer, messages: list[dict[str, str]], disable_thinking: bool) -> str:
    kwargs = {
        "tools": WEBSHOP_TOOL_SCHEMAS,
        "add_generation_prompt": True,
        "tokenize": False,
    }
    if disable_thinking:
        try:
            return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
        except TypeError:
            pass
    return tokenizer.apply_chat_template(messages, **kwargs)


def parse_env_step(response_text: str, parser_mode: str) -> tuple[str | None, dict[str, Any]]:
    meta: dict[str, Any] = {
        "parser_mode": parser_mode,
        "tool_call_found": False,
        "tool_name": None,
        "parse_error": None,
    }

    matches = TOOL_CALL_RE.findall(response_text)
    if matches:
        meta["tool_call_found"] = True
        for match in matches:
            try:
                payload = json.loads(match.strip())
                name = payload.get("name")
                arguments = payload.get("arguments", {})
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                meta["tool_name"] = name
                if name == "env_step":
                    command = str(arguments.get("command", "")).strip()
                    return command or None, meta
            except Exception as exc:
                meta["parse_error"] = str(exc)
        return None, meta

    if parser_mode == "strict":
        meta["parse_error"] = "missing <tool_call>...</tool_call>"
        return None, meta

    stripped = response_text.strip()
    try:
        payload = json.loads(stripped)
        name = payload.get("name")
        arguments = payload.get("arguments", {})
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        meta["tool_name"] = name
        if name == "env_step":
            return str(arguments.get("command", "")).strip() or None, meta
    except Exception:
        pass

    first_line = next((line.strip() for line in stripped.splitlines() if line.strip()), "")
    return first_line or None, meta


class LocalGenerator:
    def __init__(
        self,
        model_path: str,
        *,
        dtype: str,
        device_map: str,
        trust_remote_code: bool,
        disable_thinking: bool,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
        torch_dtype: Any = "auto"
        if dtype != "auto":
            torch_dtype = getattr(torch, dtype)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
        )
        self.model.eval()
        self.disable_thinking = disable_thinking

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> tuple[str, str]:
        import torch

        prompt_text = apply_chat_template(self.tokenizer, messages, self.disable_thinking)
        inputs = self.tokenizer(prompt_text, return_tensors="pt")
        device = next(self.model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        do_sample = temperature > 0.0
        generation_kwargs: dict[str, Any] = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            generation_kwargs["temperature"] = temperature
            generation_kwargs["top_p"] = top_p
        with torch.inference_mode():
            output_ids = self.model.generate(**generation_kwargs)
        response_ids = output_ids[0, inputs["input_ids"].shape[-1] :]
        response_text = self.tokenizer.decode(response_ids, skip_special_tokens=False)
        return prompt_text, response_text


class WebShopSyncClient:
    def __init__(self, base_url: str, timeout: float) -> None:
        self.client = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout)

    def health(self) -> dict[str, Any]:
        resp = self.client.get("/health")
        resp.raise_for_status()
        return resp.json()

    def reset(self, goal_index: int) -> dict[str, Any]:
        resp = self.client.post("/reset", json={"goal_index": int(goal_index)})
        resp.raise_for_status()
        return resp.json()

    def step(self, goal_index: int, env_state: dict[str, Any], action: str) -> dict[str, Any]:
        resp = self.client.post(
            "/step",
            json={"goal_index": int(goal_index), "env_state": env_state, "action": action},
        )
        resp.raise_for_status()
        return resp.json()

    def close(self) -> None:
        self.client.close()


def load_rows(data_root: Path, split: str, max_samples: int, sample_mode: str, seed: int) -> list[dict[str, Any]]:
    df = pd.read_parquet(data_root / f"{split}.parquet")
    if max_samples > 0:
        max_samples = min(max_samples, len(df))
        if sample_mode == "head":
            df = df.head(max_samples)
        elif sample_mode == "random":
            df = df.sample(n=max_samples, random_state=seed)
        elif sample_mode == "category_stratified":
            rng = random.Random(seed)
            category_to_indices: dict[str, list[int]] = defaultdict(list)
            for idx, extra_info in df["extra_info"].items():
                category_to_indices[str(extra_info.get("category", ""))].append(idx)
            for indices in category_to_indices.values():
                rng.shuffle(indices)
            selected: list[int] = []
            while len(selected) < max_samples and category_to_indices:
                for category in sorted(list(category_to_indices)):
                    if not category_to_indices[category]:
                        category_to_indices.pop(category)
                        continue
                    selected.append(category_to_indices[category].pop())
                    if len(selected) >= max_samples:
                        break
            df = df.loc[selected]
        else:
            raise ValueError(f"Unknown sample_mode: {sample_mode}")
    rows: list[dict[str, Any]] = []
    for idx, row in df.iterrows():
        item = to_jsonable(row.to_dict())
        item["_row_index"] = int(idx)
        rows.append(item)
    return rows


def run_sample(
    row: dict[str, Any],
    generator: LocalGenerator,
    env_client: WebShopSyncClient,
    *,
    max_steps: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    parser_mode: str,
    save_prompts: bool,
) -> dict[str, Any]:
    extra_info = row["extra_info"]
    instruction = str(extra_info.get("instruction") or row["prompt"][0]["content"]).strip()
    goal_index = int(extra_info["goal_index"])
    reset_payload = env_client.reset(goal_index)
    observation = str(reset_payload["observation"])
    env_state = reset_payload["env_state"]
    available_actions = (reset_payload.get("info") or {}).get("available_actions") or []

    history_actions: list[str] = []
    trajectory: list[dict[str, Any]] = []
    final_reward = 0.0
    success = False
    done = False
    invalid_tool_call_count = 0
    terminated_reason = "max_steps"
    final_info: dict[str, Any] = {}

    for step_idx in range(1, max_steps + 1):
        messages = build_webshop_messages(
            instruction=instruction,
            observation=observation,
            history_actions=history_actions,
            available_actions=available_actions,
        )
        prompt_text, response_text = generator.generate(
            messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        command, parse_meta = parse_env_step(response_text, parser_mode)

        step_record: dict[str, Any] = {
            "step": step_idx,
            "observation_before": observation,
            "env_state_before": env_state,
            "raw_response": response_text,
            "parsed_command": command,
            "parse_meta": parse_meta,
        }
        if save_prompts:
            step_record["prompt_text"] = prompt_text

        if not command:
            invalid_tool_call_count += 1
            reason = parse_meta.get("parse_error") or f"expected env_step tool, got {parse_meta.get('tool_name')!r}"
            observation = build_invalid_tool_call_observation(observation, reason)
            history_actions.append(f"INVALID_TOOL_CALL: {reason}")
            step_record.update(
                {
                    "observation_after": observation,
                    "env_state_after": env_state,
                    "reward": 0.0,
                    "done": False,
                    "info": {"error": reason},
                    "invalid_tool_call": True,
                }
            )
            trajectory.append(step_record)
            continue

        result = env_client.step(goal_index, env_state, command)
        observation = str(result["observation"])
        env_state = result["env_state"]
        reward = float(result["reward"])
        done = bool(result["done"])
        info = result.get("info") or {}
        available_actions = info.get("available_actions", available_actions)
        history_actions.append(command)

        if done:
            final_reward = reward
            success = bool(info.get("success", reward >= 0.999))
            final_info = info
            terminated_reason = "env_done"

        step_record.update(
            {
                "observation_after": observation,
                "env_state_after": env_state,
                "reward": reward,
                "done": done,
                "info": info,
            }
        )
        trajectory.append(step_record)

        if done:
            break

    return {
        "goal_index": goal_index,
        "split": extra_info.get("split"),
        "asin": extra_info.get("asin"),
        "category": extra_info.get("category"),
        "query": extra_info.get("query"),
        "instruction": instruction,
        "success": success,
        "score": final_reward,
        "done": done,
        "num_steps": len(trajectory),
        "invalid_tool_call_count": invalid_tool_call_count,
        "terminated_reason": terminated_reason,
        "history_actions": history_actions,
        "final_observation": observation,
        "final_info": final_info,
        "trajectory": trajectory,
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "num_samples": len(results),
        "success_rate": sum(float(row["success"]) for row in results) / len(results) if results else 0.0,
        "mean_score": sum(float(row["score"]) for row in results) / len(results) if results else 0.0,
        "by_category": {},
        "terminated_reason_counts": {},
    }
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        by_category[str(row.get("category", ""))].append(row)
        reason = str(row.get("terminated_reason"))
        summary["terminated_reason_counts"][reason] = summary["terminated_reason_counts"].get(reason, 0) + 1
    for category, rows in sorted(by_category.items()):
        summary["by_category"][category] = {
            "num_samples": len(rows),
            "success_rate": sum(float(row["success"]) for row in rows) / len(rows),
            "mean_score": sum(float(row["score"]) for row in rows) / len(rows),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Zero-shot WebShop evaluation using the StepPO WebShop recipe.")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_root", default="data/webshop")
    parser.add_argument("--env_base_url", default="http://127.0.0.1:4100")
    parser.add_argument("--output_dir", default="outputs/webshop_zeroshot")
    parser.add_argument("--splits", nargs="+", default=["test"])
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--sample_mode", default="head", choices=["head", "random", "category_stratified"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=15)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--parser_mode", default="strict", choices=["strict", "permissive"])
    parser.add_argument("--trust_remote_code", action="store_true", default=True)
    parser.add_argument("--enable_thinking", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--disable_thinking", action="store_true")
    parser.add_argument("--save_prompts", action="store_true")
    parser.add_argument("--print_steps", action="store_true", help="Print raw responses/actions/observations.")
    parser.add_argument("--print_observation_chars", type=int, default=500)
    parser.add_argument("--env_timeout", type=float, default=30.0)
    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    env_client = WebShopSyncClient(args.env_base_url, timeout=args.env_timeout)
    health = env_client.health()
    print(f"WebShop env health: {health}")

    generator = LocalGenerator(
        args.model_path,
        dtype=args.dtype,
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
        disable_thinking=args.disable_thinking and not args.enable_thinking,
    )

    all_results: list[dict[str, Any]] = []
    split_summaries: dict[str, Any] = {}
    try:
        for split in args.splits:
            rows = load_rows(data_root, split, args.max_samples, args.sample_mode, args.seed)
            split_output = output_dir / f"{split}.jsonl"
            split_results: list[dict[str, Any]] = []
            with split_output.open("w", encoding="utf-8") as f:
                for idx, row in enumerate(rows):
                    result = run_sample(
                        row,
                        generator,
                        env_client,
                        max_steps=args.max_steps,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        parser_mode=args.parser_mode,
                        save_prompts=args.save_prompts,
                    )
                    result["row_index"] = row.get("_row_index", idx)
                    split_results.append(result)
                    all_results.append(result)
                    f.write(json.dumps(to_jsonable(result), ensure_ascii=False) + "\n")
                    print(
                        f"[{split}] {idx + 1}/{len(rows)} "
                        f"score={result['score']:.3f} success={result['success']} "
                        f"steps={result['num_steps']} reason={result['terminated_reason']} "
                        f"category={result['category']}"
                    )
                    if args.print_steps:
                        for step in result["trajectory"]:
                            print(f"  step={step['step']}")
                            print(f"  response={step.get('raw_response', '').strip()}")
                            print(f"  parsed_command={step.get('parsed_command')}")
                            if step.get("parse_meta", {}).get("parse_error"):
                                print(f"  parse_error={step['parse_meta']['parse_error']}")
                            print(f"  reward={step.get('reward')} done={step.get('done')}")
                            obs = step.get("observation_after") or step.get("observation_before") or ""
                            obs = obs.replace("\n", "\\n")
                            print(f"  observation={obs[: args.print_observation_chars]}")
            split_summaries[split] = summarize(split_results)
    finally:
        env_client.close()

    summary = {
        "model_path": args.model_path,
        "data_root": str(data_root),
        "env_base_url": args.env_base_url,
        "parser_mode": args.parser_mode,
        "sample_mode": args.sample_mode,
        "seed": args.seed,
        "disable_thinking": args.disable_thinking and not args.enable_thinking,
        "max_steps": args.max_steps,
        "max_samples": args.max_samples,
        "splits": split_summaries,
        "merged": summarize(all_results),
    }
    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(summary), f, ensure_ascii=False, indent=2)

    print(json.dumps(to_jsonable(summary), ensure_ascii=False, indent=2))
    print(f"Wrote outputs to {output_dir}")


if __name__ == "__main__":
    main()
