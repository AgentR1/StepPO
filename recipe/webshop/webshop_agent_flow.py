from __future__ import annotations

import json
import logging
import os
from typing import Any
from uuid import uuid4

from transformers import AutoProcessor, AutoTokenizer

from arft.agent_flow.agent_flow import AgentFlowBase, AgentFlowOutput, AgentFlowStep, register
from arft.reward_loop import ARFTRewardLoopWorker as RewardLoopWorker
from recipe.webshop.prompts import WEBSHOP_SYSTEM_PROMPT, WEBSHOP_TOOL_SCHEMAS, WEBSHOP_USER_PROMPT
from recipe.webshop.utils import WebShopEnvClient, format_history_actions
from verl.experimental.agent_loop.agent_loop import AsyncLLMServerManager, DictConfigWrap
from verl.experimental.agent_loop.tool_parser import ToolParser
from verl.utils.profiler import simple_timer

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("webshop_agent")
class WebShopAgentFlow(AgentFlowBase):
    def __init__(
        self,
        trainer_config: DictConfigWrap,
        server_manager: AsyncLLMServerManager,
        reward_loop_worker: RewardLoopWorker,
        tokenizer: AutoTokenizer,
        processor: AutoProcessor,
        **kwargs,
    ):
        super().__init__(trainer_config, server_manager, reward_loop_worker, tokenizer, processor, **kwargs)
        self.max_steps = int(kwargs.get("max_steps", 15))
        self.max_parallel_calls = 1
        self.tool_parser = ToolParser.get_tool_parser(
            self.config.actor_rollout_ref.rollout.multi_turn.format,
            self.tokenizer,
        )
        self.response_length = self.config.actor_rollout_ref.rollout.response_length
        self.tool_schemas = WEBSHOP_TOOL_SCHEMAS
        self.client = WebShopEnvClient(timeout=float(kwargs.get("env_timeout", 30.0)))
        self.steps: list[AgentFlowStep] = []

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentFlowOutput:
        extra_info = kwargs.get("extra_info") or {}
        raw_prompt = list(kwargs.get("raw_prompt") or kwargs.get("prompt") or [])
        instruction = str(extra_info.get("instruction") or (raw_prompt[0]["content"] if raw_prompt else "")).strip()
        goal_index = int(extra_info.get("goal_index"))
        split = extra_info.get("split", "train")
        asin = extra_info.get("asin")

        reset_payload = await self.client.reset(goal_index)
        current_observation = str(reset_payload["observation"])
        env_state = reset_payload["env_state"]
        history_actions: list[str] = []
        self.steps = []

        metrics: dict[str, Any] = {}
        done = False
        final_reward = 0.0
        final_info: dict[str, Any] = {}
        num_steps = 0

        while num_steps < self.max_steps and not done:
            num_steps += 1
            messages = [
                {"role": "system", "content": WEBSHOP_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": WEBSHOP_USER_PROMPT.format(
                        instruction=instruction,
                        observation=current_observation,
                        history_actions=format_history_actions(history_actions),
                    ),
                },
            ]
            prompt_ids = await self.apply_chat_template(messages, tools=self.tool_schemas)

            with simple_timer("generate_sequences", metrics):
                output = await self.server_manager.generate(
                    request_id=uuid4().hex,
                    prompt_ids=prompt_ids,
                    sampling_params=sampling_params,
                )

            response_ids = output.token_ids[: self.response_length]
            _, tool_calls = await self.tool_parser.extract_tool_calls(response_ids)

            if not tool_calls:
                step = AgentFlowStep(
                    prompt_ids=prompt_ids,
                    response_ids=response_ids,
                    response_logprobs=output.log_probs[: self.response_length] if output.log_probs else None,
                    reward_score=None,
                    extra_fields={
                        "reward_extra_info": {
                            "final_reward": final_reward,
                            "success": bool(final_info.get("success", False)),
                            "num_steps": num_steps,
                            "goal_index": goal_index,
                            "split": split,
                            "asin": asin,
                            "selected_asin": final_info.get("selected_asin"),
                        }
                    },
                )
                step = await self._postprocess(step, **kwargs)
                self.steps.append(step)
                break

            command = ""
            tool_call = tool_calls[0]
            if tool_call.name == "env_step":
                try:
                    command = str(json.loads(tool_call.arguments).get("command", "")).strip()
                except Exception as exc:
                    logger.warning("Failed to parse env_step arguments: %r", exc)

            env_reward = 0.0
            step_info: dict[str, Any] = {}
            if command:
                try:
                    result = await self.client.step(goal_index, env_state, command)
                    current_observation = str(result["observation"])
                    env_state = result["env_state"]
                    env_reward = float(result["reward"])
                    done = bool(result["done"])
                    step_info = result.get("info") or {}
                    history_actions.append(command)
                    if done:
                        final_reward = env_reward
                        final_info = step_info
                except Exception as exc:
                    logger.warning("WebShop env step failed: %r", exc)
                    step_info = {"error": str(exc)}

            reward_extra_info = {
                "step_env_reward": env_reward,
                "final_reward": final_reward if done else 0.0,
                "success": bool(step_info.get("success", final_info.get("success", False))),
                "num_steps": num_steps,
                "goal_index": goal_index,
                "split": split,
                "asin": asin,
                "selected_asin": step_info.get("selected_asin", final_info.get("selected_asin")),
            }
            step = AgentFlowStep(
                prompt_ids=prompt_ids,
                response_ids=response_ids,
                response_logprobs=output.log_probs[: self.response_length] if output.log_probs else None,
                reward_score=env_reward if done else 0.0,
                extra_fields={"reward_extra_info": reward_extra_info},
            )
            step = await self._postprocess(step, **kwargs)
            self.steps.append(step)

            if done:
                break

        return AgentFlowOutput(steps=self.steps, metrics=metrics)

