from __future__ import annotations

import json
import logging
import os
from typing import Any
from uuid import uuid4

from transformers import AutoProcessor, AutoTokenizer

from arft.agent_flow.agent_flow import AgentFlowBase, AgentFlowOutput, AgentFlowStep, register
from arft.reward_loop import ARFTRewardLoopWorker as RewardLoopWorker
from recipe.alfworld.prompts import ALFWORLD_SYSTEM_PROMPT, ALFWORLD_TOOL_SCHEMAS, ALFWORLD_USER_PROMPT
from recipe.alfworld.utils import AlfworldToolExecutor, format_history_actions
from verl.experimental.agent_loop.agent_loop import AsyncLLMServerManager, DictConfigWrap
from verl.experimental.agent_loop.tool_parser import ToolParser
from verl.utils.profiler import simple_timer

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("alfworld_agent")
class AlfworldAgentFlow(AgentFlowBase):
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
        self.max_steps = kwargs.get("max_steps", 20)
        self.max_parallel_calls = 1
        self.max_episode_steps = kwargs.get("max_episode_steps", 50)

        self.tool_parser = ToolParser.get_tool_parser(
            self.config.actor_rollout_ref.rollout.multi_turn.format,
            self.tokenizer,
        )
        self.prompt_length = self.config.actor_rollout_ref.rollout.prompt_length
        self.response_length = self.config.actor_rollout_ref.rollout.response_length
        self.tool_schemas = ALFWORLD_TOOL_SCHEMAS

        self.executor = AlfworldToolExecutor(max_episode_steps=self.max_episode_steps)
        self.current_observation: str = ""
        self.history_actions: list[str] = []
        self.steps: list[AgentFlowStep] = []

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentFlowOutput:
        extra_info = kwargs.get("extra_info") or {}
        task_id = extra_info.get("task_id")
        split = extra_info.get("split", "train")
        task_type_raw = extra_info.get("task_type_raw")
        task_family = extra_info.get("task_family")
        game_relative_path = extra_info.get("game_relative_path")
        if not game_relative_path:
            raise ValueError("ALFWorld sample is missing extra_info.game_relative_path")

        self.current_observation = self.executor.reset(game_relative_path=game_relative_path, task_id=task_id)
        self.history_actions = []
        self.steps = []

        metrics: dict[str, Any] = {}
        num_steps = 0
        done = False
        final_success_flag: bool | None = None
        dense_reward_sum = 0.0

        while num_steps < self.max_steps and not done:
            num_steps += 1

            messages = [
                {"role": "system", "content": ALFWORLD_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": ALFWORLD_USER_PROMPT.format(
                        observation=self.current_observation,
                        history_actions=format_history_actions(self.history_actions),
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
                final_step = AgentFlowStep(
                    prompt_ids=prompt_ids,
                    response_ids=response_ids,
                    response_logprobs=output.log_probs[: self.response_length] if output.log_probs else None,
                    reward_score=None,
                    extra_fields={
                        "reward_extra_info": {
                            "success": final_success_flag,
                            "num_steps": num_steps,
                            "dense_reward_sum": dense_reward_sum,
                            "task_id": task_id,
                            "split": split,
                            "task_type_raw": task_type_raw,
                            "task_family": task_family,
                        },
                    },
                )
                final_step = await self._postprocess(final_step, **kwargs)
                self.steps.append(final_step)
                break

            tool_call = tool_calls[0]
            env_reward = 0.0

            if tool_call.name == "env_step":
                try:
                    tool_args = json.loads(tool_call.arguments)
                    command = str(tool_args.get("command", "")).strip()
                except Exception as e:
                    logger.warning("Failed to parse env_step arguments: %s", e)
                    command = ""

                if command:
                    result = self.executor.step(command)
                    self.current_observation = result["observation"]
                    env_reward = float(result["reward"])
                    done = bool(result["done"])
                    info = result.get("info", {}) or {}
                    self.history_actions = result.get("history_actions", self.history_actions)
                    if "success" in info:
                        final_success_flag = bool(info["success"])
                    elif "won" in info:
                        final_success_flag = bool(info["won"])
                    dense_reward_sum += env_reward

            step = AgentFlowStep(
                prompt_ids=prompt_ids,
                response_ids=response_ids,
                response_logprobs=output.log_probs[: self.response_length] if output.log_probs else None,
                reward_score=0.0,
                extra_fields={
                    "reward_extra_info": {
                        "step_env_reward": env_reward,
                        "dense_reward_sum": dense_reward_sum,
                        "success": final_success_flag,
                        "num_steps": num_steps,
                        "task_id": task_id,
                        "split": split,
                        "task_type_raw": task_type_raw,
                        "task_family": task_family,
                    },
                },
            )
            step = await self._postprocess(step, **kwargs)
            self.steps.append(step)

            if done:
                final_step = AgentFlowStep(
                    prompt_ids=prompt_ids,
                    response_ids=response_ids,
                    response_logprobs=output.log_probs[: self.response_length] if output.log_probs else None,
                    reward_score=None,
                    extra_fields={
                        "reward_extra_info": {
                            "success": final_success_flag,
                            "num_steps": num_steps,
                            "dense_reward_sum": dense_reward_sum,
                            "task_id": task_id,
                            "split": split,
                            "task_type_raw": task_type_raw,
                            "task_family": task_family,
                        },
                    },
                )
                final_step = await self._postprocess(final_step, **kwargs)
                self.steps.append(final_step)
                break

        return AgentFlowOutput(steps=self.steps, metrics=metrics)
