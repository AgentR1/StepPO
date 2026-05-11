import asyncio
import json
import logging
import os
import re
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import httpx
from openai import OpenAI

import env_config  # noqa: F401

# -----------------------------------------------------------------------------
# Prompts mirrored from ``recipe/paper_search/prompts.py`` for standalone runs.
# Inference-only: includes ``{date_range_instruction}`` before ``### Output Format`` (CLI month range).
# If you change training prompts, update this block to match.
# -----------------------------------------------------------------------------
PAPERSEARCH_SYSTEM_PROMPT = "You are a research agent. Your goal is to find papers relevant to the User Query."

PAPERSEARCH_USER_PROMPT = """### User Query
{user_query}

### History Actions
{history_actions}

### Paper List
{paper_list}

### Instructions
Analyze the **Paper List** and **History Actions** to determine the next set of actions. Enclose your analysis of the state and decision logic within `<analysis>...</analysis>` tags.
**You support parallel tool calling.** You should output multiple tool calls in a single step if several independent actions are valuable at the current state.
**Attend to the history actions and avoid repeating the same search query or expanding the same paper.**

{date_range_instruction}
### Output Format
<analysis>
[Your analysis of the current state and decision logic...]
</analysis>
<tool_call>
[Tool call 1]
</tool_call>
<tool_call>
[Tool call 2]
</tool_call>
...
"""

PAPERSEARCH_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search for relevant papers with the hybrid retrieval API.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "A single search query in natural language or keywords. "
                            "Must differ from all history queries."
                        ),
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "expand",
            "description": (
                "Expand from an existing paper by merging its citations and references to surface more related works."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "paper_id": {
                        "type": "string",
                        "description": "The paper identifier of a paper already present in the current paper list.",
                    }
                },
                "required": ["paper_id"],
            },
        },
    },
]


from utils import (
    Paper,
    PaperPool,
    PaperSearchV2Client,
    httpx_request_with_retry,
    parse_year_month_str,
    paper_overlaps_year_month_range,
)


# Same pattern as ``verl.experimental.agent_loop.tool_parser.HermesToolParser.tool_call_regex`` (``regex.DOTALL`` ≡ ``re.DOTALL``).
_HERMES_TOOL_CALL_REGEX = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


def _optional_year_month_cli(raw: Optional[str]) -> tuple[Optional[str], Optional[tuple[int, int]]]:
    if raw is None:
        return None, None
    s = str(raw).strip()
    if not s or s.lower() == "none":
        return None, None
    y, m = parse_year_month_str(s)
    return s, (y, m)


def call_openai_chat(
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    api_key: str,
    api_base: str,
    *,
    tools: Optional[list[dict[str, Any]]] = None,
    tool_choice: Optional[str] = None,
):
    client = OpenAI(api_key=api_key, base_url=api_base)

    kwargs: dict[str, Any] = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.1,
    }
    if tools is not None:
        kwargs["tools"] = tools
    if tool_choice is not None:
        kwargs["tool_choice"] = tool_choice

    return client.chat.completions.create(**kwargs)


class PaperSearchV2Agent:
    def __init__(self, logger: logging.Logger, *args, **kwargs):
        self.logger = logger
        self.thought_log_path = kwargs.get("thought_log_path")
        self.paper_pool = PaperPool()
        self._paper_pool_lock = threading.RLock()
        self.ordered_paper_ids: list[str] = []
        self.history_search_queries: dict[str, int] = {}
        self.history_actions: list[tuple[str, str]] = []
        self.user_query = ""
        self.steps: list[dict[str, Any]] = []

        self.max_steps = kwargs.get("max_steps", 15)
        self.max_parallel_calls = kwargs.get("max_parallel_calls", 5)
        self.reward_top_k = kwargs.get("reward_top_k", 3)
        self.search_cost = kwargs.get("search_cost", 0.0)
        self.expand_cost = kwargs.get("expand_cost", 0.0)
        self.min_candidate_score = kwargs.get("min_candidate_score", 0.0)
        self.target_paper = kwargs.get("target_paper", 500)
        self.max_tolerance = kwargs.get("max_tolerance", 5)
        self.search_top_k = kwargs.get("search_top_k", 10)
        self.search_source = kwargs.get("search_source", os.getenv("PAPER_AGENT_V2_SEARCH_SOURCE", "local_db")).strip()
        if self.search_source not in {"local_db", "google"}:
            raise ValueError("PAPER_AGENT_V2_SEARCH_SOURCE must be either 'local_db' or 'google'")
        _fetch_mult = int(os.getenv("PAPER_AGENT_V2_SEARCH_FETCH_MULT", "4"))
        self.search_api_limit = int(
            kwargs.get(
                "search_api_limit",
                min(100, max(40, self.search_top_k * _fetch_mult)),
            )
        )
        self.search_max_to_score = int(kwargs.get("search_max_to_score", self.search_api_limit))
        self.paper_from_month, self._paper_from_ym = _optional_year_month_cli(
            kwargs.get("paper_from_month", os.getenv("PAPER_AGENT_V2_PAPER_FROM"))
        )
        self.paper_to_month, self._paper_to_ym = _optional_year_month_cli(
            kwargs.get("paper_to_month", os.getenv("PAPER_AGENT_V2_PAPER_TO"))
        )
        self.citations_limit = kwargs.get("citations_limit", 30)
        self.references_limit = kwargs.get("references_limit", -1)
        self.tool_schemas = PAPERSEARCH_TOOL_SCHEMAS

        _ps_api_key = kwargs.get("paper_search_api_key", os.getenv("PAPER_SEARCH_V2_API_KEY"))
        _ps_base = kwargs.get("paper_search_base_url", os.getenv("PAPER_SEARCH_V2_BASE_URL"))
        self.client = PaperSearchV2Client(
            base_url=_ps_base,
            timeout=30.0,
            api_key=_ps_api_key,
        )
        self.selector_client = httpx.AsyncClient(timeout=10.0)

        self.model_name = kwargs.get("model_name", os.getenv("PAPER_AGENT_V2_MODEL_NAME", "Qwen3-4b-instruct"))
        self.api_key = kwargs.get("api_key", os.getenv("PAPER_AGENT_V2_API_KEY", self.model_name))
        self.api_base = kwargs.get("api_base", os.getenv("PAPER_AGENT_V2_API_BASE", "http://localhost:8998/v1"))
        self.selector_url = kwargs.get("selector_url", os.getenv("PAPER_AGENT_V2_SELECTOR_URL", "http://localhost:8993/classify"))
        self.selector_model_name = kwargs.get(
            "selector_model_name", os.getenv("PAPER_AGENT_V2_SELECTOR_MODEL", "selector")
        )

        prompt_path = Path(__file__).resolve().parent / "agent_prompt.json"
        with open(prompt_path, "r", encoding="utf-8") as f:
            self.prompts = json.load(f)

    async def close(self) -> None:
        await self.selector_client.aclose()
        await self.client.close()

    def _format_history_actions(self) -> str:
        if not self.history_actions:
            return "None"

        lines: list[str] = []
        for action, value in self.history_actions:
            if action == "search":
                lines.append(f"[Search] {value}")
            elif action == "expand":
                lines.append(f"[Expand] {value}")
            else:
                raise ValueError(f"Invalid action: {action}")
        return "\n".join(lines)

    def _date_range_instruction(self) -> str:
        if self.paper_from_month is None and self.paper_to_month is None:
            return (
                "**Publication date:** no start/end month filter is applied "
                "(search API and local filtering use the full range returned by the server).\n"
            )
        parts: list[str] = []
        if self.paper_from_month is not None:
            parts.append(f"on or after **{self.paper_from_month}** (inclusive)")
        if self.paper_to_month is not None:
            parts.append(f"on or before **{self.paper_to_month}** (inclusive)")
        joined = " and ".join(parts)
        return (
            f"**Publication date:** retrieved papers should be {joined}, using API month range plus "
            f"arXiv id / year overlap when refining results.\n"
        )

    @staticmethod
    def _clean_user_prompt(user_prompt: str) -> str:
        return re.sub(r"[\u200b\u200c\u200d\uFEFF\u00A0]", " ", user_prompt)

    def _parse_hermes_style_tool_calls(self, text: str) -> tuple[str, list[Any]]:
        """Decode tool calls from assistant text like ``HermesToolParser.extract_tool_calls`` (no tokenizer)."""
        tool_calls: list[Any] = []
        if not text:
            return "", tool_calls
        if "<tool_call>" not in text or "</tool_call>" not in text:
            return text, tool_calls

        for match in _HERMES_TOOL_CALL_REGEX.findall(text):
            try:
                function_call = json.loads(match)
                name, arguments = function_call["name"], function_call["arguments"]
                tool_calls.append(
                    SimpleNamespace(
                        function=SimpleNamespace(
                            name=name,
                            arguments=json.dumps(arguments, ensure_ascii=False),
                        )
                    )
                )
            except Exception as exc:  # noqa: BLE001
                self.logger.info("Failed to decode tool call: %s", exc)

        content = _HERMES_TOOL_CALL_REGEX.sub("", text)
        return content, tool_calls

    def _get_next_turn_message(self, user_query: str):
        system_prompt = PAPERSEARCH_SYSTEM_PROMPT
        user_prompt = PAPERSEARCH_USER_PROMPT.format(
            user_query=user_query,
            paper_list=self.paper_pool.paper_list,
            history_actions=self._format_history_actions(),
            date_range_instruction=self._date_range_instruction(),
        )
        user_prompt = self._clean_user_prompt(user_prompt)

        use_native_tools = os.getenv("PAPER_AGENT_V2_NATIVE_TOOLS", "").strip().lower() in ("1", "true", "yes")

        try:
            if use_native_tools:
                response = call_openai_chat(
                    model_name=self.model_name,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    api_key=self.api_key,
                    api_base=self.api_base,
                    tools=self.tool_schemas,
                    tool_choice="required",
                )
            else:
                # Align with RL training: plain completion + Hermes <tool_call> blocks (see verl HermesToolParser).
                response = call_openai_chat(
                    model_name=self.model_name,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    api_key=self.api_key,
                    api_base=self.api_base,
                    tools=None,
                    tool_choice=None,
                )
        except Exception as exc:
            self.logger.info("Failed to get next turn message: %s", exc)
            self.logger.info(user_prompt)
            return None, None

        msg = response.choices[0].message

        if use_native_tools:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                return msg, msg.tool_calls
            return msg, None

        raw_content = getattr(msg, "content", None)
        if isinstance(raw_content, list):
            texts: list[str] = []
            for part in raw_content:
                if isinstance(part, dict) and part.get("type") == "text":
                    texts.append(str(part.get("text", "")))
            text = "\n".join(texts)
        elif isinstance(raw_content, str):
            text = raw_content
        else:
            text = ""

        thought_text, parsed = self._parse_hermes_style_tool_calls(text)
        display_content = thought_text if thought_text else text
        faux = SimpleNamespace(content=display_content)
        return faux, parsed

    def _ordered_ranked_entries(self):
        return list(reversed(self.paper_pool.ranked_papers))

    def _build_save_items(self) -> dict[str, Any]:
        ranked_entries = self._ordered_ranked_entries()
        save_items: dict[str, Any] = {
            "ordered_ids": list(self.ordered_paper_ids),
            "sorted_ids": [entry.paper.paper_id for entry in ranked_entries],
            "details": {},
        }

        for entry in ranked_entries:
            paper = entry.paper
            save_items["details"][paper.paper_id] = {
                "paper_id": paper.paper_id,
                "raw_paper_id": paper.raw_paper_id,
                "arxiv_id": paper.arxiv_id,
                "title": paper.title,
                "abstract": paper.abstract,
                "authors": paper.authors,
                "year": paper.year,
                "score": entry.score,
                "source": entry.source,
                "origin": entry.origin,
                "expand": entry.expand,
            }
        return save_items

    @staticmethod
    def _normalize_tool_argument(value: Any) -> str:
        return str(value).strip() if value is not None else ""

    @staticmethod
    def _normalize_thought_text(content: Optional[str]) -> str:
        if not content:
            return "None"
        return content.strip() or "None"

    def _summarize_tool_calls(self, tool_calls: Optional[list]) -> list[dict[str, Any]]:
        summaries: list[dict[str, Any]] = []
        for tool_call in tool_calls or []:
            raw_arguments = getattr(tool_call.function, "arguments", "")
            try:
                arguments = json.loads(raw_arguments) if raw_arguments else {}
            except Exception:
                arguments = {"raw_arguments": raw_arguments}
            summaries.append(
                {
                    "name": tool_call.function.name,
                    "arguments": arguments,
                }
            )
        return summaries

    def _write_thought_log(
        self,
        step_idx: int,
        thought_text,
        tool_call_summaries: list[dict[str, Any]],
    ) -> None:
        if not self.thought_log_path:
            return

        # print(thought_text)
        # exit(0)

        lines = [
            f"==================== Step {step_idx + 1} ====================",
            "[Assistant Reply]",
            thought_text,
            "",
            "[Tool Calls]",
        ]
        if tool_call_summaries:
            for idx, tool_call in enumerate(tool_call_summaries, start=1):
                lines.append(f"{idx}. {tool_call['name']}: {json.dumps(tool_call['arguments'], ensure_ascii=False)}")
        else:
            lines.append("None")
        lines.extend(["", ""])

        with open(self.thought_log_path, "a", encoding="utf-8") as f:
            f.write("\n".join(lines))

    async def _execute_tool_calls(self, tool_calls: list, **kwargs) -> tuple[float, list[dict[str, Any]]]:
        tasks = []
        summaries: list[dict[str, Any]] = []
        seen_search_queries: set[str] = set()
        seen_expand_ids: set[str] = set()

        for tool_call in tool_calls:
            try:
                tool_args = json.loads(tool_call.function.arguments)
            except Exception as exc:
                self.logger.info("Failed to parse tool arguments: %s", exc)
                continue

            if tool_call.function.name == "search":
                query = self._normalize_tool_argument(tool_args.get("query"))
                if query:
                    if query in seen_search_queries:
                        self.logger.info("Skip duplicated search tool call in the same step: %s", query)
                        continue
                    seen_search_queries.add(query)
                    tasks.append(self.search(query, **kwargs))
                    summaries.append({"name": "search", "arguments": {"query": query}})
                    self.history_actions.append(("search", query))
            elif tool_call.function.name == "expand":
                paper_id = self._normalize_tool_argument(tool_args.get("paper_id"))
                if paper_id:
                    if paper_id in seen_expand_ids:
                        self.logger.info("Skip duplicated expand tool call in the same step: %s", paper_id)
                        continue
                    seen_expand_ids.add(paper_id)
                    tasks.append(self.expand(paper_id, **kwargs))
                    summaries.append({"name": "expand", "arguments": {"paper_id": paper_id}})
                    self.history_actions.append(("expand", paper_id))

        if not tasks:
            return 0.0, summaries

        scores = await asyncio.gather(*tasks)
        return float(sum(scores)), summaries

    async def run(self, user_query: str, save_path: str, **kwargs):
        self.user_query = user_query

        if os.path.exists(save_path):
            self.logger.info("Save path %s already exists, skipping...", save_path)
            return

        step_idx = 0
        tolerance = 0
        last_paper_count = self.paper_pool._len_relevant_papers()
        break_flag = False

        while step_idx < self.max_steps:
            self.logger.info("\n\n-------- Action %d --------", step_idx + 1)
            self.logger.info("History Actions:\n%s", self._format_history_actions())

            paper_list_before = self.paper_pool.paper_list
            msg, tool_calls = self._get_next_turn_message(user_query)

            if msg is None and tool_calls is None:
                break

            tool_calls = (tool_calls or [])[: self.max_parallel_calls]
            # thought_analysis = self._normalize_thought_text(getattr(msg, "content", None))
            thought_analysis = msg.content
            thought_tool_calls = self._summarize_tool_calls(tool_calls)
            self._write_thought_log(step_idx, thought_analysis, thought_tool_calls)

            if not tool_calls:
                self.logger.info("No tool calls.")
                tolerance += 1
                if tolerance >= self.max_tolerance:
                    self.logger.info("Stopping: consecutive no-new-paper limit %d reached.", self.max_tolerance)
                    break_flag = True
                step_idx += 1
                continue

            self.logger.info("Number of tool calls: %d", len(tool_calls))

            tool_reward_score, tool_call_summaries = await self._execute_tool_calls(tool_calls, **kwargs)
            paper_list_after = self.paper_pool.paper_list

            result = {
                "step_idx": step_idx,
                "tool_calls": tool_call_summaries,
                "paper_list_before": paper_list_before,
                "paper_list_after": paper_list_after,
                "reward_score": tool_reward_score,
            }
            self.steps.append(result)

            new_paper_count = self.paper_pool._len_relevant_papers()
            self.logger.info(
                "Step %d: New papers added: %d, Total papers: %d, Tolerance: %d",
                step_idx + 1,
                new_paper_count - last_paper_count,
                new_paper_count,
                tolerance,
            )

            if new_paper_count > last_paper_count:
                tolerance = 0
            else:
                tolerance += 1
            last_paper_count = new_paper_count

            if new_paper_count >= self.target_paper:
                self.logger.info("Stopping: collected target_paper (%d) papers.", self.target_paper)
                break_flag = True
            if tolerance >= self.max_tolerance:
                self.logger.info("Stopping: consecutive no-new-paper limit %d reached.", self.max_tolerance)
                break_flag = True

            if break_flag:
                self.logger.info("================== User Prompt ==================")
                self.logger.info(
                    PAPERSEARCH_USER_PROMPT.format(
                        user_query=user_query,
                        paper_list=self.paper_pool.paper_list,
                        history_actions=self._format_history_actions(),
                        date_range_instruction=self._date_range_instruction(),
                    )
                )
                if getattr(msg, "content", None):
                    self.logger.info("%s", msg.content)
                break

            step_idx += 1

        save_items = self._build_save_items()
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(save_items, f, ensure_ascii=False, indent=4)

        return save_items["ordered_ids"]

    async def search(self, query: str, **kwargs) -> float:
        self.logger.info("[Search]: %s", query)
        if query in self.history_search_queries:
            return -0.5

        try:
            papers = await self.client.search(
                query=query,
                limit=self.search_api_limit,
                source=self.search_source,
                from_month=self.paper_from_month,
                to_month=self.paper_to_month,
            )
        except Exception as exc:
            self.logger.info("Error in search %s: %r", query, exc)
            self.history_search_queries[query] = 0
            return 0.0

        filtered = [
            p for p in papers if paper_overlaps_year_month_range(p, self._paper_from_ym, self._paper_to_ym)
        ]
        if len(filtered) != len(papers):
            self.logger.info(
                "[Search] month range filter %s..%s: %d -> %d papers (api_limit=%d)",
                self.paper_from_month or "-",
                self.paper_to_month or "-",
                len(papers),
                len(filtered),
                self.search_api_limit,
            )

        scored_slice = filtered[: self.search_max_to_score]
        if len(scored_slice) < len(filtered):
            self.logger.info(
                "[Search] scoring first %d of %d after month filtering (search_max_to_score)",
                len(scored_slice),
                len(filtered),
            )

        new_papers: list[Paper] = []
        tasks = []
        seen_paper_ids: set[str] = set()

        for paper in scored_slice:
            if not paper.paper_id or paper.paper_id in seen_paper_ids:
                continue
            seen_paper_ids.add(paper.paper_id)
            with self._paper_pool_lock:
                if self.paper_pool.has_paper(paper.paper_id):
                    continue

            new_papers.append(paper)
            tasks.append(self.get_relevance_score(self.user_query, paper, **kwargs))

        relevance_scores = await asyncio.gather(*tasks) if tasks else []

        kept_scores: list[float] = []
        for paper, score in zip(new_papers, relevance_scores):
            if score < self.min_candidate_score:
                continue
            with self._paper_pool_lock:
                if self.paper_pool.has_paper(paper.paper_id):
                    continue
                self.paper_pool.add_paper(paper, "search", query, score)
                self.ordered_paper_ids.append(paper.paper_id)
            kept_scores.append(score)
            self.logger.info("[%.3f] %s", score, paper.title)

        self.history_search_queries[query] = len(kept_scores)
        if not kept_scores:
            return 0.0

        return sum(sorted(kept_scores, reverse=True)[: self.reward_top_k]) - self.search_cost

    async def expand(self, paper_id: str, **kwargs) -> float:
        self.logger.info("[Expand]: %s", paper_id)
        with self._paper_pool_lock:
            paper_pool_entry = self.paper_pool.get_paper(paper_id)
            if not paper_pool_entry:
                return -0.5
            if paper_pool_entry.expand:
                return -0.5
            paper_pool_entry.expand = True

        try:
            citations, references = await asyncio.gather(
                self.client.get_citations(paper_id, limit=self.citations_limit),
                self.client.get_references(paper_id, limit=self.references_limit),
            )
        except Exception as exc:
            self.logger.info("Error in expand %s: %r", paper_id, exc)
            return 0.0

        merged_candidates: list[Paper] = []
        seen_paper_ids: set[str] = set()
        for paper in citations + references:
            if not paper.paper_id or paper.paper_id == paper_id or paper.paper_id in seen_paper_ids:
                continue
            seen_paper_ids.add(paper.paper_id)
            with self._paper_pool_lock:
                if self.paper_pool.has_paper(paper.paper_id):
                    continue
            merged_candidates.append(paper)

        async def _hydrate_candidate(candidate: Paper) -> Optional[Paper]:
            if candidate.abstract:
                return candidate

            try:
                detail = await self.client.get_paper(candidate.paper_id)
            except Exception as exc:
                self.logger.info("Error fetching detail for %s: %r", candidate.paper_id, exc)
                detail = None

            paper = detail or candidate
            if not paper.abstract:
                return None
            return paper

        missing_detail_count = sum(1 for paper in merged_candidates if not paper.abstract)
        self.logger.info(
            "Expand %s: merged_candidates=%d, missing_detail_candidates=%d",
            paper_id,
            len(merged_candidates),
            missing_detail_count,
        )

        hydrated_candidates = await asyncio.gather(*(_hydrate_candidate(paper) for paper in merged_candidates))
        filtered_candidates = [
            paper
            for paper in hydrated_candidates
            if paper is not None and paper_overlaps_year_month_range(paper, self._paper_from_ym, self._paper_to_ym)
        ]
        if len(filtered_candidates) != sum(1 for paper in hydrated_candidates if paper is not None):
            self.logger.info(
                "[Expand] month range filter %s..%s: %d -> %d candidates",
                self.paper_from_month or "-",
                self.paper_to_month or "-",
                sum(1 for paper in hydrated_candidates if paper is not None),
                len(filtered_candidates),
            )

        new_papers: list[Paper] = []
        tasks = []
        for paper in filtered_candidates:
            with self._paper_pool_lock:
                if self.paper_pool.has_paper(paper.paper_id):
                    continue
            new_papers.append(paper)
            tasks.append(self.get_relevance_score(self.user_query, paper, **kwargs))

        kept_scores: list[float] = []
        for paper, score in zip(new_papers, await asyncio.gather(*tasks) if tasks else []):
            if score < self.min_candidate_score:
                continue
            with self._paper_pool_lock:
                if self.paper_pool.has_paper(paper.paper_id):
                    continue
                origin = f"[{paper_pool_entry.paper.paper_id}] {paper_pool_entry.paper.title}"
                self.paper_pool.add_paper(paper, "expand", origin, score)
                self.ordered_paper_ids.append(paper.paper_id)
            kept_scores.append(score)
            self.logger.info("[%.3f] %s", score, paper.title)

        if not kept_scores:
            return 0.0

        return sum(sorted(kept_scores, reverse=True)[: self.reward_top_k]) - self.expand_cost

    async def get_relevance_score(self, query: str, paper: Paper, **kwargs) -> float:
        prompt = self.prompts["get_selected"].format(title=paper.title, abstract=paper.abstract, user_query=query)
        payload = {
            "model": self.selector_model_name,
            "input": [prompt],
        }
        resp = await httpx_request_with_retry(
            self.selector_client,
            "POST",
            self.selector_url,
            json=payload,
            max_retries=3,
        )
        resp.raise_for_status()
        response = resp.json()
        predictions = response["data"]
        scores = [pred["probs"][0] for pred in predictions]
        return float(scores[0])


PaperSearchAgent = PaperSearchV2Agent
