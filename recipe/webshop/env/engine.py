from __future__ import annotations

import re
from typing import Any

from recipe.webshop.env.data import ProductIndex, normalize_options, normalize_text, product_attributes_text, product_price
from recipe.webshop.env.schemas import EnvState, StepResponse


ACTION_RE = re.compile(r"^\s*(search|click)\[(.*)\]\s*$", re.IGNORECASE | re.DOTALL)


def parse_action(action: str) -> tuple[str, str] | None:
    match = ACTION_RE.match(action or "")
    if not match:
        return None
    return match.group(1).lower(), normalize_text(match.group(2))


def _short(text: Any, limit: int = 220) -> str:
    value = " ".join(str(text or "").split())
    return value if len(value) <= limit else value[: limit - 3] + "..."


def render_home(goal: dict[str, Any]) -> str:
    return (
        "Amazon Shopping Game\n"
        f"Instruction: {goal['instruction']}\n"
        "You may search for products with search[query]."
    )


def render_search_results(goal: dict[str, Any], query: str, results: list[dict[str, Any]], page_num: int = 0) -> str:
    lines = [
        "Search Results",
        f"Instruction: {goal['instruction']}",
        f"Query: {query}",
        "Clickable products:",
    ]
    if not results:
        lines.append("No results found.")
        return "\n".join(lines)
    for i, result in enumerate(results, start=1):
        item = result["item"]
        lines.append(
            f"[{i}] {item.get('asin')} | {_short(item.get('name'), 180)} | "
            f"{item.get('pricing') or '$100.00'} | {item.get('query') or item.get('category')}"
        )
    lines.append("Click a product ASIN, or search again.")
    return "\n".join(lines)


def render_item(goal: dict[str, Any], item: dict[str, Any], selected_options: dict[str, str]) -> str:
    options = normalize_options(item.get("customization_options"))
    lines = [
        "Product Page",
        f"Instruction: {goal['instruction']}",
        f"ASIN: {item.get('asin')}",
        f"Title: {_short(item.get('name'), 260)}",
        f"Price: {item.get('pricing') or '$100.00'}",
        f"Category: {item.get('product_category') or item.get('category')}",
    ]
    small_description = item.get("small_description")
    if isinstance(small_description, list) and small_description:
        lines.append("Summary:")
        for bullet in small_description[:3]:
            lines.append(f"- {_short(bullet, 220)}")
    elif small_description:
        lines.append(f"Summary: {_short(small_description, 260)}")

    if options:
        lines.append("Options:")
        for option_name, choices in options.items():
            current = selected_options.get(option_name, "not selected")
            values = ", ".join(choice["value"] for choice in choices[:30])
            if len(choices) > 30:
                values += f", ... ({len(choices) - 30} more)"
            lines.append(f"- {option_name} (selected: {current}): {values}")
    else:
        lines.append("Options: none")

    lines.append("Clickable controls: Description, Features, Reviews, Buy Now, Back to Search")
    return "\n".join(lines)


def render_subpage(goal: dict[str, Any], item: dict[str, Any], subpage: str) -> str:
    subpage_norm = normalize_text(subpage)
    lines = [
        subpage.title(),
        f"Instruction: {goal['instruction']}",
        f"ASIN: {item.get('asin')}",
        f"Title: {_short(item.get('name'), 260)}",
    ]
    if subpage_norm == "description":
        lines.append(_short(item.get("full_description") or item.get("small_description") or "No description.", 2000))
    elif subpage_norm == "features":
        small_description = item.get("small_description")
        if isinstance(small_description, list):
            lines.extend(f"- {_short(x, 260)}" for x in small_description[:10])
        else:
            lines.append(_short(small_description or "No features.", 2000))
    elif subpage_norm == "reviews":
        lines.append(f"Average rating: {item.get('average_rating') or 'N.A.'}")
        lines.append(f"Total reviews: {item.get('total_reviews') or 'N.A.'}")
    lines.append("Clickable controls: Back to Item, Back to Search, Buy Now")
    return "\n".join(lines)


def _option_lookup(item: dict[str, Any]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for option_name, choices in normalize_options(item.get("customization_options")).items():
        for choice in choices:
            lookup[choice["value"]] = option_name
    return lookup


def compute_reward(index: ProductIndex, goal: dict[str, Any], state: EnvState) -> tuple[float, dict[str, Any]]:
    item = index.asin_to_product.get(state.asin or "")
    if not item:
        return 0.0, {"success": False, "reason": "no_product_selected"}

    target_asin = str(goal.get("asin") or "")
    selected_asin = str(item.get("asin") or "")
    product_match = 1.0 if selected_asin == target_asin else 0.0

    attr_record = index.attrs.get(selected_asin) or {}
    searchable_text = product_attributes_text(item, attr_record)
    target_attrs = [normalize_text(x) for x in goal.get("instruction_attributes") or goal.get("attributes") or []]
    attr_hits = [attr for attr in target_attrs if attr and attr in searchable_text]
    attr_score = (len(attr_hits) / len(target_attrs)) if target_attrs else 1.0

    goal_options = {normalize_text(k): normalize_text(v) for k, v in (goal.get("goal_options") or {}).items()}
    selected_options = {normalize_text(k): normalize_text(v) for k, v in state.selected_options.items()}
    option_hits = {
        name: selected_options.get(name) == value
        for name, value in goal_options.items()
    }
    option_score = (sum(option_hits.values()) / len(goal_options)) if goal_options else 1.0

    price = product_price(item)
    price_upper = float(goal.get("price_upper") or 1e9)
    price_score = 1.0 if price <= price_upper else 0.0

    score = 0.35 * product_match + 0.30 * attr_score + 0.25 * option_score + 0.10 * price_score
    if selected_asin != target_asin:
        score *= 0.5
    score = max(0.0, min(1.0, score))
    return score, {
        "success": score >= 0.999,
        "target_asin": target_asin,
        "selected_asin": selected_asin,
        "product_match": product_match,
        "attr_score": attr_score,
        "option_score": option_score,
        "price_score": price_score,
        "matched_attributes": attr_hits,
        "option_hits": option_hits,
        "selected_options": selected_options,
        "goal_options": goal_options,
    }


class WebShopEngine:
    def __init__(self, index: ProductIndex, *, search_top_k: int = 10):
        self.index = index
        self.search_top_k = search_top_k

    def reset(self, goal_index: int) -> tuple[str, EnvState, dict[str, Any]]:
        goal = self.index.goal(goal_index)
        state = EnvState(page_type="home")
        info = {
            "goal_index": goal_index,
            "asin": goal.get("asin"),
            "instruction": goal.get("instruction"),
        }
        return render_home(goal), state, info

    def step(self, goal_index: int, state: EnvState, action: str) -> StepResponse:
        goal = self.index.goal(goal_index)
        parsed = parse_action(action)
        if parsed is None:
            return StepResponse(
                observation=self._render_current(goal, state),
                env_state=state,
                reward=0.0,
                done=False,
                info={"error": "invalid_action_format", "expected": "search[...] or click[...]"},
            )

        action_name, value = parsed
        new_state = state.model_copy(deep=True)
        new_state.last_action = action

        if action_name == "search":
            new_state.page_type = "search_results"
            new_state.query = value
            new_state.page_num = 0
            new_state.asin = None
            new_state.subpage = None
            observation = self._render_current(goal, new_state)
            return StepResponse(observation=observation, env_state=new_state, reward=0.0, done=False, info={})

        if value == "buy now":
            new_state.page_type = "done"
            reward, reward_info = compute_reward(self.index, goal, new_state)
            reward_info.update({"final_reward": reward, "goal_index": goal_index})
            return StepResponse(
                observation="Episode complete.",
                env_state=new_state,
                reward=reward,
                done=True,
                info=reward_info,
            )

        if value == "back to search":
            new_state.page_type = "search_results" if new_state.query else "home"
            new_state.subpage = None
            observation = self._render_current(goal, new_state)
            return StepResponse(observation=observation, env_state=new_state, reward=0.0, done=False, info={})

        if value == "back to item":
            if new_state.asin:
                new_state.page_type = "item"
                new_state.subpage = None
            observation = self._render_current(goal, new_state)
            return StepResponse(observation=observation, env_state=new_state, reward=0.0, done=False, info={})

        asin = value.upper()
        if asin in self.index.asin_to_product:
            new_state.page_type = "item"
            new_state.asin = asin
            new_state.subpage = None
            new_state.selected_options = {}
            observation = self._render_current(goal, new_state)
            return StepResponse(observation=observation, env_state=new_state, reward=0.0, done=False, info={})

        if value in {"description", "features", "reviews"} and new_state.asin:
            new_state.page_type = "subpage"
            new_state.subpage = value
            observation = self._render_current(goal, new_state)
            return StepResponse(observation=observation, env_state=new_state, reward=0.0, done=False, info={})

        if new_state.asin:
            item = self.index.asin_to_product[new_state.asin]
            option_name = _option_lookup(item).get(value)
            if option_name:
                new_state.selected_options[option_name] = value
                new_state.page_type = "item"
                new_state.subpage = None
                observation = self._render_current(goal, new_state)
                return StepResponse(observation=observation, env_state=new_state, reward=0.0, done=False, info={})

        return StepResponse(
            observation=self._render_current(goal, state),
            env_state=state,
            reward=0.0,
            done=False,
            info={"error": "click_target_not_available", "target": value},
        )

    def _render_current(self, goal: dict[str, Any], state: EnvState) -> str:
        if state.page_type == "home":
            return render_home(goal)
        if state.page_type == "search_results":
            results = self.index.search(state.query, top_k=self.search_top_k)
            return render_search_results(goal, state.query, results, state.page_num)
        if state.page_type == "item" and state.asin in self.index.asin_to_product:
            return render_item(goal, self.index.asin_to_product[state.asin], state.selected_options)
        if state.page_type == "subpage" and state.asin in self.index.asin_to_product:
            return render_subpage(goal, self.index.asin_to_product[state.asin], state.subpage or "")
        return render_home(goal)

