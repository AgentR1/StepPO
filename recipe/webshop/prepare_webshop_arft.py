#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from recipe.webshop.env.data import DEFAULT_SEED, build_goals, load_products_and_attrs


def _row(goal: dict[str, Any], split: str, row_index: int) -> dict[str, Any]:
    prompt = [{"role": "user", "content": goal["instruction"]}]
    goal_options = [
        {"name": name, "value": value}
        for name, value in sorted((goal.get("goal_options") or {}).items())
    ]
    ground_truth = {
        "asin": goal["asin"],
        "instruction": goal["instruction"],
        "attributes": goal.get("instruction_attributes") or goal.get("attributes") or [],
        "goal_options": goal_options,
        "price_upper": goal.get("price_upper"),
        "goal_index": goal["goal_index"],
    }
    return {
        "data_source": f"webshop_small_{split}",
        "prompt": prompt,
        "reward_model": {"ground_truth": ground_truth, "style": "rule"},
        "extra_info": {
            "index": row_index,
            "split": split,
            "goal_index": goal["goal_index"],
            "asin": goal["asin"],
            "instruction": goal["instruction"],
            "category": goal.get("category"),
            "query": goal.get("query"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare WebShop small parquet data for StepPO.")
    parser.add_argument("--input_dir", default="webshop_data")
    parser.add_argument("--output_dir", default="data/webshop")
    parser.add_argument("--train_size", type=int, default=6710)
    parser.add_argument("--test_size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    products, attrs = load_products_and_attrs(args.input_dir)
    goals = build_goals(products, attrs, seed=args.seed)
    requested = args.train_size + args.test_size
    if requested > len(goals):
        raise ValueError(f"Requested {requested} rows but only built {len(goals)} goals.")

    train_goals = goals[: args.train_size]
    test_goals = goals[args.train_size : requested]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.DataFrame([_row(goal, "train", i) for i, goal in enumerate(train_goals)])
    test_df = pd.DataFrame([_row(goal, "test", i) for i, goal in enumerate(test_goals)])
    train_df.to_parquet(out_dir / "train.parquet", index=False)
    test_df.to_parquet(out_dir / "test.parquet", index=False)

    stats = {
        "input_dir": str(Path(args.input_dir).resolve()),
        "output_dir": str(out_dir.resolve()),
        "num_products": len(products),
        "num_goals": len(goals),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "seed": args.seed,
    }
    with (out_dir / "stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"Built {len(goals)} WebShop small goals")
    print(f"Wrote train -> {out_dir / 'train.parquet'} ({len(train_df)} rows)")
    print(f"Wrote test  -> {out_dir / 'test.parquet'} ({len(test_df)} rows)")
    print(f"Wrote stats -> {out_dir / 'stats.json'}")


if __name__ == "__main__":
    main()
