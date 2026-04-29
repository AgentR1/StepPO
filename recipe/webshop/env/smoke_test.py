#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys

import httpx


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test the WebShop environment service.")
    parser.add_argument("--base_url", default="http://127.0.0.1:4100")
    parser.add_argument("--goal_index", type=int, default=0)
    args = parser.parse_args()

    with httpx.Client(base_url=args.base_url, timeout=30.0) as client:
        health = client.get("/health")
        health.raise_for_status()
        print("health:", health.json())

        reset = client.post("/reset", json={"goal_index": args.goal_index})
        reset.raise_for_status()
        payload = reset.json()
        print("reset observation:", payload["observation"][:500].replace("\n", " "))

        instruction = payload["info"].get("instruction") or "product"
        state = payload["env_state"]
        search = client.post(
            "/step",
            json={"goal_index": args.goal_index, "env_state": state, "action": f"search[{instruction}]"},
        )
        search.raise_for_status()
        payload = search.json()
        print("search observation:", payload["observation"][:500].replace("\n", " "))

        asin = payload["info"].get("asin") or None
        target_asin = client.post("/reset", json={"goal_index": args.goal_index}).json()["info"]["asin"]
        item = client.post(
            "/step",
            json={"goal_index": args.goal_index, "env_state": payload["env_state"], "action": f"click[{target_asin}]"},
        )
        item.raise_for_status()
        payload = item.json()
        print("item observation:", payload["observation"][:500].replace("\n", " "))

        buy = client.post(
            "/step",
            json={"goal_index": args.goal_index, "env_state": payload["env_state"], "action": "click[Buy Now]"},
        )
        buy.raise_for_status()
        payload = buy.json()
        print("buy:", {"reward": payload["reward"], "done": payload["done"], "info": payload["info"]})
        if not payload["done"]:
            sys.exit("Buy Now did not finish the episode.")


if __name__ == "__main__":
    main()
