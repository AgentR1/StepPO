from __future__ import annotations

import os
from functools import lru_cache

from fastapi import FastAPI, HTTPException

from recipe.webshop.env.data import load_product_index
from recipe.webshop.env.engine import WebShopEngine
from recipe.webshop.env.schemas import ResetRequest, ResetResponse, StepRequest, StepResponse


@lru_cache(maxsize=1)
def get_engine() -> WebShopEngine:
    data_dir = os.getenv("WEBSHOP_DATA_DIR", "webshop_data")
    index_dir = os.getenv("WEBSHOP_INDEX_DIR", "data/webshop/index")
    search_top_k = int(os.getenv("WEBSHOP_SEARCH_TOP_K", "10"))
    return WebShopEngine(load_product_index(data_dir=data_dir, index_dir=index_dir), search_top_k=search_top_k)


app = FastAPI(title="WebShop Small Environment", version="0.1.0")


@app.get("/health")
def health() -> dict:
    engine = get_engine()
    return {
        "status": "ok",
        "pid": os.getpid(),
        "num_products": len(engine.index.products),
        "num_goals": len(engine.index.goals),
        "search_top_k": engine.search_top_k,
    }


@app.post("/reset", response_model=ResetResponse)
def reset(req: ResetRequest) -> ResetResponse:
    try:
        observation, state, info = get_engine().reset(req.goal_index)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ResetResponse(observation=observation, env_state=state, info=info)


@app.post("/step", response_model=StepResponse)
def step(req: StepRequest) -> StepResponse:
    try:
        return get_engine().step(req.goal_index, req.env_state, req.action)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

