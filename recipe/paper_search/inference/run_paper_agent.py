import argparse
import asyncio
import json
import logging
import os
from typing import Iterable, Optional, TypeVar

try:
    from tqdm.auto import tqdm
except ImportError:

    T = TypeVar("T")

    def tqdm(iterable: Iterable[T], **_kwargs: object) -> Iterable[T]:  # type: ignore[no-redef]
        """Fallback when tqdm is not installed."""
        return iterable


import env_config  # noqa: F401
from paper_agent import PaperSearchAgent
from utils import parse_year_month_str


def get_logger(save_dir: str, idx: int) -> logging.Logger:
    log_save_dir = os.path.join(save_dir, "logs")
    os.makedirs(log_save_dir, exist_ok=True)

    log_path = os.path.join(log_save_dir, f"Falcon_{idx}.log")
    if os.path.exists(log_path):
        os.remove(log_path)

    logger = logging.getLogger(f"Falcon_v2_{idx}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def get_thought_log_path(save_dir: str, idx: int) -> str:
    thought_log_save_dir = os.path.join(save_dir, "th_logs")
    os.makedirs(thought_log_save_dir, exist_ok=True)

    thought_log_path = os.path.join(thought_log_save_dir, f"Falcon_{idx}.log")
    if os.path.exists(thought_log_path):
        os.remove(thought_log_path)
    return thought_log_path


def _cli_year_month(value: str) -> str:
    parse_year_month_str(value)
    return value.strip()


def _normalize_optional_year_month(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    normalized = str(value).strip()
    if normalized.lower() == "none" or not normalized:
        return None
    return _cli_year_month(normalized)


async def run_single_query(
    logger: logging.Logger,
    query: str,
    save_path: str,
    *,
    thought_log_path: Optional[str] = None,
    paper_from_month: Optional[str] = None,
    paper_to_month: Optional[str] = None,
) -> None:
    agent = PaperSearchAgent(
        logger,
        thought_log_path=thought_log_path,
        paper_from_month=paper_from_month,
        paper_to_month=paper_to_month,
    )
    try:
        await agent.run(query, save_path)
    finally:
        await agent.close()


def _count_text_lines(path: str) -> int:
    """Count newline characters in a text file using a buffered binary read.

    Args:
        path: UTF-8 text file path (e.g. JSONL).

    Returns:
        Number of ``\\n`` bytes seen; matches ``enumerate(open text lines)`` for typical JSONL.
    """
    count = 0
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            count += chunk.count(b"\n")
    return count


def load_existing_ids(details_dir: str) -> set[int]:
    existing_ids: set[int] = set()
    if not os.path.exists(details_dir):
        return existing_ids

    for fname in os.listdir(details_dir):
        if not (fname.startswith("Falcon_") and fname.endswith(".json")):
            continue
        try:
            existing_ids.add(int(fname[:-5].split("_")[1]))
        except Exception:
            continue
    return existing_ids


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run paper search agent over a JSONL query file.")
    parser.add_argument(
        "--paper-from",
        dest="paper_from",
        default=None,
        metavar="YYYY-MM",
        help="Inclusive start month for search (YYYY-MM). Omit or use 'none' for no lower bound. "
        "Overrides env PAPER_AGENT_V2_PAPER_FROM when set.",
    )
    parser.add_argument(
        "--paper-to",
        dest="paper_to",
        default=None,
        metavar="YYYY-MM",
        help="Inclusive end month for search (YYYY-MM). Omit or use 'none' for no upper bound. "
        "Overrides env PAPER_AGENT_V2_PAPER_TO when set.",
    )
    args = parser.parse_args()

    paper_from = _normalize_optional_year_month(
        args.paper_from if args.paper_from is not None else os.getenv("PAPER_AGENT_V2_PAPER_FROM")
    )
    paper_to = _normalize_optional_year_month(
        args.paper_to if args.paper_to is not None else os.getenv("PAPER_AGENT_V2_PAPER_TO")
    )
    if paper_from is not None and paper_to is not None:
        if parse_year_month_str(paper_from) > parse_year_month_str(paper_to):
            raise ValueError(f"--paper-from {paper_from} must be earlier than or equal to --paper-to {paper_to}")

    save_dir = os.getenv("PAPER_AGENT_V2_SAVE_DIR", "")
    test_file_path = os.getenv("PAPER_AGENT_V2_DATASET", "")
    retry_rounds = int(os.getenv("PAPER_AGENT_V2_RETRY_ROUNDS", "10"))

    # print(test_file_path)
    # exit(0)

    os.makedirs(os.path.join(save_dir, "details"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "th_logs"), exist_ok=True)

    line_total = _count_text_lines(test_file_path)

    for retry_idx in range(retry_rounds):
        details_dir = os.path.join(save_dir, "details")
        existing_ids = load_existing_ids(details_dir)

        try:
            with open(test_file_path, "r", encoding="utf-8") as f:
                bar = tqdm(
                    enumerate(f),
                    total=line_total,
                    desc=f"Paper agent (round {retry_idx + 1}/{retry_rounds})",
                    unit="line",
                    dynamic_ncols=True,
                )
                for idx, line in bar:
                    if idx in existing_ids:
                        continue

                    line = line.strip()
                    if not line:
                        continue

                    query = json.loads(line)["question"]
                    save_path = os.path.join(details_dir, f"Falcon_{idx}.json")
                    if os.path.exists(save_path):
                        continue

                    if hasattr(bar, "set_postfix"):
                        bar.set_postfix(idx=idx, refresh=False)
                    logger = get_logger(save_dir, idx)
                    thought_log_path = get_thought_log_path(save_dir, idx)
                    asyncio.run(
                        run_single_query(
                            logger,
                            query,
                            save_path,
                            thought_log_path=thought_log_path,
                            paper_from_month=paper_from,
                            paper_to_month=paper_to,
                        )
                    )
        except Exception as exc:
            print(exc)
            continue
