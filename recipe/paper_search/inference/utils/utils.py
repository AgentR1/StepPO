import asyncio
import json
import os
import re
import threading
from functools import total_ordering
from typing import Any, Optional

import httpx
from pydantic import BaseModel
from sortedcontainers import SortedList

import env_config  # noqa: F401
from .http_retry import httpx_request_with_retry

DEFAULT_PAPER_FIELDS = "title,abstract,year,authors,externalIds"

# Inclusive publication month range for /paper/search (YYYY-MM). Override if backend uses other names.
SEARCH_RANGE_FROM_PARAM = os.getenv("PAPER_SEARCH_V2_RANGE_FROM_PARAM", "from")
SEARCH_RANGE_TO_PARAM = os.getenv("PAPER_SEARCH_V2_RANGE_TO_PARAM", "to")
SERPER_SEARCH_URL = os.getenv("SERPER_SEARCH_URL", "https://google.serper.dev/search")


class ApiKeyPool:
    def __init__(self, keys: list[str]):
        self.keys = list(keys)
        self.current_index = 0
        self._lock = threading.Lock()

    def get_next_key(self) -> Optional[str]:
        with self._lock:
            if not self.keys:
                return None
            key = self.keys[self.current_index % len(self.keys)]
            self.current_index = (self.current_index + 1) % len(self.keys)
            return key

    def remove_key(self, key: str) -> None:
        with self._lock:
            if key not in self.keys:
                return
            self.keys.remove(key)
            if self.keys:
                self.current_index %= len(self.keys)
            else:
                self.current_index = 0

    def snapshot(self) -> list[str]:
        with self._lock:
            return list(self.keys)


def _format_authors(authors: Any) -> str:
    if not authors:
        return ""
    if isinstance(authors, str):
        return authors
    if isinstance(authors, list):
        names: list[str] = []
        for author in authors:
            if isinstance(author, dict):
                name = author.get("name")
                if name:
                    names.append(str(name))
            elif author:
                names.append(str(author))
        return ", ".join(names)
    return str(authors)


GOOGLE_KEYS = [
    "43518041099660a2575a863feb50b0f945f9cf8e",
    "e855c371aaeb1d1318ce28dd7ca12dc643c011e6",
    "dfddf51e7b4eda349fd966b6786585180cb1ae51",
    "7dd8e1ebcc5caa7af1401cc188a614db0adbb2d3",
    "73381556152daadd9cf49f4dfdfe3d6aca9f1804",
    "572f22bd262c4f29f8da80e601526226632aa937",
    "ea3161476f70afdcc2c36e180b35b7ba92305c03",
    "3d3f23dd4044caaf381c08b039c7933f071f8b98",
    "2323273e281a2a3a76d173fa9ac8c105728d4360",
    "5b804c43d17564a7688ca9799ad4e14ccba25a56",
    "f6b0a4dd35bc089a1491d634d1c1f5e578e4b4a6",
    "7e5fa1ff9b55f93a49d567906215b9e8a0e08244",
    "917b4b3977317ab6803917fc693d7bb2578c8f69",
    "21684c929292fbf81142dbb88b35443850654cff",
    "a5205f5e1d47fc0006bd06c620a84af560abe571",
    "6dbec7f212f4676a95186c3b2fb9e7a4fc7f85e8",
    "ed6a22d574158595bfbab57dbd4bab37fabbb3b8",
    "ec2e36c3b9445cbf03fdbc92f725a2465742bebd",
    "522b2e8777db125279e957f07bb95dc6ed7e5c87",
    "2a9a23b1bf4a5c236946f784c22f568ded363c91",
    "dd2e0875e6f4a233b8cea6bf62ab3d480a5097fb",
]
serper_api_key_pool = ApiKeyPool(GOOGLE_KEYS)


class Paper(BaseModel):
    paper_id: str
    raw_paper_id: str = ""
    arxiv_id: str = ""
    title: str
    abstract: str
    authors: str = ""
    year: Optional[int] = None
    score: float = 0.0


_ARXIV_NEW_STYLE_PREFIX = re.compile(
    r"^(?:[a-z.-]+/)?(\d{2})(\d{2})\.\d{4,5}(?:v\d+)?$",
    re.IGNORECASE,
)
_ARXIV_URL_PATTERN = re.compile(
    r"arxiv\.org/(?:abs|pdf|html)/([a-z.-]+/\d{7}|\d{4}\.\d{4,5})(?:v\d+)?(?:\.pdf)?(?:[/?#].*)?$",
    re.IGNORECASE,
)


def parse_arxiv_submission_year_month(arxiv_like: str) -> Optional[tuple[int, int]]:
    """Parse arXiv new-style id YYMM.NNNNN (optionally with category prefix) into (full_year, month)."""
    if not arxiv_like:
        return None
    t = arxiv_like.strip()
    for p in ("arxiv:", "arXiv:"):
        if t.lower().startswith(p):
            t = t[len(p) :].strip()
    m = _ARXIV_NEW_STYLE_PREFIX.match(t)
    if not m:
        return None
    yy, mm = int(m.group(1)), int(m.group(2))
    if not 1 <= mm <= 12:
        return None
    full_year = 1900 + yy if yy >= 91 else 2000 + yy
    return full_year, mm


def paper_on_or_before_year_month(paper: Paper, max_year: int, max_month: int) -> bool:
    """Keep papers with arXiv YYMM on/before (max_year, max_month), or year < max_year when no arXiv month."""
    cutoff = (max_year, max_month)
    for candidate in (paper.arxiv_id, paper.paper_id, paper.raw_paper_id):
        ym = parse_arxiv_submission_year_month(candidate)
        if ym:
            return ym <= cutoff
    if paper.year is not None:
        if paper.year < max_year:
            return True
        if paper.year > max_year:
            return False
        return False
    return False


def parse_year_month_str(value: str) -> tuple[int, int]:
    """Parse 'YYYY-MM' into (year, month)."""
    t = value.strip()
    if len(t) != 7 or t[4] != "-":
        raise ValueError(f"Expected YYYY-MM, got {value!r}")
    y, m = int(t[:4]), int(t[5:7])
    if not 1 <= m <= 12:
        raise ValueError(f"Invalid month in {value!r}")
    return y, m


def paper_publication_month_bounds(paper: Paper) -> Optional[tuple[tuple[int, int], tuple[int, int]]]:
    """Return inclusive ((y1,m1),(y2,m2)) if a publication month range can be inferred."""
    for candidate in (paper.arxiv_id, paper.paper_id, paper.raw_paper_id):
        ym = parse_arxiv_submission_year_month(candidate)
        if ym:
            return ym, ym
    if paper.year is not None:
        y = paper.year
        return (y, 1), (y, 12)
    return None


def paper_overlaps_year_month_range(
    paper: Paper,
    from_ym: Optional[tuple[int, int]],
    to_ym: Optional[tuple[int, int]],
) -> bool:
    """If both bounds are None, accept all papers. Otherwise keep papers whose inferred month range overlaps [from_ym, to_ym]."""
    if from_ym is None and to_ym is None:
        return True
    bounds = paper_publication_month_bounds(paper)
    if bounds is None:
        return False
    (py_lo, py_hi) = bounds
    r_lo = from_ym or (0, 1)
    r_hi = to_ym or (9999, 12)
    return not (py_hi < r_lo or py_lo > r_hi)


@total_ordering
class PaperPoolEntry(BaseModel):
    paper: Paper
    source: str
    origin: str
    score: float
    expand: bool = False

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, PaperPoolEntry):
            return NotImplemented
        if self.score != other.score:
            return self.score < other.score
        return self.paper.paper_id < other.paper.paper_id

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PaperPoolEntry):
            return NotImplemented
        return self.score == other.score and self.paper.paper_id == other.paper.paper_id

    def __hash__(self) -> int:
        return hash(self.paper.paper_id)


class PaperPool:
    def __init__(self, max_size: int = 20, threshold: float = 0.0, max_abstract_words: int = 400):
        self.papers: dict[str, PaperPoolEntry] = {}
        self.ranked_papers: SortedList[PaperPoolEntry] = SortedList()
        self.max_size = max_size
        self.threshold = threshold
        self.max_abstract_words = max_abstract_words

    def add_paper(self, paper: Paper, source: str, origin: str, score: float) -> None:
        if paper.paper_id in self.papers:
            return

        paper_pool_entry = PaperPoolEntry(paper=paper, source=source, origin=origin, score=score)
        self.papers[paper.paper_id] = paper_pool_entry
        self.ranked_papers.add(paper_pool_entry)

    def get_paper(self, paper_id: str) -> Optional[PaperPoolEntry]:
        return self.papers.get(paper_id)

    def has_paper(self, paper_id: str) -> bool:
        return paper_id in self.papers

    def _len_relevant_papers(self) -> int:
        return len([entry for entry in self.papers.values() if entry.score >= 0.1])

    @property
    def paper_list(self) -> str:
        if not self.papers:
            return "No papers in the pool."

        expanded_entries = [e for e in self.ranked_papers if e.expand and e.score >= self.threshold]
        unexpanded_entries = [e for e in self.ranked_papers if not e.expand and e.score >= self.threshold]

        expanded_entries.reverse()
        unexpanded_entries.reverse()

        half_size = self.max_size // 2
        top_expanded = expanded_entries[:half_size]
        top_unexpanded = unexpanded_entries[:half_size]

        display_entries = top_expanded + top_unexpanded
        display_entries.sort(key=lambda x: x.score, reverse=True)

        if not display_entries:
            return "No relevant papers found above threshold."

        description = (
            "Paper Pool Status:\n"
            "- [EXP]: Paper has been expanded already.\n"
            "- [NEW]: New paper found via search or expansion.\n"
            "- Format: [paper_id] (score) [STATUS] Title\n"
        )

        lines = [description]
        for entry in display_entries:
            paper = entry.paper
            status_tag = "[EXP]" if entry.expand else "[NEW]"
            abstract = paper.abstract
            words = abstract.split()
            if len(words) > self.max_abstract_words:
                abstract = " ".join(words[: self.max_abstract_words]) + "..."

            entry_str = f"[{paper.paper_id}] ({entry.score:.2f}) {status_tag} {paper.title}\nAbstract: {abstract}"
            lines.append(entry_str)

        return "\n\n".join(lines)


class PaperSearchV2Client:
    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: float = 30.0,
        *,
        api_key: Optional[str] = None,
        serper_api_keys: Optional[list[str]] = None,
        max_concurrency: Optional[int] = 16,
        max_detail_concurrency: Optional[int] = 16,
    ):
        self.base_url = (
            base_url or os.getenv("PAPER_SEARCH_V2_BASE_URL") or "http://172.16.100.204:4000"
        ).rstrip("/")
        self.api_key = api_key if api_key is not None else os.getenv("PAPER_SEARCH_V2_API_KEY")
        self._serper_key_pool = ApiKeyPool(serper_api_keys) if serper_api_keys is not None else serper_api_key_pool
        headers: dict[str, str] = {}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=headers or None,
            timeout=httpx.Timeout(timeout, connect=10.0, pool=60.0),
            limits=httpx.Limits(max_connections=1024, max_keepalive_connections=128),
        )
        self._semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency and max_concurrency > 0 else None
        self._detail_semaphore = (
            asyncio.Semaphore(max_detail_concurrency) if max_detail_concurrency and max_detail_concurrency > 0 else None
        )

    async def _request(
        self, method: str, url: str, *, semaphore: Optional[asyncio.Semaphore] = None, **kwargs: Any
    ) -> httpx.Response:
        sem = self._semaphore if semaphore is None else semaphore
        return await httpx_request_with_retry(self.client, method, url, semaphore=sem, **kwargs)

    async def close(self) -> None:
        await self.client.aclose()

    async def __aenter__(self) -> "PaperSearchV2Client":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()

    @staticmethod
    def _record_to_paper(data: dict[str, Any]) -> Paper:
        raw_paper_id = str(data.get("paperId") or data.get("paper_id") or "")
        external_ids = data.get("externalIds") or {}
        arxiv_id = str(data.get("arxiv_id") or external_ids.get("ArXiv") or "")
        paper_id = arxiv_id or raw_paper_id
        return Paper(
            paper_id=paper_id,
            raw_paper_id=raw_paper_id,
            arxiv_id=arxiv_id,
            title=str(data.get("title") or ""),
            abstract=str(data.get("abstract") or ""),
            authors=_format_authors(data.get("authors")),
            year=data.get("year"),
            score=float(data.get("score", 0.0) or 0.0),
        )

    @staticmethod
    def _extract_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
        data = payload.get("data")
        return data if isinstance(data, list) else []

    @staticmethod
    def _normalize_arxiv_id(arxiv_id: str) -> str:
        return re.sub(r"v\d+$", "", arxiv_id.strip(), flags=re.IGNORECASE)

    @classmethod
    def _extract_arxiv_id_from_url(cls, url: str) -> Optional[str]:
        if not url:
            return None
        match = _ARXIV_URL_PATTERN.search(url.strip())
        if not match:
            return None
        return cls._normalize_arxiv_id(match.group(1))

    @staticmethod
    def _build_google_search_query(
        query: str,
        *,
        from_month: Optional[str] = None,
        to_month: Optional[str] = None,
    ) -> str:
        parts = [query.strip(), "site:arxiv.org"]
        if from_month:
            y, m = parse_year_month_str(from_month)
            parts.append(f"after:{y:04d}-{m:02d}-01")
        if to_month:
            y, m = parse_year_month_str(to_month)
            next_y, next_m = (y + 1, 1) if m == 12 else (y, m + 1)
            parts.append(f"before:{next_y:04d}-{next_m:02d}-01")
        return " ".join(part for part in parts if part)

    async def _search_google(
        self,
        query: str,
        limit: int,
        *,
        from_month: Optional[str] = None,
        to_month: Optional[str] = None,
        fields: str = DEFAULT_PAPER_FIELDS,
    ) -> list[Paper]:
        initial_keys = self._serper_key_pool.snapshot()
        if not initial_keys:
            raise ValueError("Google API key pool is empty when source='google'")
        if limit <= 0:
            return []
        if limit > 10:
            raise ValueError("Google search via Serper supports up to 10 results per request")

        search_query = self._build_google_search_query(query, from_month=from_month, to_month=to_month)
        payload = {"q": search_query, "num": limit, "page": 1}
        attempted_keys: set[str] = set()
        last_exc: Optional[Exception] = None
        resp: Optional[httpx.Response] = None
        max_attempts = len(initial_keys)

        while len(attempted_keys) < max_attempts:
            serper_api_key = self._serper_key_pool.get_next_key()
            if not serper_api_key:
                break
            attempted_keys.add(serper_api_key)
            try:
                resp = await self._request(
                    "POST",
                    SERPER_SEARCH_URL,
                    headers={
                        "X-API-KEY": serper_api_key,
                        "Content-Type": "application/json",
                    },
                    content=json.dumps(payload),
                )
                resp.raise_for_status()
                break
            except Exception as exc:
                last_exc = exc
                self._serper_key_pool.remove_key(serper_api_key)
                resp = None

        if resp is None:
            if last_exc is not None:
                raise RuntimeError("All Serper API keys failed") from last_exc
            raise ValueError("Google API key pool is empty when source='google'")

        result = resp.json()
        organic = result.get("organic")
        if not isinstance(organic, list):
            return []

        paper_ids: list[str] = []
        seen_ids: set[str] = set()
        for item in organic:
            if not isinstance(item, dict):
                continue
            paper_id = self._extract_arxiv_id_from_url(str(item.get("link") or ""))
            if not paper_id or paper_id in seen_ids:
                continue
            seen_ids.add(paper_id)
            paper_ids.append(paper_id)

        tasks = [self.get_paper(paper_id, fields=fields) for paper_id in paper_ids]
        papers = await asyncio.gather(*tasks) if tasks else []
        return [paper for paper in papers if paper is not None]

    async def search(
        self,
        query: str,
        limit: int = 10,
        *,
        source: str = "local_db",
        year: Optional[str] = None,
        from_month: Optional[str] = None,
        to_month: Optional[str] = None,
        min_citation_count: Optional[int] = None,
        fields: str = DEFAULT_PAPER_FIELDS,
    ) -> list[Paper]:
        if source == "google":
            return await self._search_google(
                query=query,
                limit=10,
                from_month=from_month,
                to_month=to_month,
                fields=fields,
            )
        if source != "local_db":
            raise ValueError(f"Invalid source: {source}")

        params: dict[str, Any] = {"query": query, "limit": limit}
        if year:
            params["year"] = year
        if from_month:
            params[SEARCH_RANGE_FROM_PARAM] = from_month
        if to_month:
            params[SEARCH_RANGE_TO_PARAM] = to_month
        if min_citation_count is not None:
            params["minCitationCount"] = min_citation_count
        if fields:
            params["fields"] = fields

        resp = await self._request("GET", "/paper/search", params=params)
        resp.raise_for_status()
        payload = resp.json()
        return [self._record_to_paper(item) for item in self._extract_items(payload)]

    async def get_paper(self, paper_id: str, fields: str = DEFAULT_PAPER_FIELDS) -> Optional[Paper]:
        params = {"fields": fields} if fields else None
        resp = await self._request("GET", f"/paper/{paper_id}", params=params, semaphore=self._detail_semaphore)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, dict) or not data:
            return None
        return self._record_to_paper(data)

    async def get_citations(
        self, paper_id: str, limit: int = 50, fields: str = DEFAULT_PAPER_FIELDS
    ) -> list[Paper]:
        params: dict[str, Any] = {"limit": limit}
        if fields:
            params["fields"] = fields

        resp = await self._request("GET", f"/paper/{paper_id}/citations", params=params)
        resp.raise_for_status()
        payload = resp.json()
        items = self._extract_items(payload)
        papers: list[Paper] = []
        for item in items:
            citing_paper = item.get("citingPaper")
            if isinstance(citing_paper, dict):
                papers.append(self._record_to_paper(citing_paper))
        return papers

    async def get_references(
        self, paper_id: str, limit: int = 50, fields: str = DEFAULT_PAPER_FIELDS
    ) -> list[Paper]:
        if limit < 0:
            limit = 99

        params: dict[str, Any] = {"limit": limit}
        if fields:
            params["fields"] = fields

        resp = await self._request("GET", f"/paper/{paper_id}/references", params=params)
        resp.raise_for_status()
        payload = resp.json()
        items = self._extract_items(payload)
        papers: list[Paper] = []
        for item in items:
            cited_paper = item.get("citedPaper")
            if isinstance(cited_paper, dict):
                papers.append(self._record_to_paper(cited_paper))
        return papers
