import asyncio
import logging
import unittest
from unittest.mock import AsyncMock, patch

import httpx

from paper_agent import PaperSearchAgent
from utils import Paper, PaperSearchV2Client, paper_overlaps_year_month_range


class TestPaperSearchV2Client(unittest.TestCase):
    def test_search_uses_api_key_header_and_month_range(self) -> None:
        response = httpx.Response(
            200,
            json={
                "data": [
                    {
                        "paperId": "paper-123",
                        "externalIds": {"ArXiv": "2401.00001"},
                        "title": "Test Paper",
                        "abstract": "Test abstract.",
                        "authors": [{"name": "Alice"}, {"name": "Bob"}],
                        "year": 2024,
                        "score": 12.5,
                    }
                ]
            },
            request=httpx.Request("GET", "http://testserver/paper/search"),
        )

        with patch("utils.utils.httpx_request_with_retry", new=AsyncMock(return_value=response)) as mock_request:
            papers = asyncio.run(self._run_search())

        mock_request.assert_awaited_once()
        _, method, url = mock_request.await_args.args
        kwargs = mock_request.await_args.kwargs

        self.assertEqual(method, "GET")
        self.assertEqual(url, "/paper/search")
        self.assertEqual(kwargs["params"]["query"], "graph neural networks")
        self.assertEqual(kwargs["params"]["limit"], 1)
        self.assertEqual(kwargs["params"]["from"], "2024-01")
        self.assertEqual(kwargs["params"]["to"], "2024-10")
        self.assertEqual(
            mock_request.await_args.args[0].headers["X-API-Key"],
            "lw-d7ea4e41519dc1cd03b322d0faa8fb9b",
        )

        self.assertEqual(
            kwargs["params"]["fields"],
            "title,abstract,year,authors,externalIds",
        )

        self.assertEqual(len(papers), 1)
        self.assertEqual(papers[0].paper_id, "2401.00001")
        self.assertEqual(papers[0].authors, "Alice, Bob")
        self.assertEqual(papers[0].score, 12.5)

    def test_expand_endpoints_share_api_key_header(self) -> None:
        citations_response = httpx.Response(
            200,
            json={"data": [{"citingPaper": {"paperId": "paper-1", "title": "Citing", "abstract": "A"}}]},
            request=httpx.Request("GET", "http://testserver/paper/root/citations"),
        )
        references_response = httpx.Response(
            200,
            json={"data": [{"citedPaper": {"paperId": "paper-2", "title": "Referenced", "abstract": "B"}}]},
            request=httpx.Request("GET", "http://testserver/paper/root/references"),
        )

        with patch(
            "utils.utils.httpx_request_with_retry",
            new=AsyncMock(side_effect=[citations_response, references_response]),
        ) as mock_request:
            citations, references = asyncio.run(self._run_expand_requests())

        self.assertEqual(len(citations), 1)
        self.assertEqual(len(references), 1)
        self.assertEqual(mock_request.await_count, 2)
        for call in mock_request.await_args_list:
            client, method, url = call.args
            self.assertEqual(method, "GET")
            self.assertIn(url, {"/paper/root/citations", "/paper/root/references"})
            self.assertEqual(
                client.headers["X-API-Key"],
                "lw-d7ea4e41519dc1cd03b322d0faa8fb9b",
            )

    def test_google_search_rotates_key_pool_and_resolves_arxiv_papers(self) -> None:
        success_response = httpx.Response(
            200,
            json={
                "organic": [
                    {"link": "https://arxiv.org/abs/2401.00001v2"},
                    {"link": "https://arxiv.org/pdf/2401.00002.pdf"},
                    {"link": "https://example.com/not-arxiv"},
                    {"link": "https://arxiv.org/abs/2401.00001"},
                ]
            },
            request=httpx.Request("POST", "https://google.serper.dev/search"),
        )

        failed_request = httpx.Request("POST", "https://google.serper.dev/search")
        with patch(
            "utils.utils.httpx_request_with_retry",
            new=AsyncMock(side_effect=[httpx.RequestError("bad key", request=failed_request), success_response]),
        ) as mock_request:
            papers, remaining_keys = asyncio.run(self._run_google_search())

        self.assertEqual(mock_request.await_count, 2)
        first_call = mock_request.await_args_list[0]
        second_call = mock_request.await_args_list[1]

        self.assertEqual(first_call.args[1], "POST")
        self.assertEqual(first_call.args[2], "https://google.serper.dev/search")
        self.assertEqual(first_call.kwargs["headers"]["X-API-KEY"], "bad-key")
        self.assertEqual(second_call.kwargs["headers"]["X-API-KEY"], "good-key")
        self.assertEqual(second_call.kwargs["headers"]["Content-Type"], "application/json")
        self.assertEqual(
            second_call.kwargs["content"],
            '{"q": "graph neural networks site:arxiv.org after:2024-01-01 before:2024-11-01", "num": 10, "page": 1}',
        )

        self.assertEqual([paper.paper_id for paper in papers], ["2401.00001", "2401.00002"])
        self.assertEqual([paper.title for paper in papers], ["Paper 2401.00001", "Paper 2401.00002"])
        self.assertEqual(remaining_keys, ["good-key"])

    def test_google_search_requires_non_empty_key_pool(self) -> None:
        async def run() -> None:
            client = PaperSearchV2Client(base_url="http://testserver", serper_api_keys=[])
            try:
                with self.assertRaisesRegex(ValueError, "key pool is empty"):
                    await client.search("graph neural networks", source="google")
            finally:
                await client.close()

        asyncio.run(run())

    def test_google_search_raises_after_all_keys_fail(self) -> None:
        failed_request = httpx.Request("POST", "https://google.serper.dev/search")
        with patch(
            "utils.utils.httpx_request_with_retry",
            new=AsyncMock(
                side_effect=[
                    httpx.RequestError("bad key 1", request=failed_request),
                    httpx.RequestError("bad key 2", request=failed_request),
                ]
            ),
        ):
            async def run() -> None:
                client = PaperSearchV2Client(
                    base_url="http://testserver",
                    serper_api_keys=["bad-key-1", "bad-key-2"],
                )
                try:
                    with self.assertRaisesRegex(RuntimeError, "All Serper API keys failed"):
                        await client.search("graph neural networks", source="google")
                    self.assertEqual(client._serper_key_pool.snapshot(), [])
                finally:
                    await client.close()

            asyncio.run(run())

    def test_paper_overlaps_year_month_range(self) -> None:
        arxiv_paper = Paper(
            paper_id="2410.00001",
            arxiv_id="2410.00001",
            title="October Paper",
            abstract="A",
            year=2024,
        )
        year_only_paper = Paper(
            paper_id="paper-year",
            title="Year Only",
            abstract="B",
            year=2024,
        )
        old_paper = Paper(
            paper_id="2309.00001",
            arxiv_id="2309.00001",
            title="Old Paper",
            abstract="C",
            year=2023,
        )

        self.assertTrue(paper_overlaps_year_month_range(arxiv_paper, (2024, 10), (2024, 10)))
        self.assertTrue(paper_overlaps_year_month_range(year_only_paper, (2024, 10), (2024, 10)))
        self.assertFalse(paper_overlaps_year_month_range(old_paper, (2024, 10), (2024, 10)))

    def test_expand_filters_out_of_range_papers(self) -> None:
        asyncio.run(self._run_expand_filter_test())

    @staticmethod
    async def _run_search():
        client = PaperSearchV2Client(
            base_url="http://testserver",
            api_key="lw-d7ea4e41519dc1cd03b322d0faa8fb9b",
        )
        try:
            return await client.search(
                "graph neural networks",
                limit=1,
                from_month="2024-01",
                to_month="2024-10",
            )
        finally:
            await client.close()

    @staticmethod
    async def _run_expand_requests():
        client = PaperSearchV2Client(
            base_url="http://testserver",
            api_key="lw-d7ea4e41519dc1cd03b322d0faa8fb9b",
        )
        try:
            citations = await client.get_citations("root", limit=5)
            references = await client.get_references("root", limit=5)
            return citations, references
        finally:
            await client.close()

    @staticmethod
    async def _run_google_search():
        client = PaperSearchV2Client(
            base_url="http://testserver",
            api_key="lw-d7ea4e41519dc1cd03b322d0faa8fb9b",
            serper_api_keys=["bad-key", "good-key"],
        )

        async def fake_get_paper(paper_id: str, fields: str = "") -> Paper:
            return Paper(
                paper_id=paper_id,
                arxiv_id=paper_id,
                title=f"Paper {paper_id}",
                abstract=f"Abstract {paper_id}",
            )

        client.get_paper = AsyncMock(side_effect=fake_get_paper)
        try:
            papers = await client.search(
                "graph neural networks",
                limit=3,
                source="google",
                from_month="2024-01",
                to_month="2024-10",
            )
            return papers, client._serper_key_pool.snapshot()
        finally:
            assert client.get_paper.await_count == 2
            await client.close()

    @staticmethod
    async def _run_expand_filter_test():
        logger = logging.getLogger("test_expand_filters_out_of_range_papers")
        logger.handlers.clear()
        agent = PaperSearchAgent(
            logger,
            paper_to_month="2024-10",
        )
        agent.user_query = "test query"
        agent.min_candidate_score = 0.0

        root = Paper(
            paper_id="2409.00001",
            arxiv_id="2409.00001",
            title="Root",
            abstract="Root abstract",
            year=2024,
        )
        in_range = Paper(
            paper_id="2410.00002",
            arxiv_id="2410.00002",
            title="In Range",
            abstract="In range abstract",
            year=2024,
        )
        out_of_range = Paper(
            paper_id="2411.00003",
            arxiv_id="2411.00003",
            title="Out Of Range",
            abstract="Out of range abstract",
            year=2024,
        )

        agent.paper_pool.add_paper(root, "seed", "seed", 1.0)
        agent.ordered_paper_ids.append(root.paper_id)

        agent.client.get_citations = AsyncMock(return_value=[in_range, out_of_range])
        agent.client.get_references = AsyncMock(return_value=[])
        agent.get_relevance_score = AsyncMock(side_effect=[0.8])

        try:
            reward = await agent.expand(root.paper_id)
        finally:
            await agent.close()

        assert reward > 0.0
        assert agent.paper_pool.has_paper(in_range.paper_id)
        assert not agent.paper_pool.has_paper(out_of_range.paper_id)


if __name__ == "__main__":
    unittest.main()
