from datetime import date, timedelta
from typing import Any

import requests
from loguru import logger
from omegaconf import ListConfig

from .base import BaseRetriever, register_retriever
from ..protocol import Paper


def reconstruct_abstract(abstract_index: dict[str, list[int]] | None) -> str:
    if not abstract_index:
        return ""

    max_position = max(position for positions in abstract_index.values() for position in positions)
    words = [""] * (max_position + 1)
    for word, positions in abstract_index.items():
        for position in positions:
            words[position] = word
    return " ".join(word for word in words if word)


def normalize_list(value: list[str] | ListConfig | None, config_key: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, (list, ListConfig)):
        raise TypeError(f"config.source.openalex.{config_key} must be a list of strings.")
    if any(not isinstance(item, str) for item in value):
        raise TypeError(f"config.source.openalex.{config_key} must contain only strings.")
    return [item.strip() for item in value if item.strip()]


@register_retriever("openalex")
class OpenAlexRetriever(BaseRetriever):
    api_url = "https://api.openalex.org/works"

    def __init__(self, config):
        super().__init__(config)
        self.api_key = self.retriever_config.get("api_key")
        self.search_queries = normalize_list(self.retriever_config.get("search_queries"), "search_queries")
        if not self.search_queries:
            raise ValueError("config.source.openalex.search_queries must contain at least one query.")
        self.days = int(self.retriever_config.get("days", 14))
        self.per_query = int(self.retriever_config.get("per_query", 50))
        self.max_raw_papers = int(self.retriever_config.get("max_raw_papers", 200))

    def _retrieve_raw_papers(self) -> list[dict[str, Any]]:
        from_date = (date.today() - timedelta(days=self.days)).isoformat()
        seen_ids: set[str] = set()
        raw_papers: list[dict[str, Any]] = []

        for query in self.search_queries:
            logger.info(f"Retrieving OpenAlex works for query: {query}")
            params = {
                "search": query,
                "filter": f"from_publication_date:{from_date}",
                "sort": "publication_date:desc",
                "per-page": min(self.per_query, 200),
                "select": ",".join([
                    "id",
                    "doi",
                    "display_name",
                    "authorships",
                    "abstract_inverted_index",
                    "primary_location",
                    "best_oa_location",
                    "publication_date",
                ]),
            }
            if self.api_key:
                params["api_key"] = self.api_key

            response = requests.get(self.api_url, params=params, timeout=(10, 60))
            response.raise_for_status()
            results = response.json().get("results", [])
            for item in results:
                paper_id = item.get("doi") or item.get("id")
                if not paper_id or paper_id in seen_ids:
                    continue
                seen_ids.add(paper_id)
                raw_papers.append(item)
                if len(raw_papers) >= self.max_raw_papers:
                    return raw_papers[:10] if self.config.executor.debug else raw_papers

        if self.config.executor.debug:
            raw_papers = raw_papers[:10]
        return raw_papers

    def convert_to_paper(self, raw_paper: dict[str, Any]) -> Paper | None:
        title = raw_paper.get("display_name") or ""
        abstract = reconstruct_abstract(raw_paper.get("abstract_inverted_index"))
        if not title or not abstract:
            return None

        authors = [
            authorship.get("author", {}).get("display_name")
            for authorship in raw_paper.get("authorships", [])
            if authorship.get("author", {}).get("display_name")
        ]
        url = raw_paper.get("doi") or raw_paper.get("id")
        best_oa_location = raw_paper.get("best_oa_location") or {}
        primary_location = raw_paper.get("primary_location") or {}
        pdf_url = best_oa_location.get("pdf_url") or primary_location.get("pdf_url")

        return Paper(
            source=self.name,
            title=title,
            authors=authors,
            abstract=abstract,
            url=url,
            pdf_url=pdf_url,
            full_text=None,
        )
