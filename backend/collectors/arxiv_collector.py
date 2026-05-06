import arxiv
from typing import Generator
from config import settings
from tenacity import retry, stop_after_attempt, wait_exponential
import re
from datetime import datetime, timedelta, timezone
from loguru import logger
import time


class ArxivCollector:
    _DELAY_SECONDS = 3.0

    def __init__(self) -> None:
        # configuration d'un objet client
        self.client = arxiv.Client(
            page_size=100, delay_seconds=self._DELAY_SECONDS, num_retries=5
        )

    @staticmethod
    def _extract_github_urls(text: str) -> list[str]:
        pattern = r"https?://github\.com/[\w\-]+/[\w\-\.]+"
        return list(set(re.findall(pattern, text)))

    def collect(
        self,
        categories: list[str] | None = None,
        days_back: int = 7,
        max_results: int | None = None,
    ) -> Generator[dict, None, None]:
        cats = categories or settings.ARXIV_CATEGORIES
        # arrête dès que les articles sont plus vieux que tant de jours
        since = datetime.now(timezone.utc) - timedelta(days=days_back)
        limit = max_results or settings.ARXIV_MAX_RESULTS

        for cat in cats:
            logger.info(f"[ArXiv] collecte {cat} depuis {since.date()}")

            search = arxiv.Search(
                query=f"cat:{cat}",
                max_results=limit,
                sort_by=arxiv.SortCriterion.SubmittedDate,
                sort_order=arxiv.SortOrder.Descending,
            )

            count = 0
            for result in self.client.results(search):
                # si l'article est trop ancien, on arrête
                # (les résultats sont triés du plus récent au plus ancien)
                if result.published and result.published < since:
                    break
                normalized = self._normalize(result)
                if normalized:
                    yield normalized
                    count += 1
            logger.info(f"[ArXiv] {count} articles for category {cat}")
            time.sleep(self._DELAY_SECONDS)

    def collect_by_id(self, arxiv_ids: list[str]) -> list[dict]:
        search = arxiv.Search(id_list=arxiv_ids)
        return [self._normalize(r) for r in self.client.results(search) if r]

    # temps d’attente entre les essais : 2s, 4s, 10s
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
    def search(self, query: str, max_results: int = 50) -> list[dict]:
        search = arxiv.Search(
            query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance
        )
        # appelle _normalize et filtre les None
        return [n for r in self.client.results(search) if (n := self._normalize(r))]

    def _normalize(self, result: arxiv.Result) -> dict | None:
        try:
            return {
                "arxiv_id": result.get_short_id(),
                "title": result.title.strip(),
                "abstract": result.summary.strip() if result.summary else None,
                "pdf_url": result.pdf_url,
                "html_url": result.entry_id,  # accès direct PDF et page arXiv
                "doi": result.doi,  # identifiant scientfique si dispo
                "published_at": result.published,
                "updated_at": result.updated,
                "authors": [
                    {
                        "name": getattr(a, "name", None),
                    }
                    for a in (getattr(result, "authors", []) or [])
                    if getattr(a, "name", None)
                ],
                "categories": result.categories,  # catégories scientifiques
                "github_urls": self._extract_github_urls(result.comment or ""),
            }
        except Exception as ex:
            logger.warning(f"[ArXiv] Normalisation échouée: {ex}")
            logger.warning("solution de fallback")
            return {
                "arxiv_id": getattr(result, "entry_id", None),
                "title": None,
                "abstract": None,
                "pdf_url": None,
                "html_url": getattr(result, "entry_id", None),
                "doi": None,
                "published_at": None,
                "updated_at": None,
                "authors": [],
                "categories": [],
                "github_urls": [],
            }
