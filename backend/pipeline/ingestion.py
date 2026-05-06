from loguru import logger
from sqlalchemy.orm import Session
from config import settings
from sqlalchemy.dialects.postgresql import insert
from collectors.arxiv_collector import ArxivCollector
from sentence_transformers import SentenceTransformer
import torch
from typing import Optional
from database.models import Paper, Author, Category, PaperChunk


class PaperIngestionPipeline:
    """
    On injecte une session DB (SQLAlchemy) dans le pipeline
    """

    def __init__(self, session: Session) -> None:
        self.session = session  # on stocke la session pour pouvoir l'utiliser partout
        self.arxiv = ArxivCollector()  # on crée le collector arXiv une seule fois
        """
        On va mettre en place un cache pour pouvoir gérer 
        l'erreur UniqueViolation: duplicate key value violates unique constraint "categories_code_key"
        Key (code)=(cs.AI) already exists.
        """
        self._cat_cache: dict[str, Category] = {}
        self._auth_cache: dict[str, Author] = {}
        logger.info(f"Pipeline: chargement du modèle: {settings.EMBEDDING_MODEL}")
        self.embedder = SentenceTransformer(settings.EMBEDDING_MODEL)
        logger.info("[Pipeline] Modèle chargé")

    def run(
        self,
        categories: Optional[list[str]] = None,
        days_back: int = 7,
        max_results: int | None = None,
    ) -> int:
        """Lance la collecte complète. Retourne le nombre d'articles traités."""
        logger.info(
            f"[Démarrage de la pipeline] - categories={categories} jours={days_back} nb_max de papers retournés={max_results}"
        )
        total = 0

        for raw in self.arxiv.collect(
            categories=categories, days_back=days_back, max_results=max_results
        ):
            try:
                self._process_paper(raw)
                total += 1
                # commit tous les 20 articles pour ne pas perdre trop de travail
                if total % 20 == 0:
                    self.session.commit()
                    logger.info(f"[Pipeline] {total} articles traités")
            except Exception as ex:
                logger.warning(f"[Pipeline] Erreur sur {raw.get('arxiv_id')} : {ex}")
                self.session.rollback()

        self.session.commit()
        logger.info(f"[Pipeline] terminé - {total} articles ingérés")
        return total

    def _embed(self, text: str) -> list[float]:
        """
        Génère un vecteur normalisé
        On tronque à 1024 caractères
        torch.no_grad() désactive le calcul des gradients
        """
        with torch.no_grad():
            return self.embedder.encode(
                text[:1024], normalize_embeddings=True, show_progress_bar=False
            ).tolist()

    def _insert_or_update_paper(self, raw: dict, embedding: list[float]) -> Paper:
        """
        raw[quelque chose] => accès strict (erreur si non présent),
        raw.get(quelque chose) => accès flexible, retourne None si la clé n'existe pas
        """
        statement = (
            insert(Paper)
            .values(
                arxiv_id=raw["arxiv_id"],
                title=raw["title"],
                abstract=raw.get("abstract"),
                pdf_url=raw.get("pdf_url"),
                html_url=raw.get("html_url"),
                doi=raw.get("doi"),
                published_at=raw.get("published_at"),
                updated_at=raw.get("updated_at"),
                embedding=embedding,
            )
            .on_conflict_do_update(
                index_elements=["arxiv_id"],
                set_={
                    "title": raw["title"],
                    "abstract": raw.get("abstract") or "",
                    "updated_at": raw.get("updated_at"),
                    "embedding": embedding,
                },
            )
        )
        self.session.execute(statement)
        return self.session.get(Paper, raw["arxiv_id"])

    def _insert_or_update_authors(self, paper: Paper, authors: list[dict]) -> None:
        for a in authors:
            name = (a.get("name") or "").strip()
            if not name:
                continue
            """
            query: recherche en base (fait un SELECT * FROM authors WHERE name = 'nom d'un auteur'
            et retourne un objet Author sinon seconde partie: crée un objet Author en mémoire
            """
            if name in self._auth_cache:
                author = self._auth_cache[name]
            else:
                author = self.session.query(Author).filter_by(name=name).first()
                if author is None:
                    author = Author(name=name)
                    self.session.add(author)
                    self.session.flush()
                self._auth_cache[name] = author
            """
            Si Arxiv donne une affiliation, on la prend, sinon on garde l'anncienne
            """
            author.affiliation = a.get("affiliation") or author.affiliation
            self.session.add(author)
            if author not in paper.authors:
                paper.authors.append(author)

    def _insert_or_update_categories(self, paper: Paper, codes: list[str]):
        """
        le flush va pouvoir permettre d'écrire en base dans la transaction sans commiter - rend cat visible pour les prochaines requetes
        Sans flush(), les catégories restaient uniquement dans la mémoire Python de SQLAlchemy, pas dans le brouillon PostgreSQL
        Avec flush(), la catégorie est écrite dans le brouillon PostgreSQL immédiatement, sans commiter
        """
        for code in codes:
            if code in self._cat_cache:
                cat = self._cat_cache[code]
            else:
                cat = self.session.query(Category).filter_by(code=code).first()

                if cat is None:
                    cat = Category(code=code)
                    self.session.add(cat)
                    self.session.flush()
                self._cat_cache[code] = cat
            if cat not in paper.categories:
                paper.categories.append(cat)

    def _insert_or_update_abstract_chunk(self, paper: Paper, abstract: str) -> None:
        """
        Pour l'instant on fait un seul chunk par article
        chunk_index = 0: c'dst la convention pour désigner l'abstract
        Certains articles n'ont pas d'abstract (rare)"""
        if not abstract:
            return
        embedding = self._embed(abstract)
        # on cherche si le chunk existe déjà
        existing = (
            self.session.query(PaperChunk)
            .filter_by(paper_id=paper.arxiv_id, chunk_index=0)
            .first()
        )
        if existing:
            existing.content = abstract
            existing.embedding = embedding
        else:
            self.session.add(
                PaperChunk(
                    paper_id=paper.arxiv_id,
                    chunk_index=0,
                    content=abstract,
                    section="Abstract",
                    embedding=embedding,
                )
            )

    def _process_paper(self, raw: dict) -> None:
        abstract = raw.get("abstract") or ""
        """ On va mettre dans l'embedding à la fois le titre et l'abstract"""
        embedding = self._embed(raw["title"] + " " + abstract)

        paper = self._insert_or_update_paper(raw, embedding)
        self._insert_or_update_authors(paper, raw.get("authors", []))
        self._insert_or_update_categories(paper, raw.get("categories", []))
        self._insert_or_update_abstract_chunk(paper, abstract)
