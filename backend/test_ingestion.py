from pipeline.ingestion import PaperIngestionPipeline
from database.db import get_session
from database.models import Paper, Author, Category, PaperChunk

print("Test Ingestion Pipeline")

with get_session() as session:
    pipeline = PaperIngestionPipeline(session=session)

    print("Test 1: ingestion d'articles de sc.AI")
    count = pipeline.run(categories=["cs.AI"], days_back=7, max_results=50)
    print(f"  {count} articles ingérés")

    print("Test 2 : vérification que tout se passe bien en base")
    nb_papers = session.query(Paper).count()
    nb_authors = session.query(Author).count()
    nb_cats = session.query(Category).count()
    nb_chunks = session.query(PaperChunk).count()

    print(f"Papers: {nb_papers}")
    print(f"Authors: {nb_authors}")
    print(f"Cats: {nb_cats}")
    print(f"Chunks: {nb_chunks}")

    assert nb_papers > 0, "Pas d'article en base"
    assert nb_authors > 0, "Pas d'auteur en base"
    assert nb_cats > 0, "Pas de atégories en base"
    assert nb_chunks > 0, "Pas de chunk en base"

    print("Test 3: vérification des embeddings")
    paper = session.query(Paper).first()
    assert paper.embedding is not None, "Embedding manquant"
    print(f"Titre (tronqué à 60 caractères): {paper.title[:60]}")
    assert len(paper.embedding) == 384, f"Dimension incorrecte: {len(paper.embedding)}"
