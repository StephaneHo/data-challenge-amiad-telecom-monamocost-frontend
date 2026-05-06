from collectors.arxiv_collector import ArxivCollector

collector = ArxivCollector()

print("Test 1, collecte cs.AI sur 5 jours")
papers = list(collector.collect(categories=["cs.AI"], days_back=5))
print(f"articles collectés: {len(papers)}")
assert len(papers) > 0, "On devrait avoir récupéré un article"

print("Test 2, champs obligatoires présents")
p = papers[0]


assert p["arxiv_id"], "arxiv_id manquant"
assert p["title"], "titre manquant"
assert p["abstract"], "abstract manquant"
assert p["pdf_url"], "pdf_url manquant"
assert p.get("authors") is not None, "authors manquant (None)"
assert len(p["authors"]) > 0, "authors manquant (liste vide)"

print(f"  OK — titre      : {p['title'][:60]}...")
print(f"  Publié le       : {p['published_at']}")
print(f"  Auteurs         : {[a['name'] for a in p['authors'][:3]]}")
print(f"  Catégories      : {p['categories']}")
