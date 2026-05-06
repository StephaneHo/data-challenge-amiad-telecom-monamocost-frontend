import requests

BASE = "http://localhost:8000"

print("Test moteur RAG")

print("Test 1: Recherche sémantique")
r = requests.post(
    f"{BASE}/rag/search",
    json={
        "query": "What are the latests advances in large language models ?",
        "top_k": 3,
    },
)
assert r.status_code == 200, f"Erreur: {r.text}"
data = r.json()
print(f"Réponse ({data['tokens_used']}) tokens) :")
print(f"  {data['answer'][:200]}")

print(f"  Références : {len(data['references'])}")
for ref in data["references"]:
    print(f"    - [{ref['arxiv_id']}] {ref['title'][:50]}... score={ref['score']}")
