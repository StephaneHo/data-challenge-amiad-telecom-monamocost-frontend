import requests

"""
Il faut lancer avant : 
racine: docker compose up redis -d
backend: uvicorn api.app:app --reload --port 8000
test_api.py utilise requests.get("http://localhost:8000/...") 
il fait de vraies requêtes HTTP vers un serveur qui doit être déjà en train de tourner.
Uvicorn est le serveur HTTP qui reçoit les requêtes et appelle les fonctions FastAPI correspondantes.
"""


BASE = "http://localhost:8000"


print("Test API Fast API")

print("Test 1: healtth check")

r = requests.get(f"{BASE}/health")
assert r.status_code == 200
assert r.json()["status"] == "ok"

print("Test 2: List des articles")
r = requests.get(f"{BASE}/papers?limit=3")
assert r.status_code == 200, "l'API pour la liste des papers n'est pas correcte"
papers = r.json()
assert len(papers) > 0, "Aucun article trouvé"

print("Test 3: Retour d'un paper par id")
arxiv_id = papers[0]["arxiv_id"]
r = requests.get(f"{BASE}/papers/{arxiv_id}")
assert r.status_code == 200, "Pas possible de récupérer un article par arxiv_id"
