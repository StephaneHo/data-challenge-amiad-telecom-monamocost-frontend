### Pour démarrer

docker compose up -d (mode détaché => en arrière plan)



paper représente l’article entier (une ligne en base) :

Typiquement :
paper = {
    "arxiv_id": "...",
    "title": "...",
    "abstract": "...",
    "date": "...",
}
Abstract complet :
"Transformers are very powerful models. They are used in NLP..."

Chunks :
C'est le morceau de document (page ou pargraphe)
Par exemple:
"Transformers are very powerful models."
"They are used in NLP..."

Chaque chunk devient une entrée séparée en base :

chunk = {
    "paper_id": paper.arxiv_id,
    "text": "...",
    "embedding": [...]
}



# docker compose up
V"rifier que :database system is ready to accept connections



#  Initalisation de la BDD Postgres
init-db/01-init.sql est monté dans /docker-entrypoint-initdb.d/


# Ordre des opérations
Docker démarre container backend
        
entrypoint.sh
        
wait-for-it (DB ready)
        
alembic upgrade head
        
CREATE TABLE + INDEX + EXTENSIONS
        
uvicorn FastAPI


### Etape 1: Validation de l'architecture

## Docker Compose PostgreSQL + pgvector
docker compose up db -d
docker compose ps   # db doit être "healthy"

Test 1 — Connexion PostgreSQL
docker compose exec db psql -U postgres -d rag -c "\conninfo"

Test 2 — Extension pgvector active
docker compose exec db psql -U postgres -d rag -c "SELECT extname FROM pg_extension WHERE extname='vector';"

### Etape 2: Migraations Alembic - création des tables

# Vérifier que les tables ont bien été crées
docker compose exec db psql -U postgres -d rag -c "\dt"
les tables doivent bien apparaitre

# Vérifier que la colonne embedding doit apparaitre avec le bon type Vector(384) 
docker compose exec db psql -U postgres -d rag -c "\d papers"
on doit avoir 
embedding           | vector(384)   

### Etape 3: Collecte des articles ArXiv en mémoire et test du collecteur ArXiv de façon isolée 

### Etape 4  Branche la collecte sur PostgreSQL : les articles sont vectorisés (embedding) et stockés en base

### Pour voir les tables

Se connecter au conteneur PostgreSQL
 docker compose exec db psql -U postgres -d rag

On est maintenant dans le shell PostgreSQL. Voici les commandes utiles :

 -- Lister toutes les tables
\dt

-- Voir la structure d'une table
\d papers
\d paper_chunks
\d authors

-- Voir le contenu
SELECT * FROM alembic_version;
SELECT count(*) FROM papers;
SELECT arxiv_id, LEFT(title, 60) AS title FROM papers LIMIT 5;
(si on fait SELECT arxiv_id, title FROM papers LIMIT 5; on rsique de ne rien voir car le terminal ne sera pas assez large)
SELECT count(*) FROM paper_chunks;

-- Quitter
\q

# Lancement des tests

## Collecte des données

Prérequis:
Rien de Docker — pas besoin de PostgreSQL ni Redis
Juste une connexion internet (pour appeler l'API ArXiv)

cd backend
uv run python test_arxiv.py

## Test du pipeline complet ArXiv → embedding → PostgreSQL.

Prérequis :
Docker Desktop lancé
Conteneur db démarré
Connexion internet (ArXiv)

### Terminal 1 — démarrer PostgreSQL
docker compose up db -d

### Vérifier qu'il tourne
docker compose ps

### Terminal 2 — lancer le test
cd backend
uv run python test_ingestion.py

## Test des endpoints FastAPI /health, /papers

Prérequis :
Docker Desktop lancé
Conteneur db démarré (avec des données dedans)
uvicorn qui tourne dans un terminal séparé

### Terminal 1 — PostgreSQL
docker compose up db -d

### Terminal 2 — uvicorn
cd backend
uv run uvicorn api.app:app --reload --port 8000

### Terminal 3 — test (une fois uvicorn prêt)
cd backend
uv run python test_api.py

## Pour tester le RAG complet (retrieval + LLM)

### Terminal 1 — PostgreSQL
docker compose up db -d

### Terminal 2
cd backend
uv run uvicorn api.app:app --reload --port 8000

### Terminal 3 — une fois uvicorn démarré
uv run python test_rag.py

