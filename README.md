# Challenge RAG EvalLLM 2026 — Mon Amo Cost

Participation au [challenge **EvalLLM 2026** (atelier conjoint à la conférence TALN/CORIA)](https://evalllm2026.sciencesconf.org/).
Pipeline RAG complète sur corpus PDFs FR (défense, renseignement) avec deux tâches :

- **Tâche 1** — Retrieval `(doc_name, page)` + génération de réponse avec citations
- **Tâche 2** — Attribution phrase par phrase aux sources documentaires

Le projet vise les **bonus open-source / reproductibilité / frugalité** du challenge.

---

## Architecture

```
                ┌────────────────────────────────────────────────────────┐
                │                       Corpus PDFs                       │
                │              (Experimental/DATA/Corpus_raw)             │
                └────────────────────────┬───────────────────────────────┘
                                         │
                                         ▼
                ┌────────────────────────────────────────────────────────┐
   PIPELINE     │  ingest.py                                              │
   D'INDEX      │   ├─ pdf_extractor (pdfplumber + OCR Tesseract fallback)│
                │   ├─ chunker (paragraphes, max 1500 chars)              │
                │   └─ embedder (intfloat/multilingual-e5-base, 768d)     │
                └────────────────────────┬───────────────────────────────┘
                                         │
                                         ▼
                ┌────────────────────────────────────────────────────────┐
   STOCKAGE     │  Postgres + pgvector  (db = am-rag)                     │
                │   documents · document_pages · chunks (+ index HNSW)    │
                └────────────────────────┬───────────────────────────────┘
                                         │
                                         ▼
                ┌────────────────────────────────────────────────────────┐
   INFÉRENCE    │  rag_engine.py                                          │
                │   ├─ retrieve top-K chunks (cosinus pgvector)           │
                │   ├─ agrège à (doc_name, page) → Tâche 1                │
                │   └─ LLM (gpt-4o-mini) → réponse avec [doc.pdf p.N]     │
                │  attribution.py                                         │
                │   ├─ segmente la réponse en phrases                     │
                │   ├─ parse citations + embedding similarity → Tâche 2   │
                │   └─ marque [] les phrases non sourcées                 │
                └────────────────────────┬───────────────────────────────┘
                                         │
                                         ▼
                ┌────────────────────────────────────────────────────────┐
   SORTIE       │  run_challenge.py → JSON Tâche 1 + JSON Tâche 2         │
                │  api/app.py (FastAPI) → POST /challenge/run             │
                └────────────────────────────────────────────────────────┘
```

---

## Prérequis

| Composant | Version | Notes |
|---|---|---|
| Python | 3.11 | Le projet utilise `uv` |
| Docker | récent | pour Postgres + pgvector |
| Tesseract | 5.4+ | OCR français (`fra.traineddata`) |
| Poppler | 26+ | converti PDF → image pour Tesseract |

Sur Windows, Tesseract + Poppler s'installent en user-local (`%LOCALAPPDATA%\tessdata`, `%LOCALAPPDATA%\poppler`). Voir le memo dans `~/.claude/.../memory/reference_environment_setup.md`.

---

## Démarrage rapide

### 1. Cloner et configurer

```powershell
cd backend
uv sync --group experimental    # installe deps + outils notebook
```

Créer `backend/.env` à partir de `.env.example` (à créer si manquant) :
```
DATABASE_URL=postgresql+psycopg2://postgres:postgres@localhost:5432/am-rag
EMBEDDING_MODEL=intfloat/multilingual-e5-base
EMBEDDING_DIM=768

OPENAI_API_KEY=sk-...
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini

# OCR (Windows user-local)
TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
TESSDATA_DIR=C:\Users\<user>\AppData\Local\tessdata
POPPLER_PATH=C:\Users\<user>\AppData\Local\poppler\poppler-26.02.0\Library\bin
OCR_LANG=fra
```

### 2. Démarrer Postgres + appliquer les migrations

```powershell
docker compose up -d db
cd backend
.\.venv\Scripts\python.exe -m alembic upgrade head
```

Trois migrations seront appliquées :
1. `f136a1278da9` — schéma initial (legacy arXiv)
2. `b1c2d3e4f5a6` — drop arXiv + create `documents` / `document_pages` / `chunks`
3. `c2d3e4f5a6b7` — resize embedding column à 768d (e5-base)

### 3. Ingestion du corpus

```powershell
# Tout le corpus
.\.venv\Scripts\python.exe ingest.py

# Sous-ensemble ciblé (8 PDFs des sample queries)
.\.venv\Scripts\python.exe ingest.py --files `
  "..\Experimental\DATA\Corpus_raw\20260108_NP_Obsdrones_Bulletin-de-veille-n12_0.pdf" `
  ...
```

Vérification :
```powershell
docker exec postgres-rag psql -U postgres -d am-rag -c "SELECT doc_name, n_pages, extraction_mode FROM documents;"
```

### 4. Exécuter le challenge

```powershell
# Lance Tâche 1 + Tâche 2 sur un fichier de questions au format challenge
.\.venv\Scripts\python.exe run_challenge.py `
  --input ..\Experimental\DATA\training\sample_queries.json `
  --out-task1 ..\Experimental\DATA\runs\sample_task1.json `
  --out-task2 ..\Experimental\DATA\runs\sample_task2.json
```

Ou via l'API FastAPI :
```powershell
.\.venv\Scripts\python.exe -m uvicorn api.app:app --reload
# Puis :  POST http://localhost:8000/challenge/run  (body = JSON challenge)
# Docs :  http://localhost:8000/docs
```

---

## Reranking (optionnel)

Le reranking ajoute un second tri après le retrieval dense/BM25/hybride. Il permet
de sortir un grand pool initial (`--top-k-chunks 100` ou `1000`), puis de garder
les chunks les plus pertinents pour l'agrégation `(doc_name, page)` et le contexte LLM.

Configuration possible dans `backend/.env` :

```env
RERANK_ENABLED=false
RERANKER_MODEL=BAAI/bge-reranker-v2-m3
RERANKER_TOP_K=30
RERANKER_BATCH_SIZE=16
```

Test standard avec pool initial de 100 chunks :

```powershell
cd data-challenge-amiad-telecom-monamocost-frontend\backend
.\.venv\Scripts\python.exe run_challenge.py `
  --input ..\Experimental\DATA\training\sample_queries.json `
  --out-task1 ..\Experimental\DATA\runs\sample_task1_rerank.json `
  --out-task2 ..\Experimental\DATA\runs\sample_task2_rerank.json `
  --retrieval-mode hybrid `
  --top-k-chunks 100 `
  --rerank `
  --reranker-top-k 30 `
  --context-chunks 10
```

Test plus large pour les documents très similaires :

```powershell
.\.venv\Scripts\python.exe run_challenge.py `
  --input ..\Experimental\DATA\training\sample_queries.json `
  --out-task1 ..\Experimental\DATA\runs\sample_task1_rerank_k1000.json `
  --out-task2 ..\Experimental\DATA\runs\sample_task2_rerank_k1000.json `
  --retrieval-mode hybrid `
  --top-k-chunks 1000 `
  --rerank `
  --reranker-top-k 30 `
  --context-chunks 10
```

Pour comparer uniquement le retrieval/reranking, sans appel LLM :

```powershell
.\.venv\Scripts\python.exe run_challenge.py `
  --input ..\Experimental\DATA\training\sample_queries.json `
  --out-task1 ..\Experimental\DATA\runs\sample_task1_rerank_retrieval_only.json `
  --no-task2 `
  --retrieval-only `
  --retrieval-mode hybrid `
  --top-k-chunks 100 `
  --rerank `
  --reranker-top-k 30
```

Via l'API FastAPI :

```http
POST /challenge/run?rerank=true&reranker_top_k=30
```

---

## Décodage contraint (optionnel)

Le pipeline officiel peut forcer la sortie du LLM dans un schéma JSON strict via
le flag `--constrained-decoding`. Utile pour :

- garantir un format de réponse stable même sur Mistral local / Qwen / Llama,
- récupérer les citations structurées (`doc_name` + `page`) sans regex,
- détecter les **citations hors-pool** (hallucinations type `[doc_inconnu.pdf p.X]`)
  en les loggant en `metadata.invalid_citations`.

Le module `backend/utils/structured_decoder.py` dispatch selon le provider :

| Provider configuré | Backend automatique | Mécanisme |
|---|---|---|
| `openai` (cloud) | `openai_schema` | Structured Outputs `json_schema` strict |
| `openai` + `LLM_BASE_URL` | `vllm_guided` | vLLM `extra_body.guided_json` |
| `mistral` | `mistral_json` | JSON mode + validation Pydantic |
| `anthropic` | `anthropic_tool` | Tool use `input_schema` |
| LLM HF local | `outlines_local` | Outlines `generate.json` (in-process) |

Pour le backend `outlines_local` (Outlines côté client) :
```powershell
.\.venv\Scripts\uv.exe sync --group constrained
```

Utilisation :
```powershell
# Backend choisi automatiquement selon LLM_PROVIDER
.\.venv\Scripts\python.exe run_challenge.py `
  --input ..\Experimental\DATA\training\sample_queries.json `
  --constrained-decoding

# Forcer un backend explicite (ex: Outlines local sur un Mistral-7B HF)
.\.venv\Scripts\python.exe run_challenge.py `
  --input ..\Experimental\DATA\training\sample_queries.json `
  --constrained-decoding `
  --constrained-backend outlines_local
```

Via l'API FastAPI :
```http
POST /challenge/run?constrained_decoding=true&constrained_backend=auto
```

Le JSON Tâche 1 produit annote chaque question avec :
```json
"metadata": {
  "tokens_used": 4913,
  "constrained_backend": "openai_schema",
  "structured_citations": [{"doc_name": "...", "page": 17}],
  "invalid_citations": []
}
```

`structured_citations` est la liste des citations *présentes dans le pool retrieval*,
`invalid_citations` la liste des citations rejetées (doc/page que le LLM a inventé
ou mal recopié, hors top-K). Le champ `answer` reste un texte avec citations inline
`[doc.pdf p.N]` pour rester compatible avec l'attribution Tâche 2.

---

## Fine-tuning de l'embedder

### Pourquoi fine-tuner ?

`intfloat/multilingual-e5-base` est pré-entraîné sur du texte web généraliste. Sur le corpus
défense/renseignement du challenge, le baseline zero-shot atteint **hits@10 = 63 %** seulement
sur le sample (scores cosinus tous tassés entre 0,83 et 0,88 → le modèle a du mal à discriminer
les chunks proches).

L'idée du fine-tuning : montrer au modèle **quelles questions matchent quels chunks** dans
*notre* domaine, pour qu'il pousse les paires positives plus haut et éloigne les négatives.

### Comment ça marche (vue conceptuelle)

**Loss contrastive multi-positifs InfoNCE symétrique** (trame reprise du notebook
`Experimental/Exploration_données.ipynb`).

Pour chaque batch :

```
batch = [
    (q1, paragraph_A_page17, key=("docA.pdf", 17)),
    (q1, paragraph_B_page17, key=("docA.pdf", 17)),   # même page → positif partagé
    (q2, paragraph_C_page5,  key=("docB.pdf", 5)),
    ...
]
```

1. On encode **questions** avec préfixe `query: ...` (convention e5)
2. On encode **paragraphes** avec préfixe `passage: ...`
3. On calcule la matrice de similarité cosinus `Q · Pᵀ` divisée par `temperature=0.07`
4. **Tous les paragraphes ayant la même clé `(doc_name, page)` que la question** sont positifs
5. On rétropropage une loss qui maximise la sim positifs et minimise la sim négatifs
   (dans les deux sens : query→passage et passage→query)

→ Les poids du modèle bougent pour rapprocher les vrais matchs et écarter les faux.

### Données d'entraînement (1796 paires)

Construites depuis 3 sources, qu'on combine via `--gold` + `--extra-examples` :

| Source | Paires | Origine | Qualité |
|---|--:|---|---|
| `sample_queries.json` (gold) | **23** | 4 questions du challenge × ~6 chunks chacune (les chunks DB des `(doc, page)` annotés) | ⭐⭐⭐ ground truth |
| `extra_training_examples.json` | **1** | Exemple OSINT/SSL du `target.txt` (hors corpus) | ⭐⭐⭐ ground truth |
| `synthetic_questions.json` | **1772** | gpt-4o-mini, 2 questions générées par chunk indexé | ⭐⭐ qualité variable |
| **Total** | **1796** | | |

Le **split train/val est stratifié par question** (pas par paire) — ainsi les paragraphes
d'une même question ne se retrouvent jamais à cheval entre train et val.

### Étape 1 — Générer les questions synthétiques

```powershell
cd backend
.\.venv\Scripts\python.exe synth_questions.py `
  --output ..\Experimental\DATA\training\synthetic_questions.json `
  --questions-per-chunk 2
```

**Mécanisme** :
- Lit les chunks de la table `chunks` (filtrés à >= 80 chars)
- Pour chaque chunk, appelle gpt-4o-mini en mode JSON forcé avec un prompt qui demande
  N questions naturelles auxquelles le chunk répond
- Écrit chaque résultat dans le fichier de sortie avec traçabilité
  (`_chunk_id`, `_generator_model`, `_generated_at`)

**Idempotence** : avant chaque génération, le script charge le fichier existant et
**saute les chunks déjà couverts**. Tu peux relancer après avoir ingéré de nouveaux PDFs
sans re-payer les anciens chunks.

**Coût** : ~$0,10 pour 900 chunks (gpt-4o-mini). Durée : ~15-30 min selon rate limit.

### Étape 2 — Paraphraser les questions gold (recommandé)

Pour éviter la **mémorisation lexicale** quand on upweight les paires gold (étape 3 ci-dessous),
on diversifie le wording des questions ground-truth via paraphrasing LLM :

```powershell
.\.venv\Scripts\python.exe paraphrase_gold.py `
  --paraphrases-per-question 10
```

**Mécanisme** :
- Charge les paires gold (sample_queries.json + extra_training_examples.json)
- Pour chaque question unique, demande 10 paraphrases à gpt-4o-mini (temperature 0.9 pour diversité)
- Préserve les entités nommées (USV, MQ-9 Reaper, OSINT...)
- Émet 1 entry par (paraphrase, paragraphe) → ~240 entries pour 5 questions gold × 10 paraphrases

**Coût** : ~$0,001 par run complet (négligeable). Durée : ~1 min.

Le résultat va dans `Experimental/DATA/training/gold_paraphrases.json`, format prêt pour `--gold-extra`.

### Étape 3 — Lancer le fine-tuning

**Commande complète avec toutes les protections** (paraphrases + upweight modéré + dropout) :

```powershell
.\.venv\Scripts\python.exe finetune.py `
  --gold ..\Experimental\DATA\training\sample_queries.json `
  --gold-extra ..\Experimental\DATA\training\extra_training_examples.json `
               ..\Experimental\DATA\training\gold_paraphrases.json `
  --extra-examples ..\Experimental\DATA\training\synthetic_questions.json `
  --output-dir models\e5-base-monamo-ft `
  --epochs 3 --batch-size 8 --lr 2e-5 --val-ratio 0.2 `
  --gold-upweight 5 --question-dropout-gold 0.10
```

**Pourquoi cette combinaison de flags ?**

Le déséquilibre brut est de 264 paires gold contre 1772 synthétiques (~1:7). Sans
rééquilibrage, le modèle s'oriente vers le style synthétique uniforme et oublie le
signal des vraies questions du challenge. Trois mécanismes empilés :

| Mécanisme | Effet |
|---|---|
| **`paraphrase_gold.py`** (étape 2) | Démultiplie le ground truth en 10 reformulations → 264 paires « gold-like » au lieu de 24 |
| **`--gold-upweight 5`** | Duplique × 5 les paires gold dans le **train** (split val préservé pour mesure honnête) → 1110 gold + 1416 synth = ratio ~1:1.3 |
| **`--question-dropout-gold 0.10`** | À chaque epoch, droppe 10 % des mots des questions gold aléatoirement → empêche la mémorisation lexicale des paraphrases |

**Hyperparams clés** :

| Param | Défaut | Rôle |
|---|---|---|
| `--epochs 3` | 3 passages complets sur le training set | Au-delà, risque d'overfitting sur ~2000 paires |
| `--batch-size 8` | 8 paires par batch (CPU-friendly) | GPU : monter à 32-64 |
| `--lr 2e-5` | Learning rate AdamW | Standard pour fine-tuning SentenceTransformers |
| `--temperature 0.07` | Softmax sharpness InfoNCE | Plus bas = pénalise davantage les négatifs |
| `--val-ratio 0.2` | 20 % des questions en validation | Détecter l'overfitting (val_loss qui remonte) |
| `--seed 42` | Reproductibilité complète (CPU + CUDA + cudnn deterministic) | |
| `--gold-upweight 5` | Duplication des paires gold uniquement dans le train | Reco : 5 si paraphrases activées, 20-30 sans paraphrases |
| `--question-dropout-gold 0.10` | Word-dropout aléatoire sur questions gold | 0 = off ; reco : 0.10-0.15 avec upweight ≥ 5 |

**Optimisations automatiques** côté code :
- `gradient_checkpointing_enable()` → ~30 % de VRAM en moins
- `torch.amp.autocast(fp16)` + `GradScaler` si CUDA détecté → ~2x plus rapide
- Si CPU : tout en fp32, plus lent mais fonctionnel

**Ce que tu vois pendant le run** :
```
[Finetune] device=cuda | model=e5-base | epochs=3 bs=32 lr=2e-5
[Finetune] GPU=RTX 3060  VRAM=12.0 GB
[Finetune] Gradient checkpointing : activé
[Finetune] AMP=fp16 cuda
[Finetune] ep1 step  10 | loss=1.0241 | vram=4.2 GB
[Finetune] ep1 step  20 | loss=0.8137 | vram=4.3 GB
...
[Finetune] === epoch 1/3 : train_loss=0.7423 | val_loss=0.6891 ✓ best
[Finetune] === epoch 2/3 : train_loss=0.4521 | val_loss=0.5234 ✓ best
[Finetune] === epoch 3/3 : train_loss=0.3102 | val_loss=0.5891          ← overfitting commence
[Finetune] Modèle sauvegardé : models/e5-base-monamo-ft
```

Le meilleur checkpoint (val_loss minimale) est sauvé dans `models/e5-base-monamo-ft.best/`.
Le final est dans `models/e5-base-monamo-ft/`. **Prends le `.best` pour la suite.**

**Durées indicatives** pour 1796 paires × 3 epochs :

| Setup | Durée |
|---|---|
| CPU (16 GB RAM, batch 8) | ~4-5 h |
| GPU RTX 3060 (12 GB, batch 32) | ~15-20 min |
| GPU A10 / A100 (batch 64) | ~5-10 min |

### Étape 4 — Évaluer le gain

1. **Re-pointer la config** vers le modèle fine-tuné. Dans `backend/.env` :
   ```
   EMBEDDING_MODEL=models/e5-base-monamo-ft.best
   ```
2. **Re-générer les embeddings** du corpus avec ce nouveau modèle :
   ```powershell
   .\.venv\Scripts\python.exe ingest.py --force-reingest
   ```
3. **Comparer** les métriques avant/après sur le sample :
   ```powershell
   .\.venv\Scripts\python.exe run_challenge.py `
     --input ..\Experimental\DATA\training\sample_queries.json `
     --out-task1 ..\Experimental\DATA\runs\sample_task1_ft.json
   ```
   Comparer manuellement les `retrieved` du JSON produit aux gold de sample_queries.

**Gain attendu** sur ce volume modeste de données :
- hits@5 : de 50 % → 60-75 %
- hits@10 : de 63 % → 75-90 %

Les gains varient beaucoup selon la diversité du corpus. Plus de paires (ingestion complète
des 374 PDFs + génération synthétique élargie) = signal plus fort.

### Passation à un collègue GPU

Si tu n'as pas de GPU local, tu peux préparer un **JSON autonome de paires
d'entraînement** (sans dépendance à la DB Postgres) :

```powershell
.\.venv\Scripts\python.exe finetune.py `
  --gold ..\Experimental\DATA\training\sample_queries.json `
  --extra-examples ..\Experimental\DATA\training\extra_training_examples.json `
                   ..\Experimental\DATA\training\synthetic_questions.json `
  --export-to ..\Experimental\DATA\training\training_pairs_full.json
```

Ton collègue n'a alors besoin **que de 2 fichiers** :
- `training_pairs_full.json` (1796 paires, ~2,7 MB)
- `backend/finetune.py` + `backend/pipeline/finetune.py` + `backend/config.py`

Puis sur sa machine GPU :
```bash
python finetune.py --extra-examples training_pairs_full.json \
  --output-dir models/e5-base-monamo-ft --epochs 3 --batch-size 32
```

Documentation détaillée pour le collègue : [backend/HANDOFF_FINETUNE.md](backend/HANDOFF_FINETUNE.md).

Une fois le modèle reçu, copier le dossier dans `backend/models/` et reprendre à l'étape 3 ci-dessus.

---

## Structure du backend

```
backend/
├── alembic/                      # migrations DB
├── api/
│   ├── app.py                    # FastAPI : /health, /challenge/run
│   └── challenge.py              # ChallengeRunner + schémas Pydantic
├── database/
│   ├── db.py                     # session SQLAlchemy
│   └── models.py                 # Document / DocumentPage / Chunk
├── pipeline/
│   ├── pdf_extractor.py          # pdfplumber + OCR fallback
│   ├── chunker.py                # split paragraphes
│   ├── embedder.py               # SentenceTransformer + préfixes e5
│   ├── ingestion.py              # orchestration ingestion
│   ├── synth_questions.py        # génération synthétique questions
│   └── finetune.py               # training contrastif e5
├── rag/
│   ├── rag_engine.py             # retrieval + génération LLM
│   └── attribution.py            # phrase → (doc, page)
├── config.py                     # settings Pydantic
├── ingest.py                     # CLI ingestion
├── run_challenge.py              # CLI Tâche 1 + Tâche 2 (JSON challenge)
├── synth_questions.py            # CLI génération synthétique
├── finetune.py                   # CLI fine-tuning
└── HANDOFF_FINETUNE.md           # doc pour collègue GPU
```

---

## État actuel & baseline

**Ingéré en DB (`am-rag`) :** 8 PDFs / 74 pages / 901 chunks (sous-ensemble des sample_queries)

**Baseline retrieval** (modèle e5-base zero-shot, sample_queries Q1-Q4) :
- hits@5 : 4/8 (50%)
- hits@10 : 5/8 (63%)

**Réponses LLM** : gpt-4o-mini, ~3500-4500 tokens/question, citations `[doc.pdf p.N]` correctement insérées.

**Attribution Tâche 2** : segmentation FR + parsing citations + fallback embedding similarity, marque correctement les titres markdown comme `[]`.

**Training set préparé pour fine-tuning** : 1796 paires (`training_pairs_full.json`)
- 23 gold (sample_queries Q1-Q4)
- 1 OSINT (target.txt)
- 1772 synthétiques (gpt-4o-mini)

---

## Format challenge

### Entrée (questions seules)
```json
{
  "run_id": "monamo-cost-evallm-2026",
  "parameters": {},
  "results": [
    {"qid": "Q1", "question": "Quel est l'objectif du projet Beehive ?"}
  ]
}
```

### Sortie Tâche 1 (retrieval + génération)
```json
{
  "run_id": "...",
  "parameters": {"embedding_model": "...", "llm_model": "gpt-4o-mini", ...},
  "results": [
    {
      "qid": "Q1",
      "question": "...",
      "retrieved": [
        {"rank": 1, "doc_name": "20260108_NP_Obsdrones....pdf", "page": 17, "metadata": {}}
      ],
      "answer": "Le projet Beehive vise à... [20260108_NP_Obsdrones....pdf p.17]",
      "metadata": {"tokens_used": 4383}
    }
  ]
}
```

### Sortie Tâche 2 (attribution phrase)
```json
{
  "run_id": "...",
  "results": [
    {
      "qid": "Q1",
      "attributions": [
        {"sid": "Q1_s0", "text": "# Titre markdown", "attributed_to": []},
        {"sid": "Q1_s1", "text": "Le projet vise à...", "attributed_to": [
          {"doc_name": "...", "page": 17}
        ]}
      ]
    }
  ]
}
```

---

## Liens utiles

- [Challenge EvalLLM 2026](https://evalllm2026.sciencesconf.org/) — site officiel
- [Experimental/target.txt](Experimental/target.txt) — règlement complet et exemples
- [backend/HANDOFF_FINETUNE.md](backend/HANDOFF_FINETUNE.md) — passation GPU
- [Experimental/Exploration_données.ipynb](Experimental/Exploration_données.ipynb) — notebook d'exploration initiale
