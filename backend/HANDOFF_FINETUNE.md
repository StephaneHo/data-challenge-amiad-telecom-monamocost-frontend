# Fine-tuning de l'embedder e5-base — handoff GPU

Document à fournir au collègue qui dispose d'un GPU pour exécuter le fine-tuning.

---

## Objectif

Fine-tuner `intfloat/multilingual-e5-base` (768d) sur ~1800 paires `(question, paragraphe)`
extraites du corpus PDFs FR du challenge **EvalLLM 2026**, avec une loss contrastive
multi-positifs symétrique (InfoNCE).

Le modèle entraîné servira d'embedder de retrieval dans la pipeline RAG du projet.

---

## Fichiers à transférer

| Fichier | Rôle |
|---|---|
| `backend/finetune.py` | CLI |
| `backend/pipeline/finetune.py` | Logique training (dataset, loss, boucle) |
| `backend/pipeline/__init__.py` | Init module |
| `backend/config.py` | Settings Pydantic (lit `EMBEDDING_MODEL`, etc.) |
| `backend/.env.example` (à créer si manquant) | Variables d'env minimales |
| `Experimental/DATA/training_pairs_full.json` | **1796 paires d'entraînement (autonome, pas de DB)** |

> 💡 Le fichier `training_pairs_full.json` ne nécessite **pas** Postgres ni de corpus PDF —
> il contient déjà les paires `(question, paragraph, doc_name, page)` prêtes à l'emploi.

---

## Setup environnement (GPU)

```bash
# Python 3.11
python -m venv .venv
source .venv/bin/activate  # ou .venv\Scripts\Activate.ps1 sur Windows

# Dépendances minimales pour le fine-tuning
pip install \
  "torch>=2.1" \
  "sentence-transformers>=5.3" \
  "transformers>=4.41" \
  "loguru>=0.7" \
  "pydantic-settings>=2.13" \
  "sqlalchemy>=2.0"  # nécessaire pour l'import config.py (mais pas de connexion)
```

Variables d'env (créer `.env` minimal) :
```
EMBEDDING_MODEL=intfloat/multilingual-e5-base
EMBEDDING_DIM=768
DATABASE_URL=sqlite:///dummy.db   # placeholder, jamais utilisé en mode --extra-examples
```

---

## Commande de fine-tuning sur GPU

```bash
cd backend
python finetune.py \
  --extra-examples ../Experimental/DATA/training_pairs_full.json \
  --output-dir models/e5-base-monamo-ft \
  --model intfloat/multilingual-e5-base \
  --epochs 3 \
  --batch-size 32 \
  --lr 2e-5 \
  --temperature 0.07 \
  --val-ratio 0.2 \
  --log-every 10
```

### Hyperparams recommandés (GPU)

| Param | Valeur | Justification |
|---|---|---|
| `--batch-size 32` | Plus de positifs partagés dans le batch = meilleur signal contrastif |
| `--epochs 3` | Suffisant pour ~1800 paires sans overfitting |
| `--lr 2e-5` | Standard pour fine-tuning sentence-transformers |
| `--temperature 0.07` | Hyperparam classique InfoNCE |
| `--val-ratio 0.2` | 20% des questions en val (split stratifié par question) |

Pour aller plus large :
- Batch 64 si VRAM > 16 GB
- Epochs 5 avec early stopping sur `val_loss`
- `--model intfloat/multilingual-e5-large` pour passer en 1024d (nécessite re-migration côté projet — voir « Retour modèle »)

### Activations automatiques

Le code (`pipeline/finetune.py`) détecte CUDA et active :
- **Mixed precision fp16** (`torch.amp.autocast` + `GradScaler`)
- **Gradient checkpointing** (économise ~30% VRAM)

ETA approximatif sur GPU mid-range (RTX 3060 / A10) : **15-30 min** pour 3 epochs × 1800 paires.

---

## Sortie attendue

À la fin, dans `models/e5-base-monamo-ft/` :
- `config.json`, `model.safetensors`, `tokenizer.json`, `modules.json`, etc.
  → format SentenceTransformers standard, chargeable par
    `SentenceTransformer("models/e5-base-monamo-ft")`

Si `--val-ratio > 0`, un dossier `models/e5-base-monamo-ft.best/` contient
le meilleur checkpoint (val loss minimale).

---

## Retour modèle vers le projet

1. Copier le dossier `models/e5-base-monamo-ft.best/` (ou `.../e5-base-monamo-ft/` si pas de val)
   dans `backend/models/` du projet principal.

2. Côté projet, mettre à jour `backend/.env` :
   ```
   EMBEDDING_MODEL=models/e5-base-monamo-ft.best
   ```

3. Re-embedder le corpus indexé :
   ```powershell
   cd backend
   python ingest.py --force-reingest
   ```
   (pas de migration Alembic nécessaire, e5-base reste 768d)

4. Évaluer le gain :
   ```powershell
   python run_challenge.py --input ../Experimental/DATA/sample_queries.json --filter-qids Q1,Q2,Q3,Q4
   ```
   Comparer les `hits@5` / `hits@10` du JSON Tâche 1 produit avec ceux du baseline e5-base zero-shot.

---

## Format du training set (pour info)

`training_pairs_full.json` est une liste de paires plates :
```json
[
  {
    "question": "Quel est l'objectif du projet Beehive de la Royal Navy ?",
    "paragraph": "...",
    "doc_name": "20260108_NP_Obsdrones_Bulletin-de-veille-n12_0.pdf",
    "page": 17
  },
  ...
]
```

Composition (1796 paires totales) :
- **23 paires** issues du gold `sample_queries.json` (Q1-Q4, 8 docs)
- **1 paire** OSINT/SSL du target.txt (`guide_osint_infrastructure_v2.pdf`)
- **1772 paires synthétiques** générées par gpt-4o-mini sur les chunks ingérés (2 questions par chunk)

La clé positive pour la loss contrastive multi-positifs est `(doc_name, page)` —
toutes les questions pointant vers la même page sont considérées positives entre elles.

---

## Logs et debug

Le script logge via `loguru`. Pour rediriger dans un fichier :
```bash
python finetune.py ... 2>&1 | tee finetune.log
```

Métriques affichées par epoch :
- `train_loss` (moyenne sur l'epoch)
- `val_loss` (si `--val-ratio > 0`)
- VRAM consommée (si CUDA)

Une diminution de la val_loss epoch après epoch indique un bon entraînement.
Si val_loss augmente après l'epoch 1, c'est probablement de l'overfitting → réduire `--epochs` ou augmenter `--val-ratio`.
