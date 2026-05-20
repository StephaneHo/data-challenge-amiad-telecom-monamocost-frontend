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
  --gold-upweight 5 \
  --question-dropout-gold 0.10 \
  --log-every 10
```

### Hyperparams recommandés (GPU)

| Param | Valeur | Justification |
|---|---|---|
| `--batch-size 32` | Plus de positifs partagés dans le batch = meilleur signal contrastif |
| `--epochs 3` | Suffisant pour ~2000 paires sans overfitting |
| `--lr 2e-5` | Standard pour fine-tuning sentence-transformers |
| `--temperature 0.07` | Hyperparam classique InfoNCE |
| `--val-ratio 0.2` | 20% des questions en val (split stratifié par question) |
| `--gold-upweight 5` | Compense le 1:7 gold/synth sans noyer le ground-truth. La duplication s'applique **uniquement au train** (val préservée) |
| `--question-dropout-gold 0.10` | Word-dropout aléatoire sur questions gold (différent par epoch) → casse la mémorisation lexicale liée à l'upweight |

> ℹ️ Le fichier `training_pairs_full.json` contient un champ `_is_gold` par entry, conservé
> automatiquement à l'export. `load_extra_examples` le respecte → `--gold-upweight` cible
> les bonnes paires (264 gold sur 2036 totales).

Pour aller plus large :
- Batch 64 si VRAM > 16 GB
- Epochs 5 avec early stopping sur `val_loss`
- `--model intfloat/multilingual-e5-large` pour passer en 1024d (nécessite re-migration côté projet — voir « Retour modèle »)

### Activations automatiques

Le code (`pipeline/finetune.py`) détecte CUDA et active :
- **Mixed precision fp16** (`torch.amp.autocast` + `GradScaler`)
- **Gradient checkpointing** (économise ~30% VRAM)
- **Full determinism CUDA** (`cuda.manual_seed_all` + `cudnn.deterministic`) → 2 runs avec même seed = mêmes résultats au bit près, au prix de ~5-15 % de vitesse. Permet de comparer des hyperparams entre eux.

ETA approximatif sur GPU mid-range (RTX 3060 / A10) : **15-30 min** pour 3 epochs × 2500 paires (après upweight).

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

Composition (**2036 paires totales** — incluant les paraphrases LLM des gold) :
- **23 paires** issues du gold `sample_queries.json` (Q1-Q4, 8 docs)  → `_is_gold: true`
- **1 paire** OSINT/SSL du target.txt (`guide_osint_infrastructure_v2.pdf`) → `_is_gold: true`
- **240 paires** paraphrases LLM des 5 questions gold ci-dessus (10 paraphrases × ~5 questions × ~5 paragraphes en moyenne) → `_is_gold: true`
- **1772 paires synthétiques** générées par gpt-4o-mini sur les chunks ingérés → `_is_gold: false`

**Total `_is_gold: true` = 264 paires** ; **synthétique = 1772**.

La clé positive pour la loss contrastive multi-positifs est `(doc_name, page)` —
toutes les questions pointant vers la même page sont considérées positives entre elles.

> 🔑 Les paraphrases sont **essentielles** : sans elles, dupliquer 30x les 24 paires gold
> originales ferait simplement *mémoriser* le wording, sans généraliser. Avec les paraphrases
> + word dropout, le modèle voit ~250 formulations différentes pour les mêmes concepts.

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
