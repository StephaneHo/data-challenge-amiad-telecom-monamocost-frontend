# Responsible NLP Research checklist — Mon Amo Cost

Réponses aux questions de la checklist
[ACL ARR Responsible NLP Research](https://aclrollingreview.org/responsibleNLPresearch/).

---

## A. Pour tous les papiers

### A1. Le résumé/intro identifie-t-il clairement les contributions ?

Oui : pipeline RAG complète pour le challenge EvalLLM 2026 (FR défense), avec :
- Retrieval `(doc_name, page)` via e5-base fine-tuné
- Attribution phrase → source hybride (citations LLM + embedding similarity + filtre entité)
- Tracking complet de l'empreinte carbone et reproductibilité chiffrée

### A2. Le résumé/intro pose-t-il des claims qui dépassent les résultats ?

Non. Les résultats baseline (hits@10 = 63% sur 4 questions) sont **explicitement annoncés
comme indicatifs et non significatifs** (cf. MODEL_CARD.md, section « Évaluation »).

### A3. Limites discutées ?

Oui — voir MODEL_CARD.md section « Limites » :
- Volume d'entraînement faible (2036 paires)
- Biais possibles des données synthétiques (gpt-4o-mini)
- Domaine étroit (défense FR)
- Multi-hop questions mal gérées (cf. Q3 du sample)
- OCR imparfait sur PDFs scannés

### A4. Considérations éthiques ?

Oui — voir MODEL_CARD.md section « Considérations éthiques ». Notamment :
- Le modèle ne doit pas être utilisé pour produire des contenus militaires offensifs
- Dépendance à gpt-4o-mini (modèle propriétaire) à terme à remplacer par open-source
- Biais possibles encodés via les questions synthétiques

---

## B. Si vous utilisez des données existantes ou des modèles existants

### B1. Citez-vous les créateurs ?

Oui :
- Corpus PDFs : organisateurs du challenge EvalLLM 2026 (à citer dans le rapport final)
- `intfloat/multilingual-e5-base` (Wang et al. 2024)
- `gpt-4o-mini` (OpenAI, 2024-2026)
- Loss multi-positifs InfoNCE : trame du notebook d'exploration (cf. `Experimental/Exploration_données.ipynb`)

### B2. Mentionnez-vous la licence ?

- Code : MIT (ce repo)
- e5-base : MIT (HuggingFace)
- gpt-4o-mini : Terms of Service OpenAI
- Corpus PDFs : ⚠️ propriété de leurs auteurs respectifs — **non redistribués** (cf. `.gitignore`)

### B3. Respectez-vous l'utilisation prévue par les créateurs ?

Oui :
- Corpus utilisé strictement dans le cadre prévu (challenge EvalLLM 2026)
- e5-base utilisé en mode retrieval, son cas d'usage principal
- Pas de scraping ou de re-distribution interdite

### B4. Discutez-vous des informations personnelles dans le dataset ?

Le corpus mentionne des noms publics (sénateurs, responsables défense) liés à leur fonction.
Pas de données médicales, financières privées, ou autres données sensibles.

### B5. Discutez-vous des contenus offensifs ?

Le corpus traite de **drones armés, cyberattaques, renseignement**. Les contenus restent
factuels et institutionnels (rapports parlementaires, communications du Ministère des Armées).
Pas de contenu haineux, discriminatoire ou explicitement offensant.

---

## C. Si vous avez fait tourner des expériences

### C1. Les ressources de calcul sont-elles mentionnées ?

Oui :
- **Ingestion PDF + embedding** : CPU, ~10 min pour 8 PDFs / 886 chunks
- **Fine-tuning** : prévu GPU (RTX 3060 ou similaire), ~15-30 min sur 2526 paires × 3 epochs
- **Inférence retrieval + génération** : CPU, ~10-30s par requête (gpt-4o-mini via API)
- **Empreinte carbone tracée** par run via `backend/utils/carbon.py` (formule Green Algorithms)

### C2. Les hyperparamètres sont-ils listés ?

Oui :
- Tous dans `backend/HANDOFF_FINETUNE.md` (training)
- Tous dans `backend/api/challenge.py` `_parameters()` (inférence)
- Inclus automatiquement dans le JSON de sortie (`run.parameters`) à chaque appel

Liste exhaustive :
| Étage | Hyperparams |
|---|---|
| Embedder | `intfloat/multilingual-e5-base`, 768d, préfixes `query:`/`passage:` |
| Fine-tuning | `lr=2e-5`, `batch_size=32` (GPU), `epochs=3`, `temperature=0.07`, `seed=42`, `gold_upweight=5`, `question_dropout_gold=0.10` |
| Retrieval | `top_k_chunks=20`, `top_n_pages=10`, `context_chunks=10` |
| LLM | `gpt-4o-mini`, `temperature=0.2` |
| Attribution | `threshold=0.80`, `secondary_threshold=0.85`, `topk_per_sentence=3`, `entity_filter=True` |

### C3. Mesures d'erreur / variance ?

À effectuer après le run GPU : 3 runs avec seeds différents pour estimer la variance des
hits@K. Pour l'instant, baseline n'a tourné qu'avec seed=42.

### C4. Sélection d'hyperparams documentée ?

- Seuils d'attribution : grid search via `backend/calibrate_attribution.py` sur sample_queries
  (sortie : `Experimental/DATA/runs/attribution_calibration.json`)
- Autres hyperparams : valeurs par défaut conservatrices, à ajuster après mesures GPU

### C5. Implémentation disponible publiquement ?

Oui, ce repo (lien à ajouter dans le rapport final). Pas de dépendances exotiques.

---

## D. Si vous avez utilisé des données / artefacts générés par humains

Non concerné directement. Les questions synthétiques sont **générées par LLM** (gpt-4o-mini),
pas par des humains.

Pour les annotations gold du `sample_queries.json` (créées par les organisateurs du challenge) :
ces annotations ne nous appartiennent pas et sont utilisées telles que reçues.

---

## E. Empreinte carbone (bonus EvalLLM 2026)

Méthodologie : formule [Green Algorithms](http://calculator.green-algorithms.org/) — Lannelongue
et al. 2021. Voir `backend/utils/carbon.py` pour l'implémentation exacte.

Constantes :
- Intensité carbone France 2024 : `0.052 kgCO2eq/kWh` (RTE)
- PUE data center : `1.5`
- Tokens LLM : `0.04 g CO2eq / 1000 tokens output` (estimation publique gpt-4o-mini)

Empreinte totale du projet (à finaliser après run GPU) :

| Étape | Compute | LLM tokens | CO2 (g) |
|---|---|---|---|
| Ingestion PDF (8 docs) | ~10 min CPU | 0 | ~0.6 |
| Génération synthétique | ~31 min CPU | ~500K | ~10 |
| Paraphrasing gold | ~1 min CPU | ~12K | ~0.5 |
| Fine-tuning (GPU) | ~20 min RTX 3060 | 0 | ~6 |
| Inférence par requête | ~10s CPU | ~5K | ~0.3 |

**Total estimatif pour reproduire ce projet entier : ~20-30 g CO2eq** (équivalent à un trajet
en voiture de ~150-200 m). À comparer avec un fine-tuning d'un modèle 7B sur GPU = ~10-100 kg.

Les valeurs réelles seront mises à jour dans `models/<run>/carbon_metrics.json` après chaque
run, et dans le rapport final.

---

## Reproductibilité

Tout est en place pour qu'un tiers reproduise les expériences :

1. Cloner le repo
2. Installer les deps (`uv sync --group experimental`)
3. Démarrer Postgres (`docker compose up -d`)
4. Appliquer les migrations (`alembic upgrade head`)
5. Ingestion du corpus (`python ingest.py`)
6. (Optionnel) Génération synthétique + paraphrases (`python synth_questions.py` + `python paraphrase_gold.py`)
7. Fine-tuning (`python finetune.py ...` ou utiliser le JSON exporté sur GPU séparé)
8. Évaluation (`python run_challenge.py --input <gold>`)

Seeds fixes (42 par défaut), `cudnn.deterministic=True` pour GPU, données complètes incluses
dans le repo. Le modèle fine-tuné sera publié sur HuggingFace Hub après le challenge.
