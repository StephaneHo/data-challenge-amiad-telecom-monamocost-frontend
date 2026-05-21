# Model Card — Mon Amo Cost RAG (EvalLLM 2026)

Suit les recommandations [Mitchell et al. 2019](https://arxiv.org/abs/1810.03993) et la
[checklist ACL Responsible NLP Research](https://aclrollingreview.org/responsibleNLPresearch/).

## Détails du modèle

- **Nom** : `monamocost/e5-base-monamo-ft` (ou `intfloat/multilingual-e5-base` en baseline zero-shot)
- **Type** : Sentence-Transformer (encodeur bi-directionnel XLM-RoBERTa) fine-tuné par loss
  contrastive multi-positifs (InfoNCE symétrique)
- **Backbone** : `intfloat/multilingual-e5-base` (278M paramètres, 768 dimensions)
- **Conventions de préfixes** : `query: ` pour les requêtes, `passage: ` pour les documents
- **Licence** : MIT (même licence que le backbone e5)
- **Version** : 0.1.0 (mai 2026)
- **Code source** : ce repo (voir `README.md`)

## Cas d'usage prévus

- **Tâche principale** : retrieval dense `(doc_name, page)` sur corpus FR défense/renseignement
  dans le cadre du challenge EvalLLM 2026
- **Tâche secondaire** : attribution phrase → source documentaire (via embedding similarity)
- **Hors scope** : génération de texte, classification, reranking exhaustif

## Hors-cas d'usage

- ❌ Détection de documents hors-corpus (le retriever n'a aucune notion d'OOD)
- ❌ Décisions automatisées impactant des personnes (le modèle est un outil d'aide à
  l'analyste, pas un système de décision)
- ❌ Utilisation sans supervision humaine sur du contenu sensible
- ❌ Évaluation de la véracité factuelle (la similarité sémantique n'est pas une vérité)

## Données d'entraînement

Voir [DATASHEET.md](DATASHEET.md) pour les détails complets. Résumé :

- **Source ground truth** : 4 questions gold du `sample_queries.json` officiel du challenge
  (chacune avec 1-3 pages annotées dans le corpus FR)
- **Source augmentée** : 240 paraphrases LLM de ces questions (10 par question, via
  `gpt-4o-mini`, temperature 0.9)
- **Source synthétique** : 1772 paires `(question, paragraphe)` générées par `gpt-4o-mini` sur
  les 886 chunks indexés du corpus (2 questions par chunk)
- **Total** : 2036 paires d'entraînement, dont 264 marquées `_is_gold: true`

## Procédure d'entraînement

- **Loss** : InfoNCE multi-positifs symétrique, clé positive = `(doc_name, page)`
- **Optimiseur** : AdamW, lr=2e-5, weight_decay=0.01
- **Batch size** : 32 (GPU) / 8 (CPU)
- **Epochs** : 3
- **Temperature** : 0.07
- **Gradient checkpointing** : activé (économise ~30% VRAM)
- **Mixed precision** : fp16 si CUDA
- **Reproductibilité** : seed=42, `cudnn.deterministic=True`
- **Gold upweight** : ×5 dans le train uniquement (val préservée pour mesure honnête)
- **Word dropout sur questions gold** : 10% (chaque epoch = un dropout différent)
- **Split** : 80/20 stratifié par question (pas de leakage)

## Évaluation

### Métriques visées (challenge)

**Tâche 1 — retrieval `(doc_name, page)`** :
- Précision @ K, Rappel @ K, NDCG @ K

**Tâche 2 — attribution phrase → source** :
- Précision, Rappel, F1 sur les attributions `attributed_to`
- Précision, Rappel sur les segments `attributed_to: []` (détection hallucinations)

### Résultats baseline (e5-base zero-shot, sample_queries Q1-Q4)

| Métrique | Valeur |
|---|---|
| hits@5 (proxy doc-page) | 4/8 (50%) |
| hits@10 (proxy doc-page) | 5/8 (63%) |

Résultats sur 4 questions seulement → indicatif, pas significatif.

### Résultats après fine-tuning

À renseigner après le run GPU.

## Limites

- **Volume d'entraînement faible** (2036 paires) → risque de variance importante entre seeds
- **Données synthétiques majoritaires** (1772 / 2036) → biais de style gpt-4o-mini possible
- **Domaine étroit** (défense FR) → modèle peu généralisable à d'autres domaines
- **Multi-hop questions** : retrieval dense seul peut rater (cf. Q3 du sample) ; pas
  d'amélioration spécifique implémentée (BM25 hybride, reranking → pistes futures)
- **OCR imparfait** : certaines pages issues de PDFs scannés ont du texte bruité, ce qui
  dégrade les embeddings ; impact non quantifié
- **Détection des hallucinations** : le filtre entités est basé sur regex (acronymes, nombres,
  noms propres) — pas un NER complet. Faux positifs/négatifs possibles.

## Considérations éthiques

- **Corpus sensible** : documents publics de défense/renseignement, mais traitant de sujets
  potentiellement controversés (cyber-armement, renseignement étranger, drones armés).
  Le modèle ne doit pas être utilisé pour produire des contenus offensifs ou militaires.
- **Sources LLM externes** : l'utilisation de gpt-4o-mini (modèle propriétaire) introduit une
  dépendance + un coût récurrent. Migration vers un LLM open-source (Mistral, Qwen FR) à
  envisager pour la version finale.
- **Données générées par LLM** : les questions synthétiques peuvent encoder des biais du
  modèle générateur. La traçabilité est assurée (`_generator_model` dans le JSON).
- **Empreinte carbone** : voir section dédiée + `models/.../carbon_metrics.json` par run.

## Empreinte environnementale

Mesurée via une approche conservatrice inspirée de
[Green Algorithms](http://calculator.green-algorithms.org/) (Lannelongue et al. 2021).

Constantes utilisées :
- Carbon intensity France 2024 : 0.052 kg CO2eq / kWh (RTE)
- PUE typique data center : 1.5
- Puissances CPU/GPU : selon le hardware utilisé pour chaque run

**Empreinte mesurée** (à remplir après run GPU) :
- Génération synthétique : ~$0.10 + ~10g CO2 estimés (gpt-4o-mini, US)
- Paraphrasing gold : ~$0.001 + ~0.1g CO2 estimés
- Fine-tuning : à mesurer sur GPU
- Inférence par requête : à mesurer

Détails par run dans `models/<run>/carbon_metrics.json` et dans le champ `parameters.carbon`
du JSON de sortie challenge.

## Maintenance et reproductibilité

- **Seed unique** : `42` (modifié dans `config.py` si besoin)
- **Fichiers à versionner** pour reproduction complète :
  - Code : ce repo
  - Données entraînement : `Experimental/DATA/training/training_pairs_full.json`
  - Questions synthétiques : `Experimental/DATA/training/synthetic_questions.json`
  - Paraphrases gold : `Experimental/DATA/training/gold_paraphrases.json`
  - Modèle entraîné : à publier sur HuggingFace Hub (`monamocost/e5-base-monamo-ft`)
- **Hyperparams** : voir `backend/HANDOFF_FINETUNE.md`

## Contact

Projet académique pour le challenge EvalLLM 2026 — TALN/CORIA.
Toute question / issue : voir le repo.
