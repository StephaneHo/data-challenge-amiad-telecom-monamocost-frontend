# Mon Amo Cost @ EvalLLM 2026 — Une pipeline RAG frugale et reproductible pour la défense

**Authors** : `<TODO: noms, prénoms>`
**Affiliations** : `<TODO: institutions>`
**Contacts** : `<TODO: emails>`

> Cet article est une **soumission au Challenge RAG EvalLLM 2026** (atelier conjoint TALN/CORIA, Nantes, 29 juin 2026).
> Code : `<TODO: URL GitHub>` — Modèle fine-tuné : `<TODO: URL HuggingFace>`

---

## Abstract

`<TODO: à écrire en dernier ; ~150 mots>`

Brouillon :
> Nous présentons **Mon Amo Cost**, une pipeline RAG en français pour les deux tâches du challenge EvalLLM 2026 : (i) recherche documentaire + génération de réponse, et (ii) attribution phrase-par-phrase aux sources documentaires. Notre approche combine un embedder bi-directionnel multilingue (`intfloat/multilingual-e5-base`, 768d) fine-tuné par loss contrastive multi-positifs, un LLM léger (`gpt-4o-mini`) pour la génération, et un module d'attribution hybride (parsing de citations + similarité embedding + filtre par entités nommées). L'empreinte carbone totale est mesurée à `<TODO: X g CO2eq>` pour le pipeline complet. L'ensemble du code, des données d'entraînement synthétiques et du modèle ajusté sont publiés en open source. Sur les questions de calibration, nous obtenons NDCG@10 = `<TODO>`, MAP = `<TODO>`, F1 attribution = `<TODO>` et F1 détection `[]` = `<TODO>`.

---

## 1. Introduction

Le **Retrieval-Augmented Generation (RAG)** est devenu la pratique de référence pour ancrer les réponses des modèles génératifs dans des sources documentaires. Le challenge **EvalLLM 2026** propose deux tâches complémentaires sur un corpus en français du domaine défense/renseignement :

- **Tâche 1** : étant donnée une requête, retrouver les pages pertinentes `(doc_name, page_number)` du corpus, puis produire une réponse.
- **Tâche 2** : étant donnée une réponse, attribuer chaque phrase à sa (ses) source(s) ou la marquer non sourcée (`attributed_to: []`) si elle relève d'une hallucination, d'une connaissance générale, ou d'un élément de mise en forme.

Notre contribution articule trois axes :
1. Une **pipeline RAG modulaire** qui sépare cleanly retrieval, génération et attribution, conforme aux formats JSON du challenge.
2. Une **stratégie de fine-tuning frugal** combinant données gold (3%), paraphrases LLM (12%) et questions synthétiques (85%) pour ~1800 paires d'entraînement totales sur un budget de ~0,1 USD.
3. Une **détection des passages non sourcés** multi-signal : regex markdown, filtre par entités nommées (acronymes, chiffres, noms propres absents des chunks), et seuil de similarité embedding.

Nos choix techniques privilégient explicitement la **reproductibilité** et la **frugalité énergétique** ciblées par les bonus du challenge.

---

## 2. Travaux connexes

`<TODO: 1-2 paragraphes courts>`

Brouillon de points à citer :
- **Retrieval dense bi-encodeur** : Karpukhin et al. 2020 (DPR), Wang et al. 2022/2024 (e5).
- **Loss contrastive multi-positifs** : Khosla et al. 2020 (SupCon), Su et al. 2023 (E5).
- **Attribution / source grounding** : Bohnet et al. 2022, Gao et al. 2023 (RARR), Liu et al. 2023 (Self-RAG).
- **Évaluation génération** : Zhang et al. 2020 (BertScore), Es et al. 2024 (RAGAs).
- **Frugalité ML** : Strubell et al. 2019, Lannelongue et al. 2021 (Green Algorithms).
- **Responsible NLP** : Mitchell et al. 2019 (Model Cards), Gebru et al. 2021 (Datasheets).

---

## 3. Architecture

```
                                     ┌──────────────────────────────────┐
                                     │  Corpus PDFs FR                  │
                                     │  (Défense / Renseignement)       │
                                     └──────────────┬───────────────────┘
                                                    │
                          ┌─────────────────────────▼────────────────────┐
                          │ INGESTION                                     │
                          │  pdfplumber + OCR Tesseract fallback (lang=fra│
                          │  chunking paragraphes (50-1500 chars)         │
                          │  Embeddings e5-base 768d (préfixe `passage:`) │
                          │  Stockage Postgres + pgvector + index HNSW    │
                          └─────────────────────────┬────────────────────┘
                                                    │
                                                    ▼
                          ┌──────────────────────────────────────────────┐
                          │ TÂCHE 1 — Retrieval + Génération              │
                          │  • dense ANN cosinus (top-K=20)               │
                          │  • [opt.] décomposition LLM multi-hop         │
                          │  • agrégation (doc_name, page) top-N=10       │
                          │  • LLM (gpt-4o-mini) avec few-shot OSINT      │
                          │  • réponse + citations [doc.pdf p.N]          │
                          └──────────────────────────┬───────────────────┘
                                                     │
                                                     ▼
                          ┌──────────────────────────────────────────────┐
                          │ TÂCHE 2 — Attribution phrase                  │
                          │  • Segmentation phrase (regex FR + abrév.)    │
                          │  1. Markdown pur → []                         │
                          │  2. Entités saillantes absentes → []          │
                          │  3. Parsing citations [doc.pdf p.N]           │
                          │  4. Top-K adaptatif embedding (seuils 0.80/0.85)│
                          │  5. Sinon → []                                │
                          └───────────────────────────────────────────────┘
```

### 3.1 Ingestion documentaire

L'ingestion (cf. [backend/pipeline/ingestion.py](backend/pipeline/ingestion.py)) :

1. Tente l'extraction texte via **`pdfplumber`** (rapide, PDFs natifs).
2. Pour les pages où l'extraction renvoie moins de 50 caractères (typiquement scans/figures), **fallback OCR Tesseract** (lang=fra, DPI=300) via `pdf2image` (Poppler).
3. **Chunking paragraphe** : segmentation sur double saut de ligne, filtre `[50, 1500]` caractères, re-split par phrase si trop long.
4. **Embedding** avec préfixe e5 `passage: ...`, vecteurs L2-normalisés.
5. Stockage Postgres + pgvector (extension `vector`), index HNSW pour ANN cosinus.

La page physique (couverture = `page=1`) est conservée conformément au règlement.

### 3.2 Retrieval (Tâche 1)

Le retrieval (cf. [backend/rag/rag_engine.py](backend/rag/rag_engine.py)) opère en 3 étapes :

1. **Embed requête** avec préfixe `query: ...` via la même `Embedder` que l'ingestion.
2. **Recherche cosinus** : `SELECT chunks ORDER BY embedding <=> :qv LIMIT k`.
3. **Agrégation** à la granularité `(doc_name, page)` en gardant le meilleur score par page.

Pour les questions multi-hop (cf. Q3 du sample : *« Comment les mesures anti-drones en France, les capacités d'Altius et Switchblade, et la stratégie chinoise illustrent-elles ensemble... »*), nous proposons en option une **décomposition par LLM** (cf. [backend/pipeline/query_decomposition.py](backend/pipeline/query_decomposition.py)) qui découpe la question en sous-questions atomiques (1-4) puis fusionne les chunks retrouvés.

### 3.3 Génération (Tâche 1)

Nous utilisons **`gpt-4o-mini`** comme générateur (paramètre `temperature=0.2` pour reproductibilité). Le prompt système enseigne :
- Citer chaque affirmation factuelle au format `[doc.pdf p.N]`.
- Ne **pas** citer les transitions, titres et énoncés de connaissance générale (pour faciliter la détection des `[]` en Tâche 2).
- Structurer avec des titres markdown si la réponse est longue.

Un **exemple OSINT du règlement** (analyse SSL/CT Logs) est inclus en few-shot pour calibrer le ton et la convention de citations (cf. `_SYSTEM_PROMPT` dans `rag_engine.py`).

### 3.4 Attribution phrase (Tâche 2)

L'attribution (cf. [backend/rag/attribution.py](backend/rag/attribution.py)) procède phrase par phrase :

1. **Segmentation FR** : split sur `.!?\s+[A-Z...]` avec masquage des abréviations courantes (`M.`, `etc.`, etc.).
2. **Markdown** : lignes purement formatage (titres `#`, séparateurs `---`) → `attributed_to: []`.
3. **Filtre par entités** : si une fraction (≥50%) des entités saillantes (acronymes capitalisés, chiffres, noms propres) de la phrase est absente du texte concaténé des chunks retrieved → `[]`.
4. **Parsing citations** : extraction des `[doc.pdf p.N]` ou `[doc.pdf]` (page résolue par sim cos dans le doc).
5. **Embedding similarity Top-K adaptatif** :
   - top-1 retenu si cosinus ≥ `threshold=0.80`,
   - top-2/3 ajoutés si cosinus ≥ `secondary_threshold=0.85` (plus strict), dédoublonnés par `(doc, page)`.
6. Sinon → `[]`.

Conformément au règlement, les chunks utilisés pour l'attribution sont **strictement ceux déclarés dans le `retrieved` de la Tâche 1** (filtrage explicite dans le runner).

---

## 4. Données

Cf. [DATASHEET.md](DATASHEET.md) pour le détail complet. Synthèse :

| Source | Paires | `_is_gold` | Origine |
|---|--:|---|---|
| `sample_queries.json` (gold du challenge) | 23 | ✓ | 4 questions × ~6 chunks par page annotée |
| `extra_training_examples.json` (OSINT du règlement) | 1 | ✓ | Inline dans `target.txt` |
| `gold_paraphrases.json` (paraphrases LLM des gold) | 240 | ✓ | 10 paraphrases × ~5 questions × ~5 paragraphes (gpt-4o-mini, T=0.9) |
| `synthetic_questions.json` (auto-générées) | 1772 | ✗ | 2 questions par chunk × 886 chunks indexés (gpt-4o-mini, T=0.7) |
| **Total** | **2036** | 264 gold / 1772 synth | |

Le coût total de génération synthétique + paraphrasing : **~0,10 USD** sur l'API OpenAI.

### Conformité du règlement

- ✅ Pas d'ajout de documents au corpus (les paraphrases et questions synthétiques sont des questions d'entraînement, pas des sources de retrieval).
- ✅ Pas de recherche web.
- ✅ Génération LLM uniquement à l'entraînement et à l'inférence sur le corpus indexé.

---

## 5. Modèle et entraînement

### 5.1 Fine-tuning

Cf. [MODEL_CARD.md](MODEL_CARD.md) et [backend/pipeline/finetune.py](backend/pipeline/finetune.py).

- **Backbone** : `intfloat/multilingual-e5-base` (278M paramètres, 768d).
- **Loss** : InfoNCE multi-positifs symétrique avec clé positive `(doc_name, page)`.
- **Optimiseur** : AdamW, `lr=2e-5`, `temperature=0.07`.
- **Batch** : 32 (GPU) / 8 (CPU).
- **Epochs** : 3.
- **Reproductibilité** : `seed=42`, `cudnn.deterministic=True`.
- **Gold upweight** : ×5 dans le train uniquement (val préservée).
- **Word dropout** sur questions gold : 10 % (chaque epoch = un dropout différent).
- **Split** : 80/20 stratifié par question (pas de leakage entre paragraphes d'une même question).

### 5.2 Justification des hyperparamètres

- Sur 1796 paires d'entraînement, une upweighting brute des 264 paires gold (par exemple ×30) entraînerait une **mémorisation lexicale** des 5 questions originelles. Nous combinons donc upweight modéré (×5) + diversification par paraphrasing LLM (10 paraphrases/question) + word dropout au runtime (10%) pour préserver le signal ground-truth sans encoder le surface du wording.
- La `temperature=0.07` est l'hyperparamètre classique de la loss InfoNCE (Chen et al. 2020). Une température plus basse pénalise davantage les négatifs.

---

## 6. Évaluation

### 6.1 Métriques

Conformément au règlement :

- **Retrieval (Tâche 1)** : Précision@K, Rappel@K, **NDCG@K**, MAP, MRR (cf. [backend/utils/metrics.py](backend/utils/metrics.py)).
- **Génération (Tâche 1)** : **BertScore** (lang=fr) + **LLM-as-Judge** sur 3 critères (factualité / complétude / clarté), notes 1-5 (cf. [backend/utils/judge.py](backend/utils/judge.py)).
- **Attribution (Tâche 2)** : Précision, Rappel, F1 sur les paires `(doc, page)` correctement attribuées.
- **Détection `[]`** (Tâche 2) : Précision, Rappel, F1 sur la classe « non sourcé » (cf. [backend/eval_attribution.py](backend/eval_attribution.py)).

### 6.2 Résultats baseline (zero-shot)

Sur le sample_queries.json (Q1-Q4, exclusion du placeholder Q5) :

| Métrique | Valeur |
|---|---|
| NDCG@10 | `<TODO: à mesurer>` |
| NDCG@5 | `<TODO>` |
| MAP | `<TODO>` |
| MRR | `<TODO>` |
| P@5 | `<TODO>` |
| R@5 | `<TODO>` |
| BertScore F1 (avg) | `<TODO>` |
| Judge factuality (avg /5) | `<TODO>` |
| Judge completeness (avg /5) | `<TODO>` |
| Judge clarity (avg /5) | `<TODO>` |
| Attribution F1 (proxy) | `<TODO>` |
| Empty detection F1 | `<TODO> (nécessite gold phrase-level)` |

**Reproduction** :
```bash
python run_challenge.py --input ../Experimental/DATA/sample_queries.json \
  --out-task1 sample_task1.json --out-task2 sample_task2.json
python eval_retrieval.py --predictions sample_task1.json --gold sample_queries.json
python eval_generation.py --predictions sample_task1.json --gold sample_queries.json
```

### 6.3 Ablations

Cf. [backend/ablation_study.py](backend/ablation_study.py).

| Config | NDCG@10 | MAP | MRR | P@5 | R@5 | Attr F1 | `[]` rate (md/ent/emb) |
|---|---|---|---|---|---|---|---|
| A0_baseline (no entity filter, top-1) | `<TODO>` | `<TODO>` | `<TODO>` | `<TODO>` | `<TODO>` | `<TODO>` | `<TODO>` |
| A1 + entity_filter | `<TODO>` | `<TODO>` | `<TODO>` | `<TODO>` | `<TODO>` | `<TODO>` | `<TODO>` |
| A2 + Top-K adaptatif | `<TODO>` | `<TODO>` | `<TODO>` | `<TODO>` | `<TODO>` | `<TODO>` | `<TODO>` |
| A3 + décomposition multi-hop | `<TODO>` | `<TODO>` | `<TODO>` | `<TODO>` | `<TODO>` | `<TODO>` | `<TODO>` |
| A4 = all | `<TODO>` | `<TODO>` | `<TODO>` | `<TODO>` | `<TODO>` | `<TODO>` | `<TODO>` |

**Lecture attendue** :
- Le filtre par entités améliore la détection des `[]` (rappel sur les hallucinations).
- Le Top-K adaptatif améliore le rappel d'attribution sur les phrases multi-sources (Q2-style, Q3-style).
- La décomposition multi-hop améliore NDCG@10 et MAP sur les questions complexes (Q3-style).

### 6.4 Impact du fine-tuning

| Modèle | NDCG@10 | MAP | BertScore F1 |
|---|---|---|---|
| e5-base zero-shot | `<TODO>` | `<TODO>` | `<TODO>` |
| e5-base fine-tuné (ours) | `<TODO>` | `<TODO>` | `<TODO>` |
| Δ | `<TODO>` | `<TODO>` | `<TODO>` |

---

## 7. Empreinte carbone

Méthodologie : formule **Green Algorithms** (Lannelongue et al. 2021 ; cf. [backend/utils/carbon.py](backend/utils/carbon.py)).

Constantes utilisées :
- Intensité carbone France 2024 : `0.052 kgCO2eq/kWh` (RTE)
- PUE typique data center : `1.5`
- Tokens LLM (gpt-4o-mini) : `0.01 g CO2eq / 1000 tokens` input, `0.04 g / 1000 tokens` output (estimations publiques OpenAI)

Empreinte cumulée du pipeline complet (sur sample_queries Q1-Q4) :

| Étape | Temps | Tokens LLM | CO2 (g) |
|---|---|---|---|
| Ingestion (8 PDFs) | `<TODO>` | 0 | `<TODO>` |
| Génération synthétique (1772 paires) | ~31 min CPU | ~560K | ~10.2 |
| Paraphrasing gold (240 paires) | ~1 min CPU | ~12K | ~0.5 |
| Fine-tuning e5-base | `<TODO>` | 0 | `<TODO>` |
| Inférence Tâche 1 par requête | ~18 s CPU | ~5K | ~0.09 |
| Inférence Tâche 2 par requête | ~30 s CPU | 0 | `<TODO>` |
| **Total reproduction complète** | | | **`<TODO>` g CO2eq** |

Pour comparaison, un fine-tuning d'un modèle 7B sur GPU consomme typiquement 10-100 kg CO2eq. Notre pipeline reste **3-4 ordres de grandeur en dessous**.

---

## 8. Limites et risques

Cf. [MODEL_CARD.md](MODEL_CARD.md), section « Limites ». Principaux :

- **Volume d'entraînement faible** (2036 paires). Variance possible entre seeds — à quantifier sur 3 runs.
- **Biais de style synthétique** : 87% des paires d'entraînement proviennent de gpt-4o-mini. Possible biais de formulation que les vraies requêtes du challenge ne partageront pas.
- **Domaine étroit** (défense FR). Pas généralisable hors corpus.
- **Multi-hop reasoning** : la décomposition aide mais ne remplace pas un vrai *chain-of-retrieval* (Q3 du sample reste difficile : hits@10 = 1/3 en baseline).
- **OCR imparfait** : sur ~5% des pages (figures/diagrammes), le texte est bruité ou absent → embeddings dégradés.
- **Détection des hallucinations** : le filtre par entités est basé sur regex (pas NER complet). Faux positifs possibles sur les noms communs capitalisés.

---

## 9. Considérations éthiques

Cf. [RESPONSIBLE_NLP.md](RESPONSIBLE_NLP.md) pour la checklist ACL ARR Responsible NLP complète. Points saillants :

- **Données publiques** : le corpus est constitué de rapports parlementaires (Sénat), communications du Ministère des Armées et autres documents publics. Mention de personnes nommées (sénateurs, responsables) liées à leur fonction officielle uniquement.
- **Pas de redistribution du corpus** : les PDFs restent propriété de leurs auteurs ; `.gitignore` exclut `*.pdf`.
- **Domaine sensible** : la pipeline ne doit pas être utilisée pour produire des contenus militaires offensifs ou opérationnels.
- **Dépendance LLM externe** : `gpt-4o-mini` introduit une dépendance et un coût récurrent. Migration vers un LLM open-source local (Mistral 7B FR ou Qwen 2.5 7B FR) est planifiée.
- **Limitations connues** annoncées explicitement (§8) plutôt que masquées.

---

## 10. Reproductibilité

Tous les artefacts pour reproduire les expériences sont publics :

| Artefact | Emplacement |
|---|---|
| Code | `<TODO: URL GitHub>` (licence MIT) |
| Données d'entraînement consolidées | `Experimental/DATA/training_pairs_full.json` (2036 paires, `_is_gold` flag conservé) |
| Modèle fine-tuné | `<TODO: URL HuggingFace `monamocost/e5-base-monamo-ft`>` |
| Prompts complets | inline dans `pipeline/synth_questions.py`, `pipeline/paraphrase_gold.py`, `pipeline/query_decomposition.py`, `rag/rag_engine.py` |
| Hyperparamètres | inclus dans chaque sortie JSON (`parameters.*`) |
| Seeds | fixes (`42` par défaut), `cudnn.deterministic=True` |
| Empreinte carbone | enregistrée par run dans `models/<run>/carbon_metrics.json` et dans `RAGResponse.carbon` |
| Model card / Datasheet | `MODEL_CARD.md`, `DATASHEET.md` |
| Pipeline complète | `docker compose up` puis `python ingest.py && python run_challenge.py` |

---

## 11. Conclusion

`<TODO: à écrire après les résultats finaux ; 1 paragraphe>`

Brouillon : nous avons présenté Mon Amo Cost, une pipeline RAG modulaire en français qui combine retrieval dense, génération assistée par LLM léger et attribution multi-signal. La conformité aux deux tâches du challenge EvalLLM 2026 est complète. Nos choix architecturaux privilégient explicitement la frugalité (`<TODO: X g CO2eq>` pour le pipeline complet) et la reproductibilité (code, modèle, données, prompts publics, seeds fixes, model card et datasheet aux standards ACL). Les pistes d'amélioration prioritaires sont (i) le passage à un LLM open-source local (Mistral / Qwen FR), (ii) un retrieval hybride BM25+dense+reranker, et (iii) une vérification NLI fine pour les passages non sourcés.

---

## Références

`<TODO: à compléter au format BibTeX/ACL>`

Notes pour le rédacteur :
- Bohnet, B. et al. (2022). *Attributed Question Answering: Evaluation and Modeling for Attributed Large Language Models*.
- Es, S. et al. (2024). *RAGAs: Automated Evaluation of Retrieval Augmented Generation*.
- Gao, L. et al. (2023). *RARR: Researching and Revising What Language Models Say*.
- Gebru, T. et al. (2021). *Datasheets for Datasets*. Comm. ACM.
- Karpukhin, V. et al. (2020). *Dense Passage Retrieval for Open-Domain Question Answering*. EMNLP.
- Khosla, P. et al. (2020). *Supervised Contrastive Learning*. NeurIPS.
- Lannelongue, L. et al. (2021). *Green Algorithms: Quantifying the Carbon Footprint of Computation*.
- Liu, X. et al. (2023). *Self-RAG: Learning to Retrieve, Generate and Critique through Self-Reflection*.
- Mitchell, M. et al. (2019). *Model Cards for Model Reporting*. FAT*.
- Strubell, E. et al. (2019). *Energy and Policy Considerations for Deep Learning in NLP*. ACL.
- Wang, L. et al. (2022/2024). *Text Embeddings by Weakly-Supervised Contrastive Pre-training* (E5).
- Zhang, T. et al. (2020). *BERTScore: Evaluating Text Generation with BERT*. ICLR.

---

## Annexes

### A. Format JSON des sorties (extraits)

#### A.1 Tâche 1

```json
{
  "run_id": "monamo-cost-evallm-2026",
  "parameters": {
    "embedding_model": "intfloat/multilingual-e5-base",
    "embedding_dim": 768,
    "retriever_top_k_chunks": 20,
    "retriever_top_n_pages": 10,
    "llm_provider": "openai",
    "llm_model": "gpt-4o-mini",
    "temperature": 0.2,
    "attribution_threshold": 0.80,
    "attribution_secondary_threshold": 0.85,
    "attribution_topk_per_sentence": 3,
    "attribution_entity_filter": true,
    "decompose_queries": false
  },
  "results": [
    {
      "qid": "Q1",
      "question": "...",
      "retrieved": [
        {"rank": 1, "doc_name": "20260108_NP_Obsdrones_Bulletin-de-veille-n12_0.pdf", "page": 17, "metadata": {}}
      ],
      "answer": "...avec citations [doc.pdf p.N]",
      "metadata": {"tokens_used": 4383}
    }
  ]
}
```

#### A.2 Tâche 2

```json
{
  "run_id": "monamo-cost-evallm-2026",
  "parameters": { /* idem */ },
  "results": [
    {
      "qid": "Q1",
      "attributions": [
        {"sid": "Q1_s0", "text": "# Titre markdown", "attributed_to": []},
        {"sid": "Q1_s1", "text": "Phrase factuelle.", "attributed_to": [{"doc_name": "...", "page": 17}]}
      ]
    }
  ]
}
```

### B. Prompts utilisés (extraits)

`<TODO: copier les SYSTEM_PROMPT depuis le code une fois figés>`

### C. Tableau récapitulatif des hyperparamètres

`<TODO: liste exhaustive>`
