# Datasheet — Données du projet Mon Amo Cost

Suit le template [Gebru et al. 2021, *Datasheets for Datasets*](https://arxiv.org/abs/1803.09010).

---

## 1. Motivation

### Pour quoi ce dataset a-t-il été créé ?

Pour l'entraînement et l'évaluation d'un système RAG dans le cadre du **challenge EvalLLM 2026**
(atelier TALN/CORIA). Le dataset agrège trois sources de paires `(question, paragraphe, doc_name, page)` :

- **Gold du challenge** : les 4 questions illustratives fournies dans `sample_queries.json`
- **OSINT du target.txt** : 1 paire illustrative fournie inline dans le règlement
- **Synthétiques** : 1772 paires générées par LLM (gpt-4o-mini) à partir des chunks indexés

### Qui a créé ce dataset ?

- Le corpus PDFs (886 chunks au moment du run) provient du challenge officiel (documents publics
  défense/renseignement français).
- Les annotations gold (questions + retrieved pages) proviennent du `sample_queries.json` fourni
  par les organisateurs du challenge.
- Les paraphrases et questions synthétiques ont été générées par notre équipe via gpt-4o-mini
  d'OpenAI dans le cadre de ce projet.

---

## 2. Composition

### Quels instances composent le dataset ?

Chaque instance est un objet JSON :
```json
{
  "question": "Quel est l'objectif du projet Beehive ?",
  "paragraph": "Le projet Beehive a pour objectif de...",
  "doc_name": "20260108_NP_Obsdrones_Bulletin-de-veille-n12_0.pdf",
  "page": 17,
  "_is_gold": true,
  "_generator_model": "gpt-4o-mini",   // optionnel (paires synthétiques)
  "_generated_at": "2026-05-20T..."   // optionnel
}
```

### Combien d'instances ?

| Source | Nombre | `_is_gold` |
|---|--:|---|
| Sample_queries.json (4 questions × ~6 chunks chacune) | 23 | true |
| OSINT du target.txt | 1 | true |
| Paraphrases LLM des questions gold (10 par question × ~5 chunks) | 240 | true |
| Synthétiques par chunk (886 chunks × 2 questions, modulo filtre) | 1772 | false |
| **Total** | **2036** | 264 gold / 1772 synth |

Fichier consolidé : `Experimental/DATA/training_pairs_full.json` (~2.7 MB).

### Le dataset est-il un échantillon d'un ensemble plus large ?

Oui : 8 PDFs ingérés en DB sur les 374 du corpus complet du challenge. C'est un sous-ensemble
**ciblé** sur les documents référencés dans `sample_queries.json` (pour permettre l'évaluation
contre le gold). L'ingestion complète prendrait ~4-5h CPU.

### Y a-t-il un découpage train/val/test ?

Pas en pré-calculé. Le découpage train/val (80/20) est appliqué **à la volée par `finetune.py`**
de façon **stratifiée par question** (les paragraphes d'une question donnée ne se retrouvent
jamais à cheval entre train et val) avec seed=42 fixe.

Le test set du challenge n'est pas connu des participants — il sera dévoilé lors de la
présentation des résultats.

### Étiquettes / annotations ?

Pour chaque question :
- **Pages gold** (annotation `retrieved` du sample_queries) — uniquement pour les 4 questions gold
- **Réponse de référence** (annotation `answer`) — texte rédigé par les organisateurs
- **Pas d'annotation Tâche 2** : `sample_queries.json` ne fournit pas d'attributions phrase
  par phrase de référence. La calibration de l'attribution est faite via un proxy
  (cf. `backend/calibrate_attribution.py`).

### Données manquantes ?

- Q5 du sample contient des placeholders `"TODO"` → exclue automatiquement par `build_training_pairs`
- Alias documentaire : `r20-7111.pdf` (sample) ↔ `2021_rapport_senat_r20-7111.pdf` (filesystem)
  → résolu via `_DOC_ALIAS` dans `pipeline/finetune.py`
- 8 pages du PDF Reaper ont du texte quasi-vide (figures/diagrammes) → 3 sur 8 récupérées via
  fallback OCR Tesseract, les 5 autres restent peu informatives

### Informations confidentielles ou sensibles ?

Le corpus est composé de **documents publics** (rapports sénat, bulletins de veille du
Ministère des Armées, articles institutionnels). Aucune donnée à caractère personnel attendue.

Cependant, **les questions synthétiques générées par LLM peuvent mentionner des personnes
nommées dans les rapports** (sénateurs, militaires, responsables d'agences). Ces noms sont
publics et liés à leur fonction officielle, mais le dataset n'est pas pensé pour de l'analyse
de personnes.

---

## 3. Collecte des données

### Comment ont été obtenues les données ?

- **PDFs du corpus** : fournis par les organisateurs du challenge à l'inscription
- **Sample queries gold** : fournies idem (`sample_queries.json` distribué officiellement)
- **OSINT** : extrait de `target.txt` (règlement du challenge)
- **Paraphrases gold** : générées via gpt-4o-mini par appel API OpenAI, prompt et code dans
  `backend/pipeline/paraphrase_gold.py` (temperature 0.9 pour diversité)
- **Questions synthétiques** : générées via gpt-4o-mini, prompt et code dans
  `backend/pipeline/synth_questions.py` (temperature 0.7)

### Quel temps de collecte ?

- Ingestion PDF : ~10 min pour 8 PDFs (pdfplumber + OCR Tesseract en fallback)
- Génération synthétique : ~31 min pour 1772 paires (~$0.10)
- Paraphrasing gold : ~1 min pour 240 paires (~$0.001)

### Quelles transformations ont été appliquées ?

1. **Extraction PDF** : pdfplumber par défaut ; fallback OCR Tesseract (lang=fra, 300 dpi) si
   pdfplumber renvoie < 50 caractères pour une page
2. **Chunking** : split sur `\n\s*\n` (paragraphes), recolage des sauts de ligne uniques,
   filtre `min_chars=50` et `max_chars=1500` (avec split phrase si trop long)
3. **Embedding** : `intfloat/multilingual-e5-base` avec préfixes `query:`/`passage:`,
   vecteurs normalisés L2

Aucune transformation n'a été appliquée aux questions/paraphrases générées (texte brut LLM
respecté).

---

## 4. Utilisation

### Tâches prévues

- **Tâche 1 du challenge** : retrieval `(doc_name, page)` + génération de réponse
- **Tâche 2 du challenge** : attribution phrase → source
- **Évaluation comparée** : baseline e5-base zero-shot vs e5-base fine-tuné

### Tâches non recommandées

- Évaluation de modèles génératifs sur d'autres domaines (sport, droit, finance...)
- Construction d'un benchmark public (le corpus appartient au challenge)
- Détection d'OOD / classification (pas la cible du dataset)

---

## 5. Distribution

- **Code** : ce repo, sous licence MIT
- **Données d'entraînement** : `Experimental/DATA/*.json` (sous-dossier du repo)
- **Modèle fine-tuné** : sera publié sur HuggingFace Hub après le challenge :
  `monamocost/e5-base-monamo-ft`
- **Corpus PDFs** : ⚠️ **non redistribuable** — les PDFs proviennent du challenge et restent
  la propriété de leurs auteurs (Ministère des Armées, Sénat, etc.). Le repo `.gitignore`
  inclut `*.pdf` pour ne pas les versionner.

---

## 6. Maintenance

- **Propriétaire** : équipe projet « Mon Amo Cost », contexte académique TALN/CORIA 2026
- **Mise à jour** : pas de plan de mise à jour post-challenge sauf demande explicite
- **Erreurs / corrections** : signaler via les issues du repo Git
