# Bascule vers un LLM open-source local

Document : comment passer de `gpt-4o-mini` (API OpenAI) à un LLM open-source en local
pour le **bonus frugalité + open-source** du challenge EvalLLM 2026.

Aucune modification du code applicatif n'est nécessaire : les modules utilisent un
client OpenAI Python qui pointe sur `LLM_BASE_URL` configurable.

---

## 1. Modèles recommandés (FR / RAG)

| Modèle | Taille | Cas d'usage | Atout |
|---|---|---|---|
| `mistralai/Mistral-7B-Instruct-v0.3` | 7B | Inférence RAG, génération FR | Bon support FR natif, licence Apache 2.0 |
| `Qwen/Qwen2.5-7B-Instruct` | 7B | Idem, JSON mode robuste | Excellent multilingue, fonction calling propre |
| `microsoft/Phi-3.5-mini-instruct` | 3.8B | Frugalité maximale (CPU possible) | Très léger, perf raisonnable en FR |
| `mistralai/Mistral-Nemo-Instruct-2407` | 12B | Si GPU 24 Go+ disponible | Meilleure qualité, contexte 128k |

**Reco par défaut** : `mistralai/Mistral-7B-Instruct-v0.3` (équilibre qualité / coût VRAM).
Pour la **frugalité maximale**, `microsoft/Phi-3.5-mini-instruct` quantifié en `int4`.

---

## 2. Trois options de déploiement

### Option A — vLLM (GPU, recommandé)

Le serveur le plus performant côté débit. Compatible OpenAI Chat Completions API + JSON mode.

```bash
pip install vllm
python -m vllm.entrypoints.openai.api_server \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 8192 \
  --dtype bfloat16
```

VRAM requise : ~16 Go fp16, ~8 Go avec `--dtype int8` ou `--quantization awq` (model AWQ-quantifié).

Côté `.env` :
```
LLM_PROVIDER=openai
LLM_BASE_URL=http://localhost:8000/v1
LLM_API_KEY=sk-no-key
LLM_MODEL=mistralai/Mistral-7B-Instruct-v0.3
LLM_JSON_VIA_PROMPT=false   # vLLM supporte response_format en v0.6+
```

### Option B — Ollama (CPU + GPU, le plus simple)

Setup en 2 commandes, parfait pour le développement local et la démo reproductible.

```bash
# Installation : https://ollama.com/download
ollama serve  # démarre le serveur (port 11434)
ollama pull mistral:7b-instruct
# ou : ollama pull qwen2.5:7b-instruct
```

Côté `.env` :
```
LLM_PROVIDER=openai
LLM_BASE_URL=http://localhost:11434/v1
LLM_API_KEY=ollama
LLM_MODEL=mistral:7b-instruct
LLM_JSON_VIA_PROMPT=true    # Ollama n'honore pas toujours response_format
```

⚠️ `LLM_JSON_VIA_PROMPT=true` est important pour Ollama — sinon les modules qui attendent
du JSON (synth, paraphrase, judge, decomposition) parseront du texte libre.

### Option C — llama-cpp-python (CPU pur, frugalité extrême)

Quantization GGUF, tourne sur CPU. ~30-60s par requête typique (vs ~2-5s sur GPU).

```bash
pip install "llama-cpp-python[server]"
# Télécharger un .gguf depuis HuggingFace (ex: TheBloke/Mistral-7B-Instruct-v0.3-GGUF Q4_K_M)
python -m llama_cpp.server \
  --model ./models/mistral-7b-instruct-v0.3.Q4_K_M.gguf \
  --host 0.0.0.0 \
  --port 8000 \
  --n_ctx 8192
```

Côté `.env` :
```
LLM_BASE_URL=http://localhost:8000/v1
LLM_MODEL=mistral-7b-instruct-v0.3
LLM_JSON_VIA_PROMPT=true
```

---

## 3. Bascule en pratique

1. **Lancer le serveur** local selon l'option choisie (A/B/C).
2. **Mettre à jour `backend/.env`** (3 variables : `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`).
3. **Relancer la pipeline** :
   ```bash
   cd backend
   python run_challenge.py --input ../Experimental/DATA/sample_queries.json \
     --out-task1 sample_task1_local.json \
     --out-task2 sample_task2_local.json
   ```
4. **Vérifier** que les générations sont en français et bien sourcées.

**Aucune modification de code applicatif n'est nécessaire.** Le client OpenAI utilisé par
`rag_engine`, `synth_questions`, `paraphrase_gold`, `query_decomposition`, et `judge`
pointera automatiquement sur le serveur local.

---

## 4. Argument frugalité / open-source pour le rapport

Avec un LLM local, l'empreinte carbone et le coût marginal par requête s'effondrent :

| Setup | Coût / 1000 requêtes | CO2 / requête |
|---|---|---|
| OpenAI gpt-4o-mini (référence) | ~0,30 USD | ~0,09 g (1) |
| Mistral 7B local (GPU RTX 3060) | 0 USD marginal | ~0,03 g (2) |
| Mistral 7B local (CPU) | 0 USD marginal | ~0,12 g (3) |
| Phi-3.5-mini local (CPU, frugal+) | 0 USD marginal | ~0,04 g (4) |

(1) tokens × prix OpenAI public + estimation g/token. (2) 5s GPU × 170W × 0.052 kgCO2/kWh × PUE 1.5. (3) 30s CPU × 65W. (4) 10s CPU × 65W (Phi est ~3x plus rapide).

**Bonus** : reproductibilité totale — le modèle local est figé (poids accessible), pas de
dépendance à un service tiers qui change ses versions/prix sans préavis.

---

## 5. Limites connues

- **JSON mode strict** : tous les modèles locaux ne supportent pas `response_format` aussi
  bien qu'OpenAI. Le fallback `LLM_JSON_VIA_PROMPT=true` ajoute une instruction dans le
  prompt + parsing tolérant ; ça marche dans >95% des cas mais reste plus fragile.
- **Latence** : sur CPU, comptez 30-60s par génération RAG (vs 2-5s pour gpt-4o-mini API).
  Pour les 1772 questions synthétiques de l'entraînement, ça rallonge la pipeline (~10h CPU).
- **Qualité** : Mistral 7B en FR est bon mais ~0.5-1 point en dessous de gpt-4o-mini sur le
  domaine défense — à mesurer via `eval_generation.py` (BertScore + LLM-Judge).

Pour le rapport : produire **deux runs** (OpenAI et Mistral local) puis publier les deux
JSON et leur écart sur les métriques officielles.
