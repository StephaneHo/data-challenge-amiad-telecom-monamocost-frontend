# Mail au coéquipier — déploiement Qwen 2.5 7B sur GPU

---

**Objet** : Challenge EvalLLM 2026 — déploiement Qwen 2.5 7B sur ton GPU pour la phase de test

Salut,

J'ai besoin de ton GPU pour la **phase de test du challenge RAG EvalLLM 2026**. L'idée
est de servir **Qwen 2.5 7B Instruct** via vLLM avec une API OpenAI-compatible, pour que
mon code (qui utilise le SDK OpenAI Python) puisse l'appeler à distance sans modification.

Sur mon CPU, une seule réponse RAG prend ~21 min → injouable pour les 3 jours de test.
Sur ton GPU on devrait tomber à ~10 s/requête et **gagner les 3 bonus du challenge en même
temps** (open source + reproductibilité + frugalité ~0.04 g CO2/req vs ~0.09 pour gpt-4o-mini).

## TL;DR — 3 commandes (à exécuter sur la machine GPU distante)

```bash
# 1. Setup env (Python 3.11+ recommandé)
pip install vllm

# 2. Téléchargement + lancement du serveur (DL ~15 GB la première fois)
#    --host 0.0.0.0 = écoute toutes les interfaces, indispensable pour accès distant.
#    --api-key SECRET_KEY = optionnel mais recommandé si l'URL est publique.
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-7B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype bfloat16 \
  --api-key sk-monamocost-2026

# 3. Smoke test depuis la machine GPU (le serveur doit afficher "Application startup complete")
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-monamocost-2026" \
  -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "messages": [{"role": "user", "content": "Bonjour, qui es-tu ?"}],
    "max_tokens": 100
  }'
```

## Ce dont j'ai besoin de ta part

1. **URL joignable depuis ma machine** vers ton serveur vLLM. Selon où tu déploies :
   - **Cloud GPU** (RunPod, Vast.ai, AWS, GCP...) → URL publique du genre
     `https://gpu-xyz.runpod.io/v1` (ajoute une clé d'auth avec `--api-key <secret>`
     côté vLLM, je la mettrai dans mon `.env`).
   - **Cluster université / lab** → soit IP publique, soit on fait un **tunnel SSH** :
     `ssh -L 8000:gpu-node:8000 user@cluster.example.fr` côté toi, puis tu me donnes
     ton hostname + on tunnelle aussi de chez moi → vers chez toi → vers le cluster.
     (Si on choisit cette option je peux t'aider à monter le double-tunnel.)
   - **Serveur interne VPN** → IP privée + connexion au VPN de mon côté.

2. M'envoyer un OK quand le serveur tourne avec l'URL et la clé d'auth si applicable,
   je pointe mon `.env` dessus et c'est parti.

## Détails utiles

- **VRAM nécessaire** :
  - `--dtype bfloat16` : ~16 Go (le défaut, qualité max)
  - `--dtype float16` : ~16 Go aussi
  - Avec AWQ-quantized (`Qwen/Qwen2.5-7B-Instruct-AWQ` à la place du modèle base) : ~8 Go,
    bonne option si ta carte fait <16 Go.

- **Conseil perf** : ajoute `--max-model-len 8192` (contexte max). Notre RAG envoie
  ~3000-4000 tokens par requête, 8192 suffit largement et libère de la KV cache.

- **Logs** : vLLM affiche les requêtes traitées en temps réel sur stdout, pratique pour
  débugger.

## Une fois le serveur up, voilà ce que je fais de mon côté

Je mets dans mon `backend/.env` :
```
LLM_PROVIDER=openai
LLM_BASE_URL=http://<TON_URL_PUBLIQUE_OU_TUNNELÉE>:8000/v1
LLM_API_KEY=sk-monamocost-2026      # même valeur que ton --api-key
LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
LLM_JSON_VIA_PROMPT=false           # vLLM supporte response_format
```

Et je lance :
```powershell
python run_challenge.py --input <questions_du_challenge>.json \
  --out-task1 task1.json --out-task2 task2.json
```

→ Le code applicatif n'a aucune modification, tout passe via `LLM_BASE_URL`.

## Tests / réponses attendues

J'ai déjà validé Qwen 2.5 7B sur Q1 du sample en local (CPU) — réponse de 17 phrases bien
sourcée, citations `[doc.pdf p.N]` correctement insérées. La pipeline est prête côté code.

Sur ton GPU on aura aussi le **fine-tuning** (l'autre handoff que je t'ai préparé :
`backend/HANDOFF_FINETUNE.md` + `Experimental/DATA/training/training_pairs_full.json`).

## Calendrier

- Phase de test challenge : 3 jours consécutifs entre le 04 et le 29 mai.
  Idéalement, le serveur doit tourner pendant ces 3 jours-là.
- Soumission article : 12 juin (j'aurai besoin de tes mesures GPU pour l'empreinte carbone
  dans le rapport).

Merci ! Je suis dispo pour aider à debug si problème de setup.

— `<TON_PRÉNOM>`

---

## Annexe — alternative Ollama (plus simple, légèrement moins perf que vLLM)

Si tu préfères une stack ultra-simple plutôt que vLLM :

```bash
# Installation : https://ollama.com/download
ollama serve  # daemon
ollama pull qwen2.5:7b-instruct
```

Puis dans le `.env` de mon côté :
```
LLM_BASE_URL=http://<TON_HOST>:11434/v1
LLM_MODEL=qwen2.5:7b-instruct
LLM_JSON_VIA_PROMPT=true   # Ollama supporte mal response_format
```

Trade-off : Ollama est plus simple à mettre en route mais ~30% plus lent que vLLM en throughput.
Pour 50 questions de test sur 3 jours, n'importe lequel des deux convient.

## Liens utiles

- Documentation vLLM : <https://docs.vllm.ai/>
- Modèle Qwen 2.5 sur HuggingFace : <https://huggingface.co/Qwen/Qwen2.5-7B-Instruct>
- Détails complets de la pipeline (côté projet) : `backend/HANDOFF_LOCAL_LLM.md` dans le repo
- Règlement challenge : <https://evalllm2026.sciencesconf.org/>
