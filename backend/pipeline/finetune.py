"""
Fine-tuning contrastif multi-positifs de l'embedder e5.

Trame reprise du notebook `Experimental/Exploration_données.ipynb` :
- `encode_with_grad()` pour bypasser le `torch.no_grad` de `model.encode()`
- `multi_positive_loss()` symétrique avec masque des positifs partagés
- gradient checkpointing pour économiser la VRAM
- mixed precision fp16 si CUDA disponible

Adaptations pour le projet :
- Données depuis la table `chunks` (doc_name, page_number) au lieu d'un CSV
- Modèle par défaut e5-**base** (CPU-faisable, l'e5-large du notebook crashait)
- Clé positive = (doc_name, page) strict
- Split train/val + early stopping
- Sauvegarde au format SentenceTransformers chargeable par `EMBEDDING_MODEL=models/...`
"""

from __future__ import annotations

import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from loguru import logger
from sentence_transformers import SentenceTransformer
from sqlalchemy import select
from sqlalchemy.orm import Session
from torch.utils.data import DataLoader, Dataset

from database.models import Chunk


# ──────────────────────────── Données ──────────────────────────────────────

@dataclass
class TrainingExample:
    """Paire (question, paragraphe) avec la clé de positifs."""

    question: str
    paragraph: str
    doc_name: str
    page_number: int
    is_gold: bool = False  # True = ground truth (sample_queries, OSINT) ; False = synthétique

    @property
    def positive_key(self) -> tuple[str, int]:
        return (self.doc_name, self.page_number)


# Aliases de docs : sample_queries.json référence des noms courts, le filesystem
# a des noms longs. Étendre ici si d'autres divergences apparaissent.
_DOC_ALIAS: dict[str, str] = {
    "r20-7111.pdf": "2021_rapport_senat_r20-7111.pdf",
}


def _resolve_doc_alias(name: str) -> str:
    return _DOC_ALIAS.get(name, name)


def load_extra_examples(extra_path: Path, is_gold: bool = False) -> list[TrainingExample]:
    """
    Charge des paires depuis un fichier JSON, format :
        [
          {
            "question": "...",
            "paragraph": "...",
            "doc_name": "guide_osint_infrastructure_v2.pdf",
            "page": 1
          },
          ...
        ]

    `is_gold=True` à passer si le fichier contient des paires ground-truth
    (ex: extra_training_examples.json avec l'OSINT du target.txt) pour qu'elles
    soient éligibles à `--gold-upweight`.
    """
    data = json.loads(Path(extra_path).read_text(encoding="utf-8"))
    out: list[TrainingExample] = []
    n_gold = 0
    for it in data:
        # `_is_gold` dans le fichier prime sur l'argument (utile pour les exports
        # de training_set produits par `finetune.py --export-to`).
        per_item_gold = it.get("_is_gold", is_gold)
        if per_item_gold:
            n_gold += 1
        out.append(
            TrainingExample(
                question=it["question"],
                paragraph=it["paragraph"],
                doc_name=it["doc_name"],
                page_number=int(it.get("page", 1)),
                is_gold=per_item_gold,
            )
        )
    logger.info(
        f"[Finetune] {len(out)} paires depuis {extra_path.name} "
        f"({n_gold} gold, {len(out) - n_gold} synth)"
    )
    return out


def build_training_pairs(
    session: Session, gold_path: Path, skip_placeholder: bool = True
) -> list[TrainingExample]:
    """
    Construit les paires d'entraînement depuis un fichier au format
    `sample_queries.json` : pour chaque (question, retrieved[]), on charge
    les chunks correspondant à (doc_name, page) depuis la DB.
    """
    data = json.loads(Path(gold_path).read_text(encoding="utf-8"))
    examples: list[TrainingExample] = []
    skipped: list[str] = []

    for q in data["results"]:
        qid = q["qid"]
        question = q["question"]
        if skip_placeholder and question.strip().upper() == "TODO":
            skipped.append(f"{qid} (question placeholder)")
            continue

        for ref in q.get("retrieved", []):
            doc = _resolve_doc_alias(ref["doc_name"])
            page = int(ref["page"])
            if doc == "TODO" or page <= 0:
                skipped.append(f"{qid} ({doc} p.{page} placeholder)")
                continue

            stmt = (
                select(Chunk)
                .where(Chunk.doc_name == doc)
                .where(Chunk.page_number == page)
                .order_by(Chunk.chunk_index)
            )
            rows = session.execute(stmt).scalars().all()
            if not rows:
                skipped.append(f"{qid} ({doc} p.{page} : aucun chunk en DB)")
                continue
            for c in rows:
                examples.append(
                    TrainingExample(
                        question=question,
                        paragraph=c.content,
                        doc_name=doc,
                        page_number=page,
                        is_gold=True,
                    )
                )

    logger.info(
        f"[Finetune] {len(examples)} paires construites depuis {gold_path.name} "
        f"({len(skipped)} skipped)"
    )
    for s in skipped:
        logger.debug(f"  skip: {s}")
    return examples


def train_val_split(
    examples: list[TrainingExample], val_ratio: float = 0.2, seed: int = 42
) -> tuple[list[TrainingExample], list[TrainingExample]]:
    """Split aléatoire stratifié sur les questions (évite la fuite entre splits)."""
    by_question: dict[str, list[TrainingExample]] = {}
    for ex in examples:
        by_question.setdefault(ex.question, []).append(ex)

    questions = list(by_question.keys())
    rng = random.Random(seed)
    rng.shuffle(questions)

    n_val = max(1, int(len(questions) * val_ratio)) if val_ratio > 0 else 0
    val_qs = set(questions[:n_val])

    train, val = [], []
    for q, exs in by_question.items():
        (val if q in val_qs else train).extend(exs)

    logger.info(
        f"[Finetune] Split par question : {len(questions) - n_val} train / {n_val} val "
        f"→ {len(train)} pairs train / {len(val)} pairs val"
    )
    return train, val


# ──────────────────────────── Dataset & loss ───────────────────────────────

def _word_dropout(text: str, p: float) -> str:
    """Droppe aléatoirement chaque mot avec probabilité p. Préserve l'ordre et la casse."""
    if p <= 0:
        return text
    tokens = text.split()
    if len(tokens) <= 3:
        return text  # trop court → pas touché
    kept = [t for t in tokens if random.random() > p]
    if not kept:
        kept = [tokens[0]]  # garantit qu'on ne renvoie pas une question vide
    return " ".join(kept)


class MultiPositiveDataset(Dataset):
    """
    Dataset PyTorch — applique la convention de préfixes e5 :
    - questions → `query: ...`
    - paragraphes → `passage: ...`

    Si `question_dropout_gold > 0`, applique un word-dropout aléatoire aux questions
    des exemples taggés `is_gold=True` (différent à chaque epoch). Casse la
    mémorisation lexicale lors de l'upweight des gold.
    """

    def __init__(
        self,
        examples: list[TrainingExample],
        question_dropout_gold: float = 0.0,
    ) -> None:
        self.examples = examples
        self.question_dropout_gold = question_dropout_gold

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        ex = self.examples[idx]
        question = ex.question
        if ex.is_gold and self.question_dropout_gold > 0:
            question = _word_dropout(question, self.question_dropout_gold)
        return {
            "question": f"query: {question}",
            "paragraph": f"passage: {ex.paragraph}",
            "key": ex.positive_key,
        }


def collate_fn(batch: list[dict]) -> dict:
    return {
        "questions": [b["question"] for b in batch],
        "paragraphs": [b["paragraph"] for b in batch],
        "keys": [b["key"] for b in batch],
    }


def encode_with_grad(
    model: SentenceTransformer, texts: list[str], device: torch.device
) -> torch.Tensor:
    """
    Forward avec gradients actifs (contournement du `torch.no_grad` interne
    de `SentenceTransformer.encode()`).
    """
    features = model.tokenize(texts)
    features = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in features.items()
    }
    out = model(features)
    return F.normalize(out["sentence_embedding"], p=2, dim=-1)


def multi_positive_loss(
    q_emb: torch.Tensor,
    p_emb: torch.Tensor,
    keys: list[tuple[str, int]],
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    InfoNCE symétrique multi-positifs : pour la question i, sont positives
    toutes les paragraphes du batch dont la clé matche (même doc_name, même page).
    """
    logits = torch.matmul(q_emb, p_emb.T) / temperature

    positive_mask = torch.tensor(
        [[keys[i] == keys[j] for j in range(len(keys))] for i in range(len(keys))],
        dtype=torch.float,
        device=q_emb.device,
    )
    # Sécurité : si une ligne n'a aucun positif (ne devrait pas arriver puisque (i,i) est positif),
    # on remplace 0 par 1 pour éviter NaN
    pos_sum = positive_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
    labels = positive_mask / pos_sum

    loss_q = -(labels * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
    loss_p = -(labels.T * F.log_softmax(logits.T, dim=1)).sum(dim=1).mean()
    return (loss_q + loss_p) / 2.0


# ──────────────────────────── Entraînement ─────────────────────────────────

@dataclass
class FineTuneConfig:
    model_name: str = "intfloat/multilingual-e5-base"
    output_dir: Path = Path("models/e5-base-monamo-ft")
    epochs: int = 3
    batch_size: int = 8
    lr: float = 2e-5
    temperature: float = 0.07
    val_ratio: float = 0.2
    seed: int = 42
    gradient_checkpointing: bool = True
    log_every: int = 5  # log toutes les N steps
    question_dropout_gold: float = 0.0  # word dropout sur questions gold (0=off)


def _evaluate(
    model: SentenceTransformer,
    val_loader: DataLoader,
    device: torch.device,
    temperature: float,
) -> float:
    model.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for batch in val_loader:
            q_emb = encode_with_grad(model, batch["questions"], device)
            p_emb = encode_with_grad(model, batch["paragraphs"], device)
            loss = multi_positive_loss(q_emb, p_emb, batch["keys"], temperature)
            total += loss.item() * len(batch["questions"])
            n += len(batch["questions"])
    return total / max(n, 1)


def fine_tune(
    train_examples: list[TrainingExample],
    val_examples: Optional[list[TrainingExample]] = None,
    config: Optional[FineTuneConfig] = None,
) -> SentenceTransformer:
    """
    Lance le fine-tuning. Sauvegarde le best model (val loss) dans `config.output_dir`.
    Retourne le modèle final chargé.
    """
    from utils.carbon import CarbonTracker

    cfg = config or FineTuneConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tracker = CarbonTracker(label="finetune", device=device.type)
    logger.info(
        f"[Finetune] device={device} | model={cfg.model_name} | "
        f"epochs={cfg.epochs} bs={cfg.batch_size} lr={cfg.lr} temp={cfg.temperature}"
    )
    if device.type == "cuda":
        logger.info(
            f"[Finetune] GPU={torch.cuda.get_device_name(0)} "
            f"VRAM={torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        )

    # Reproductibilité (CPU + CUDA)
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
        # cudnn deterministic : reproductibilité au prix de ~5-15% de vitesse.
        # Acceptable pour ce projet (on veut comparer des hyperparams entre eux).
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.info("[Finetune] CUDA full determinism activé (cudnn.deterministic=True)")

    model = SentenceTransformer(cfg.model_name).to(device)
    if cfg.gradient_checkpointing:
        model[0].auto_model.gradient_checkpointing_enable()
        logger.info("[Finetune] Gradient checkpointing : activé")

    train_loader = DataLoader(
        MultiPositiveDataset(train_examples, question_dropout_gold=cfg.question_dropout_gold),
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
    )
    val_loader: Optional[DataLoader] = None
    if val_examples:
        # Pas de dropout côté val : on veut une mesure honnête de la loss
        val_loader = DataLoader(
            MultiPositiveDataset(val_examples, question_dropout_gold=0.0),
            batch_size=cfg.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            pin_memory=(device.type == "cuda"),
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    autocast_dtype = torch.float16 if use_amp else torch.bfloat16
    logger.info(f"[Finetune] AMP={'fp16 cuda' if use_amp else 'désactivé (CPU)'}")

    best_val = float("inf")
    cfg.output_dir = Path(cfg.output_dir)
    cfg.output_dir.parent.mkdir(parents=True, exist_ok=True)

    import time as _time

    train_t0 = _time.monotonic()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for step, batch in enumerate(train_loader, start=1):
            with torch.amp.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_amp):
                q_emb = encode_with_grad(model, batch["questions"], device)
                p_emb = encode_with_grad(model, batch["paragraphs"], device)
                loss = multi_positive_loss(q_emb, p_emb, batch["keys"], cfg.temperature)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            n_batches += 1
            if step % cfg.log_every == 0:
                vram = torch.cuda.memory_allocated() / 1e9 if use_amp else 0
                logger.info(
                    f"[Finetune] ep{epoch} step{step:4d} | loss={loss.item():.4f}"
                    + (f" | vram={vram:.1f} GB" if use_amp else "")
                )

        train_avg = total_loss / max(n_batches, 1)
        msg = f"[Finetune] === epoch {epoch}/{cfg.epochs} : train_loss={train_avg:.4f}"

        if val_loader is not None:
            val_avg = _evaluate(model, val_loader, device, cfg.temperature)
            msg += f" | val_loss={val_avg:.4f}"
            if val_avg < best_val:
                best_val = val_avg
                _save_model(model, cfg.output_dir, suffix=".best")
                msg += " ✓ best"
        logger.info(msg)

    tracker.metrics.runtime_seconds += _time.monotonic() - train_t0

    # Toujours sauver le final + métriques carbone
    _save_model(model, cfg.output_dir, suffix="")
    tracker.log_summary()
    _carbon_path = cfg.output_dir / "carbon_metrics.json"
    import json as _json

    _carbon_path.write_text(_json.dumps(tracker.metrics.to_dict(), indent=2), encoding="utf-8")
    logger.info(f"[Finetune] Modèle sauvegardé : {cfg.output_dir.resolve()}")
    logger.info(f"[Finetune] Métriques carbone : {_carbon_path}")
    return model


def _save_model(model: SentenceTransformer, path: Path, suffix: str = "") -> None:
    target = path if not suffix else path.parent / (path.name + suffix)
    if target.exists():
        shutil.rmtree(target)
    model.save(str(target))
