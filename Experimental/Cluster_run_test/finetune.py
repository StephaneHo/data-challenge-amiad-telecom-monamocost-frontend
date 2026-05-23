"""
finetune.py
-----------
Fine-tuning de intfloat/multilingual-e5-large par similarité progressive
sur un corpus de paragraphes (paragraphe, nom_du_fichier, page).
"""

import os
import random
import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from sentence_transformers import SentenceTransformer
import pandas as pd


# ── Sous-échantillonnage corpus ───────────────────────────────────────────────

def subsample_paragraphs(
    df: pd.DataFrame,
    max_paragraphs_per_doc: int | None = None,
    max_samples: int | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Réduit le corpus en gardant tous les documents (similarité progressive préservée)."""
    if max_paragraphs_per_doc is None and max_samples is None:
        return df

    parts = []
    for _, group in df.groupby("nom_du_fichier", sort=False):
        g = group.reset_index(drop=True)
        if max_paragraphs_per_doc and len(g) > max_paragraphs_per_doc:
            if max_paragraphs_per_doc == 1:
                indices = [0]
            else:
                indices = sorted({
                    round(i * (len(g) - 1) / (max_paragraphs_per_doc - 1))
                    for i in range(max_paragraphs_per_doc)
                })
            g = g.iloc[indices]
        parts.append(g)

    out = pd.concat(parts, ignore_index=True)
    if max_samples and len(out) > max_samples:
        out = out.sample(n=max_samples, random_state=seed).reset_index(drop=True)
    return out


# ── Similarité progressive ────────────────────────────────────────────────────

def compute_sim(doc_order, i, j, doc_i, doc_j):
    if doc_i != doc_j:
        return 0.0
    dist = abs(doc_order[i] - doc_order[j])
    if dist == 0:
        return 1.0
    return min(0.2 + 0.5 / dist, 1.0)


# ── Dataset ───────────────────────────────────────────────────────────────────

class SoftSimilarityDataset(Dataset):
    def __init__(self, df: pd.DataFrame, k_pos: int = 4, k_neg: int = 8):
        df = df.reset_index(drop=True).copy()

        doc_order = {}
        for _, group in df.groupby("nom_du_fichier"):
            for rank, idx in enumerate(group.index):
                doc_order[idx] = rank
        self.doc_order = doc_order

        # Accès rapide par dict (évite df.loc en boucle)
        self.texts    = df["paragraphe"].to_dict()
        self.docs     = df["nom_du_fichier"].to_dict()

        self.df       = df
        self.k_pos    = k_pos
        self.k_neg    = k_neg
        self.doc_to_idx = (
            df.groupby("nom_du_fichier")
            .apply(lambda g: g.index.tolist(), include_groups=False)
            .to_dict()
        )
        self.all_idx = df.index.tolist()

    def __len__(self):
        return len(self.df)

    def _sim(self, i: int, j: int) -> float:
        return compute_sim(self.doc_order, i, j, self.docs[i], self.docs[j])

    def __getitem__(self, idx: int):
        doc        = self.docs[idx]
        same_doc   = [i for i in self.doc_to_idx[doc] if i != idx]
        other_docs = [i for i in self.all_idx if self.docs[i] != doc]

        pos_sample = random.sample(same_doc,   min(self.k_pos, len(same_doc)))
        neg_sample = random.sample(other_docs, min(self.k_neg, len(other_docs)))

        candidates = pos_sample + neg_sample
        sims       = [self._sim(idx, j) for j in candidates]

        return {
            "anchor":     "query: "    + self.texts[idx],
            "candidates": ["passage: " + self.texts[j] for j in candidates],
            "sims":       sims,
        }


def collate_fn(batch):
    anchors    = [b["anchor"] for b in batch]
    candidates = [c for b in batch for c in b["candidates"]]
    sims       = [s for b in batch for s in b["sims"]]
    k          = len(batch[0]["candidates"])
    return anchors, candidates, torch.tensor(sims, dtype=torch.float), k


# ── Encode avec gradient ──────────────────────────────────────────────────────

def encode_with_grad(model, texts, device, max_seq_length=None, encode_batch_size=None):
    """Encode texts with gradients, optionally chunked to limit VRAM."""
    if not texts:
        raise ValueError("encode_with_grad: liste de textes vide")

    if encode_batch_size is None or len(texts) <= encode_batch_size:
        features = model.tokenize(texts)
        if max_seq_length is not None:
            for key in ("input_ids", "attention_mask"):
                if key in features:
                    features[key] = features[key][:, :max_seq_length]
        features = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in features.items()
        }
        emb = model(features)["sentence_embedding"]
        return F.normalize(emb, p=2, dim=-1)

    chunks = []
    for start in range(0, len(texts), encode_batch_size):
        chunks.append(
            encode_with_grad(
                model,
                texts[start : start + encode_batch_size],
                device,
                max_seq_length=max_seq_length,
                encode_batch_size=None,
            )
        )
    return torch.cat(chunks, dim=0)


# ── Soft similarity loss ──────────────────────────────────────────────────────

def soft_similarity_loss(a_emb, c_emb, target_sims, k, temperature=0.07):
    loss_total = torch.tensor(0.0, device=a_emb.device)
    n = len(a_emb)
    for i in range(n):
        c_block = c_emb[i * k : (i + 1) * k]
        t_block = target_sims[i * k : (i + 1) * k]
        logits  = (a_emb[i] @ c_block.T) / temperature
        labels  = (
            torch.ones_like(t_block) / k
            if t_block.sum() < 1e-6
            else t_block / t_block.sum()
        )
        loss_total = loss_total + (-(labels * F.log_softmax(logits, dim=0)).sum())
    return loss_total / n


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def save_checkpoint(model, optimizer, scheduler, scaler, epoch, avg_loss, ckpt_dir):
    path = os.path.join(ckpt_dir, f"epoch_{epoch}")
    os.makedirs(path, exist_ok=True)
    model.save(path)
    torch.save(
        {
            "epoch":     epoch,
            "avg_loss":  avg_loss,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler":    scaler.state_dict(),
        },
        os.path.join(path, "training_state.pt"),
    )
    print(f"  ✔ Checkpoint sauvegardé → {path}")


def load_checkpoint(resume_from, model, optimizer, scheduler, scaler, device):
    state_path = os.path.join(resume_from, "training_state.pt")
    if not os.path.isfile(state_path):
        raise FileNotFoundError(f"Pas de training_state.pt dans {resume_from}")

    # Recharge les poids du modèle
    loaded = SentenceTransformer(resume_from).to(device)
    model.load_state_dict(loaded.state_dict())

    state = torch.load(state_path, map_location=device)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    scaler.load_state_dict(state["scaler"])

    print(f"  ✔ Reprise depuis epoch {state['epoch']} (loss={state['avg_loss']:.4f})")
    return state["epoch"]   # start_epoch = epoch déjà terminée


# ── Device ────────────────────────────────────────────────────────────────────

def _resolve_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Device : {device}")
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
        print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        return device

    device = torch.device("cpu")
    print(f"Device : {device}")
    print(f"PyTorch : {torch.__version__} (build CUDA : {torch.version.cuda or 'aucun'})")
    print(
        "⚠ CUDA indisponible — le script tourne sur CPU.\n"
        "  Cause probable : torch installé en variante +cpu.\n"
        "  Fix (venv uv backend/.venv — ne pas utiliser pip seul) :\n"
        "    uv pip uninstall torch\n"
        "    uv pip install torch --index-url https://download.pytorch.org/whl/cu128\n"
        "  Vérifier via ..\\backend\\.venv\\Scripts\\python.exe -c \"import torch; print(torch.cuda.is_available())\""
    )
    return device


# ── Fine-tuning ───────────────────────────────────────────────────────────────

def fine_tune(
    df_paragraphe: pd.DataFrame,
    model_name:   str   = "intfloat/multilingual-e5-large",
    output_dir:   str   = "models/e5-finetuned",
    epochs:       int   = 3,
    batch_size:   int   = 16,
    lr:           float = 2e-5,
    temperature:  float = 0.07,
    k_pos:        int   = 4,
    k_neg:        int   = 8,
    warmup_steps: int   = 100,
    resume_from:  str   = None,
    log_every:    int   = 10,
    max_seq_length: int | None = 512,
    encode_batch_size: int | None = 64,
) -> SentenceTransformer:

    device = _resolve_device()

    ckpt_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    model = SentenceTransformer(model_name).to(device)
    model[0].auto_model.gradient_checkpointing_enable()

    dataset    = SoftSimilarityDataset(df_paragraphe, k_pos=k_pos, k_neg=k_neg)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, pin_memory=(device.type == "cuda"),
    )

    total_steps = epochs * len(dataloader)
    optimizer   = torch.optim.AdamW(model.parameters(), lr=lr)

    # ── Scheduler : warmup linéaire puis cosine ───────────────────────────────
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)          # montée linéaire
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + torch.cos(torch.tensor(3.14159 * progress)).item()))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    # ─────────────────────────────────────────────────────────────────────────

    use_amp = device.type == "cuda"
    scaler  = GradScaler(device="cuda", enabled=use_amp)

    # Reprise éventuelle
    start_epoch = 0
    if resume_from and os.path.isdir(resume_from):
        start_epoch = load_checkpoint(resume_from, model, optimizer, scheduler, scaler, device)

    print(f"Mixed precision fp16 : {'activé' if use_amp else 'désactivé'}")
    print(f"Max seq length : {max_seq_length or 'modèle (défaut)'}")
    print(f"Encode batch size : {encode_batch_size or 'illimité (attention VRAM)'}")
    print(f"Corpus : {len(dataset)} paragraphes | {len(dataloader)} steps/epoch")
    print(f"Scheduler : warmup {warmup_steps} steps → cosine sur {total_steps} steps")
    print(f"Epochs : {start_epoch+1} → {epochs}\n")

    global_step = start_epoch * len(dataloader)

    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0.0

        for step, (anchors, candidates, target_sims, k) in enumerate(dataloader):
            target_sims = target_sims.to(device)

            with autocast(device_type=device.type, enabled=use_amp):
                a_emb = encode_with_grad(
                    model, anchors, device,
                    max_seq_length=max_seq_length,
                    encode_batch_size=encode_batch_size,
                )
                c_emb = encode_with_grad(
                    model, candidates, device,
                    max_seq_length=max_seq_length,
                    encode_batch_size=encode_batch_size,
                )
                loss = soft_similarity_loss(a_emb, c_emb, target_sims, k, temperature)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()        # ← step scheduler à chaque batch
            global_step += 1

            total_loss += loss.item()
            if log_every > 0 and step % log_every == 0:
                vram = torch.cuda.memory_allocated() / 1e9 if use_amp else 0.0
                print(
                    f"  Epoch {epoch+1}/{epochs} | Step {step:4d}/{len(dataloader)} | "
                    f"Loss {loss.item():.4f} | LR {scheduler.get_last_lr()[0]:.2e} | "
                    f"VRAM {vram:.2f} GB"
                )

        avg = total_loss / len(dataloader)
        print(f"→ Epoch {epoch+1}/{epochs} terminée — Loss moyenne : {avg:.4f}\n")

        # Checkpoint après chaque epoch
        save_checkpoint(model, optimizer, scheduler, scaler, epoch + 1, avg, ckpt_dir)

    model.save(output_dir)
    print(f"Modèle final sauvegardé → {output_dir}")
    return model