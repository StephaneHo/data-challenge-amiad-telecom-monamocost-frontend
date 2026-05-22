"""
finetune.py
-----------
Fine-tuning de intfloat/multilingual-e5-large par similarité progressive
sur un corpus de paragraphes (paragraphe, nom_du_fichier, page).
"""

import random
import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset
from sentence_transformers import SentenceTransformer
import pandas as pd


# ── Similarité progressive ────────────────────────────────────────────────────

def compute_sim(doc_order, i, j, doc_i, doc_j):
    """
    sim = 0.0                         si docs différents
    sim = 1.0                         si même paragraphe (dist=0)
    sim = 0.2 + 0.5 * (1/distance)   si même doc
    """
    if doc_i != doc_j:
        return 0.0
    dist = abs(doc_order[i] - doc_order[j])
    if dist == 0:
        return 1.0
    return min(0.2 + 0.5 / dist, 1.0)


# ── Dataset ───────────────────────────────────────────────────────────────────

class SoftSimilarityDataset(Dataset):
    """
    Pour chaque paragraphe anchor, échantillonne :
      - k_pos paragraphes du même document  (sim progressive)
      - k_neg paragraphes d'un autre doc    (sim = 0)
    """

    def __init__(self, df: pd.DataFrame, k_pos: int = 4, k_neg: int = 8):
        df = df.reset_index(drop=True).copy()

        # Position ordinale de chaque paragraphe dans son document
        doc_order = {}
        for _, group in df.groupby("nom_du_fichier"):
            for rank, idx in enumerate(group.index):
                doc_order[idx] = rank
        self.doc_order = doc_order

        self.df         = df
        self.k_pos      = k_pos
        self.k_neg      = k_neg
        self.doc_to_idx = (
            df.groupby("nom_du_fichier")
            .apply(lambda g: g.index.tolist())
            .to_dict()
        )
        self.all_idx = df.index.tolist()

    def __len__(self):
        return len(self.df)

    def _sim(self, i: int, j: int) -> float:
        return compute_sim(
            self.doc_order, i, j,
            self.df.loc[i, "nom_du_fichier"],
            self.df.loc[j, "nom_du_fichier"],
        )

    def __getitem__(self, idx: int):
        row = self.df.loc[idx]
        doc = row["nom_du_fichier"]

        same_doc   = [i for i in self.doc_to_idx[doc] if i != idx]
        other_docs = [i for i in self.all_idx if self.df.loc[i, "nom_du_fichier"] != doc]

        pos_sample = random.sample(same_doc,   min(self.k_pos, len(same_doc)))
        neg_sample = random.sample(other_docs, min(self.k_neg, len(other_docs)))

        candidates = pos_sample + neg_sample
        sims       = [self._sim(idx, j) for j in candidates]

        return {
            "anchor":     "query: "    + row["paragraphe"],
            "candidates": ["passage: " + self.df.loc[j, "paragraphe"] for j in candidates],
            "sims":       sims,
        }


def collate_fn(batch):
    anchors    = [b["anchor"] for b in batch]
    candidates = [c for b in batch for c in b["candidates"]]
    sims       = [s for b in batch for s in b["sims"]]
    k          = len(batch[0]["candidates"])
    return anchors, candidates, torch.tensor(sims, dtype=torch.float), k


# ── Encode avec gradient ──────────────────────────────────────────────────────

def encode_with_grad(
    model: SentenceTransformer,
    texts: list,
    device: torch.device,
) -> torch.Tensor:
    """Forward pass avec gradients (contourne le no_grad de sentence-transformers)."""
    features = model.tokenize(texts)
    features = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in features.items()
    }
    emb = model(features)["sentence_embedding"]
    return F.normalize(emb, p=2, dim=-1)


# ── Soft similarity loss ──────────────────────────────────────────────────────

def soft_similarity_loss(
    a_emb: torch.Tensor,
    c_emb: torch.Tensor,
    target_sims: torch.Tensor,
    k: int,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    Cross-entropy avec soft labels progressifs.
    Chaque anchor est comparé à ses k candidats ; les labels sont
    les similarités cibles normalisées (somme = 1).
    """
    loss_total = torch.tensor(0.0, device=a_emb.device)
    n = len(a_emb)

    for i in range(n):
        c_block = c_emb[i * k : (i + 1) * k]
        t_block = target_sims[i * k : (i + 1) * k]
        logits  = (a_emb[i] @ c_block.T) / temperature

        labels = (
            torch.ones_like(t_block) / k
            if t_block.sum() < 1e-6
            else t_block / t_block.sum()
        )
        loss_total = loss_total + (-(labels * F.log_softmax(logits, dim=0)).sum())

    return loss_total / n


# ── Fine-tuning ───────────────────────────────────────────────────────────────

def fine_tune(
    df_paragraphe: pd.DataFrame,
    model_name:  str   = "intfloat/multilingual-e5-large",
    output_dir:  str   = "models/e5-finetuned",
    epochs:      int   = 3,
    batch_size:  int   = 16,
    lr:          float = 2e-5,
    temperature: float = 0.07,
    k_pos:       int   = 4,
    k_neg:       int   = 8,
) -> SentenceTransformer:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    if device.type == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
        print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Modèle
    model = SentenceTransformer(model_name).to(device)
    model[0].auto_model.gradient_checkpointing_enable()
    print("Gradient checkpointing : activé")

    # DataLoader
    dataset    = SoftSimilarityDataset(df_paragraphe, k_pos=k_pos, k_neg=k_neg)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    use_amp   = device.type == "cuda"
    scaler    = GradScaler(device="cuda", enabled=use_amp)
    print(f"Mixed precision fp16 : {'activé' if use_amp else 'désactivé (CPU)'}")
    print(f"Corpus : {len(dataset)} paragraphes | {len(dataloader)} steps/epoch\n")

    # Boucle d'entraînement
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for step, (anchors, candidates, target_sims, k) in enumerate(dataloader):
            target_sims = target_sims.to(device)

            with autocast(device_type=device.type, enabled=use_amp):
                a_emb = encode_with_grad(model, anchors,    device)
                c_emb = encode_with_grad(model, candidates, device)
                loss  = soft_similarity_loss(a_emb, c_emb, target_sims, k, temperature)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            if step % 10 == 0:
                vram = torch.cuda.memory_allocated() / 1e9 if use_amp else 0.0
                print(
                    f"  Epoch {epoch+1}/{epochs} | "
                    f"Step {step:4d}/{len(dataloader)} | "
                    f"Loss {loss.item():.4f} | "
                    f"VRAM {vram:.2f} GB"
                )

        avg = total_loss / len(dataloader)
        print(f"→ Epoch {epoch+1}/{epochs} terminée — Loss moyenne : {avg:.4f}\n")

    model.save(output_dir)
    print(f"Modèle sauvegardé → {output_dir}")
    return model
