"""
run_finetune.py
---------------
Script de lancement du fine-tuning e5.
Charge le CSV de paragraphes et appelle fine_tune().

Usage :
    python run_finetune.py
    python run_finetune.py --epochs 5 --batch_size 8 --output_dir models/mon_modele
"""

import argparse
import pandas as pd
from finetune import fine_tune

# ── Arguments CLI ─────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Fine-tuning multilingual-e5-large")
parser.add_argument("--data_path",   default="data/df_paragraphe_final.csv", help="Chemin vers le CSV des paragraphes")
parser.add_argument("--model_name",  default="intfloat/multilingual-e5-large")
parser.add_argument("--output_dir",  default="models/e5-finetuned")
parser.add_argument("--epochs",      type=int,   default=3)
parser.add_argument("--batch_size",  type=int,   default=16)
parser.add_argument("--lr",          type=float, default=2e-5)
parser.add_argument("--temperature", type=float, default=0.07)
parser.add_argument("--k_pos",       type=int,   default=4,  help="Nb positifs par anchor (même doc)")
parser.add_argument("--k_neg",       type=int,   default=8,  help="Nb négatifs par anchor (autre doc)")
args = parser.parse_args()

# ── Chargement du corpus ──────────────────────────────────────────────────────

print(f"Chargement du corpus : {args.data_path}")
df = pd.read_csv(args.data_path)

# Vérification des colonnes attendues
required_cols = {"paragraphe", "nom_du_fichier", "page"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Colonnes manquantes dans le CSV : {missing}")

# Nettoyage basique
df = df.dropna(subset=["paragraphe", "nom_du_fichier"])
df = df[df["paragraphe"].str.strip() != ""]
df = df[df["paragraphe"] != "[vide]"]
df = df.reset_index(drop=True)

print(f"  {len(df)} paragraphes chargés")
print(f"  {df['nom_du_fichier'].nunique()} documents distincts\n")

# ── Lancement ─────────────────────────────────────────────────────────────────

model = fine_tune(
    df_paragraphe = df,
    model_name    = args.model_name,
    output_dir    = args.output_dir,
    epochs        = args.epochs,
    batch_size    = args.batch_size,
    lr            = args.lr,
    temperature   = args.temperature,
    k_pos         = args.k_pos,
    k_neg         = args.k_neg,
)

print("\nFine-tuning terminé.")
