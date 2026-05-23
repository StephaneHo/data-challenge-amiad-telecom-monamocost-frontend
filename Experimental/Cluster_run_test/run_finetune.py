"""
run_finetune.py
"""
print("lancement programme...")
import argparse
import pandas as pd
from finetune import fine_tune, subsample_paragraphs

parser = argparse.ArgumentParser(description="Fine-tuning multilingual-e5-large")
parser.add_argument("--data_path",     default="DATA/df_paragraphe_final.csv")
parser.add_argument("--model_name",    default="intfloat/multilingual-e5-large")
parser.add_argument("--output_dir",    default="models/e5-finetuned")
parser.add_argument("--epochs",        type=int,   default=3)
parser.add_argument("--batch_size",    type=int,   default=16)
parser.add_argument("--lr",            type=float, default=2e-5)
parser.add_argument("--temperature",   type=float, default=0.07)
parser.add_argument("--k_pos",         type=int,   default=4)
parser.add_argument("--k_neg",         type=int,   default=8)
parser.add_argument("--warmup_steps",  type=int,   default=100,  help="Steps de warmup LR")
parser.add_argument("--resume_from",   default=None,             help="Chemin vers un checkpoint ex: models/e5-finetuned/checkpoints/epoch_2")
parser.add_argument(
    "--max_paragraphs_per_doc",
    type=int,
    default=None,
    help="Max paragraphes par document (échantillonnage régulier, tous les docs conservés)",
)
parser.add_argument(
    "--max_samples",
    type=int,
    default=None,
    help="Plafond global de paragraphes après sous-échantillonnage par document",
)
parser.add_argument("--seed", type=int, default=42, help="Seed pour le sous-échantillonnage")
parser.add_argument("--log_every", type=int, default=10, help="Log toutes les N steps (0 = off)")
parser.add_argument(
    "--max_seq_length",
    type=int,
    default=512,
    help="Tronque les séquences au-delà de N tokens (réduit la VRAM)",
)
parser.add_argument(
    "--encode_batch_size",
    type=int,
    default=64,
    help="Sous-batch max pour l'encodage (anchors/candidates séparés)",
)
args = parser.parse_args()

print(f"Chargement du corpus : {args.data_path}")
df = pd.read_csv(args.data_path)

required_cols = {"paragraphe", "nom_du_fichier", "page"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Colonnes manquantes : {missing}")

df = df.dropna(subset=["paragraphe", "nom_du_fichier"])
df = df[df["paragraphe"].str.strip() != ""]
df = df[df["paragraphe"] != "[vide]"]
df = df.reset_index(drop=True)
n_before = len(df)
df = subsample_paragraphs(
    df,
    max_paragraphs_per_doc=args.max_paragraphs_per_doc,
    max_samples=args.max_samples,
    seed=args.seed,
)
if len(df) < n_before:
    print(
        f"  Sous-échantillonnage : {n_before} → {len(df)} paragraphes "
        f"({df['nom_du_fichier'].nunique()} documents)"
    )
print(f"  {len(df)} paragraphes | {df['nom_du_fichier'].nunique()} documents\n")

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
    warmup_steps  = args.warmup_steps,
    resume_from   = args.resume_from,
    log_every     = args.log_every,
    max_seq_length    = args.max_seq_length,
    encode_batch_size = args.encode_batch_size,
)

print("\nFine-tuning terminé.")