"""
run_finetune.py
"""
print("lancement programme...")
import argparse
import pandas as pd
from finetune import fine_tune

parser = argparse.ArgumentParser(description="Fine-tuning multilingual-e5-large")
parser.add_argument("--data_path",     default="data/df_paragraphe_final.csv")
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
)

print("\nFine-tuning terminé.")