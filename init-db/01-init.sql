-- Active l'extention vector (support des embeddings) dans PostgreSQL mais seulement si elle n'est pas déjà installée

CREATE EXTENSION IF NOT EXISTS vector;
