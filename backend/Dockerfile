FROM python:3.11-slim

# Installation des outils système
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl wget git bash \
    && curl -sSL https://raw.githubusercontent.com/vishnubob/wait-for-it/master/wait-for-it.sh \
       -o /usr/local/bin/wait-for-it \
    && chmod +x /usr/local/bin/wait-for-it \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Installation de uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Ajout de uv au Path
ENV PATH="/root/.local/bin:$PATH"

# Dossier de travail
WORKDIR /app

# Copie des fichiers de dépendances 
COPY pyproject.toml uv.lock* ./

# Installation des dépendances via uv
RUN uv sync --frozen || uv pip install

COPY . .

# wait-for-it pour PostgreSQL
RUN curl -sSL https://raw.githubusercontent.com/vishnubob/wait-for-it/master/wait-for-it.sh \
    -o /usr/local/bin/wait-for-it \
    && chmod +x /usr/local/bin/wait-for-it

#  entrypoint
RUN chmod +x entrypoint.sh

EXPOSE 8000

ENTRYPOINT [ "./entrypoint.sh" ]
