FROM python:3.9-slim

WORKDIR /app

# システムの依存関係とpoetryのインストール
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir poetry

# 依存関係ファイルのコピーとインストール
COPY pyproject.toml poetry.lock* ./
RUN poetry config virtualenvs.create false \
    && poetry install --no-dev --no-interaction --no-ansi

# アプリケーションのコピー
COPY . .

# アプリケーションの実行
CMD ["poetry", "run", "python", "llm_sps.py"]