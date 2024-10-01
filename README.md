# llm_sps  
Social Particle Swarm model (SPS model) with LLM

## ファイル構成
- `llm_sps.py`: メインファイル
- `sps_visualization_functions.py`: Big Five性格特性ごとの平均協力率、平均移動距離、最終スコアを可視化する関数（USE_PERSONALITY = Trueの場合）
- `graph_gen.py`: 任意のagent_data.csvに基づいてsps_visualization_functions.pyを実行

## 環境構築

1. Dockerをインストールします。
2. リポジトリをクローンします：
   ```bash
   git clone https://github.com/t-hishiki7/llm_sps.git
   cd llm_sps
   ```
3. Dockerイメージをビルドします：
   ```bash
   docker build -t llm_sps .
   ```
4. コンテナを実行します：
   ```bash
   docker run -it --rm -v $(pwd):/app llm_sps
   ```

## 開発

1. Poetryをインストールします。
2. 依存関係をインストールします：
   ```bash
   poetry install
   ```
3. 仮想環境を有効にします：
   ```bash
   poetry shell
   ```
4. アプリケーションを実行します：
   ```bash
   python llm_sps.py
   ```

## 環境変数
`.env`ファイルに以下の環境変数を設定してください：
- LINE_TOKEN: LINEの通知用トークン
- MODEL: 使用するLLMモデル
- DEBUG: デバッグモード（True/False）
- DATABASE_URL: データベース接続URL

## 注意事項
- `.env`ファイルは`.gitignore`に含まれているため、GitHubにはアップロードされません。
- `result/`ディレクトリと`sps_simulation_*`ディレクトリは現在コメントアウトされていますが、必要に応じて`.gitignore`から削除してください。

## テスト
テストを実行するには：
```bash
poetry run pytest
```

テストファイルは`tests`ディレクトリに配置してください。各テストファイルは`test_`で始まる名前にしてください。

### テストの追加方法
1. `tests`ディレクトリに新しいテストファイルを作成します（例：`test_llm_sps.py`）。
2. テスト関数を作成します。各テスト関数は`test_`で始まる名前にしてください。
3. `pytest`を使用してアサーションを書きます。

例：
```python
# tests/test_llm_sps.py
from llm_sps import some_function

def test_some_function():
    result = some_function(input_data)
    assert result == expected_output
```

### カバレッジレポートの生成
テストのカバレッジレポートを生成するには：
```bash
poetry run pytest --cov=llm_sps tests/
```

これにより、プロジェクトのコードカバレッジが表示されます。

## CI/CD
このプロジェクトはGitHub Actionsを使用して継続的インテグレーション（CI）を実行しています。プルリクエストを作成すると、自動的にテストが実行されます。

`.github/workflows/ci.yml`ファイルでCI設定を確認できます。

## コントリビューション
1. このリポジトリをフォークします。
2. 新しいブランチを作成します（`git checkout -b feature/amazing-feature`）。
3. 変更をコミットします（`git commit -m 'Add some amazing feature'`）。
4. ブランチにプッシュします（`git push origin feature/amazing-feature`）。
5. プルリクエストを作成します。


## 連絡先
プロジェクトオーナー：[Your Name](mailto:your.email@example.com)

プロジェクトリンク：[https://github.com/t-hishiki7/llm_sps](https://github.com/t-hishiki7/llm_sps)