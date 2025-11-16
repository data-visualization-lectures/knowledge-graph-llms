# Streamlit Community Cloud へのデプロイ手順

## 前提条件

1. GitHub アカウント
2. Streamlit Community Cloud アカウント（https://streamlit.io/cloud）
3. OpenAI API キー

## デプロイ手順

### 1. GitHubにプッシュ

変更をGitHubにプッシュします：

```bash
cd /Users/yuichiyazaki/Library/CloudStorage/Dropbox/Playground_GenAI/knowledge-graph-llms
git add .
git commit -m "Add Streamlit Cloud deployment configuration"
git push origin main
```

### 2. Streamlit Community Cloudで新規アプリをデプロイ

1. **Streamlit Community Cloud にアクセス**
   - https://share.streamlit.io にアクセス
   - GitHubアカウントでログイン

2. **「New app」をクリック**

3. **リポジトリ情報を入力**
   - Repository: `your-github-username/knowledge-graph-llms`
   - Branch: `main`
   - Main file path: `app.py`

4. **「Deploy」をクリック**

### 3. Secrets（環境変数）を設定

デプロイ後、以下の手順でAPIキーを設定します：

1. Streamlit Community Cloud のダッシュボードで、デプロイしたアプリの「⋮」メニューをクリック
2. 「Settings」を選択
3. 「Secrets」セクションを開く
4. 以下の内容を入力：

```toml
OPENAI_API_KEY = "your-openai-api-key-here"
```

5. 「Save」をクリック

> ⚠️ **重要**: `.env` ファイルには機密情報が含まれているため、絶対にGitHubにコミットしないでください。`.gitignore` に `.env` が含まれています。

### 4. アプリにアクセス

デプロイが完了したら、Streamlit Cloud が提供するURLでアプリにアクセスできます。

例：`https://share.streamlit.io/your-username/knowledge-graph-llms/main/app.py`

## トラブルシューティング

### モジュールが見つからないエラー

`requirements.txt` を確認し、必要なパッケージがすべてリストアップされているか確認してください。

```bash
pip freeze > requirements.txt
```

### APIキーが認識されない

Streamlit Cloud の Secrets が正しく設定されているか確認してください。コード内では `os.getenv("OPENAI_API_KEY")` でアクセスしています。

### ファイルサイズが大きい

Streamlit Cloud のアップロード制限は 200MB です。`knowledge_graph.html` などの大きなファイルは `.gitignore` に追加されているので問題ありません。

## ローカル開発時の環境設定

ローカルで開発する場合は、`.env` ファイルを作成してAPIキーを設定してください：

```
OPENAI_API_KEY=your-openai-api-key-here
```

実行：
```bash
streamlit run app.py
```
