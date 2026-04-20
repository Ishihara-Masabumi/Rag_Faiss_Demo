# Rag_Faiss_Demo

FAISS と OpenAI を使って、RAG の基本的な流れを試せる最小構成のデモです。

## 目的

このリポジトリの目的は、RAG の基本要素を小さく分かりやすく確認できるようにすることです。

- テキストを読み込む
- チャンクに分割する
- ベクトル化して FAISS に保存する
- 類似文書を検索する
- 検索結果を使って回答を生成する

学習用サンプルとしても、簡単な検証用プロジェクトとしても扱いやすい構成を目指しています。

## 特徴

- `requirements.txt` による依存関係の明示
- `README.md` だけでセットアップから実行まで追える構成
- 入力ファイル、質問文、モデル名、チャンク設定を CLI から変更可能
- FAISS インデックスの保存、再利用、再構築に対応
- LangChain の Runnable ベース構成で処理の流れを追いやすい

## ファイル構成

- `rag_demo.py`: RAG デモ本体
- `data/sample.txt`: サンプル文書
- `requirements.txt`: 依存パッケージ一覧

## 動作環境

- Python 3.11 以上を推奨
- `OPENAI_API_KEY` が必要

補足:
`faiss-cpu` は Python バージョンや環境によってインストール時に影響を受けることがあります。うまく入らない場合は、仮想環境を作り直して試すのが安全です。

## セットアップ

### 1. 仮想環境を作成して有効化

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

macOS / Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. 依存関係をインストール

```bash
pip install -r requirements.txt
```

### 3. API キーを設定

Windows PowerShell:

```powershell
$env:OPENAI_API_KEY="your-api-key"
```

macOS / Linux:

```bash
export OPENAI_API_KEY="your-api-key"
```

## 基本実行

```bash
python rag_demo.py
```

デフォルト実行では、次の処理が行われます。

1. `data/sample.txt` を読み込む
2. テキストをチャンクに分割する
3. FAISS インデックスがなければ新規作成する
4. 作成したインデックスを `faiss_index/` に保存する
5. `What is FAISS?` という質問を実行する
6. 回答と参照ソースを表示する

## 主なオプション

```bash
python rag_demo.py --help
```

主な CLI オプション:

- `--file-path`: 入力テキストファイル
- `--query`: 質問文
- `--model`: 回答生成に使うモデル名
- `--chunk-size`: チャンクサイズ
- `--chunk-overlap`: チャンクの重なり幅
- `--top-k`: 取得するチャンク数
- `--index-path`: FAISS インデックスの保存先
- `--rebuild-index`: 既存インデックスを使わず再構築する
- `--no-save-index`: インデックスを保存しない

## 実行例

質問文を変更して実行:

```bash
python rag_demo.py --query "How does RAG improve answer quality?"
```

チャンク設定を変更して実行:

```bash
python rag_demo.py --file-path data/sample.txt --chunk-size 300 --chunk-overlap 50
```

インデックスを再構築して実行:

```bash
python rag_demo.py --rebuild-index
```

保存先を変更して実行:

```bash
python rag_demo.py --index-path custom_faiss_index
```

## 環境変数による上書き

次の環境変数でも設定を上書きできます。

- `OPENAI_API_KEY`
- `RAG_FILE_PATH`
- `RAG_QUERY`
- `RAG_MODEL`
- `RAG_CHUNK_SIZE`
- `RAG_CHUNK_OVERLAP`
- `RAG_TOP_K`
- `RAG_INDEX_PATH`
- `RAG_REBUILD_INDEX`

## 依存パッケージ

このデモでは、主に以下のパッケージを利用しています。

- `langchain`
- `langchain-classic`
- `langchain-community`
- `langchain-openai`
- `langchain-text-splitters`
- `faiss-cpu`
- `openai`
- `tiktoken`

## トラブルシュート

### `OPENAI_API_KEY` が設定されていない

実行前に環境変数 `OPENAI_API_KEY` を設定してください。

### 入力ファイルが見つからない

`--file-path` で正しいファイルを指定するか、`data/sample.txt` を復元してください。

### 保存済みインデックスを作り直したい

`--rebuild-index` を付けて実行してください。

### パッケージのインストールに失敗する

仮想環境を作り直したうえで、利用中の Python バージョンが `faiss-cpu` に対応しているか確認してください。
