from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader

import os
import sys

# OpenAI API key（環境変数に設定してあることを前提）
# export OPENAI_API_KEY="your-api-key"

def main():
    # APIキーのチェック
    if not os.getenv("OPENAI_API_KEY"):
        print("エラー: OPENAI_API_KEY環境変数が設定されていません")
        print("設定方法: export OPENAI_API_KEY='your-api-key'")
        sys.exit(1)

    # ファイルパスのチェック
    file_path = "data/sample.txt"
    if not os.path.exists(file_path):
        print(f"エラー: {file_path} が見つかりません")
        sys.exit(1)

    try:
        # 1. テキストデータの読み込み
        print("テキストデータを読み込んでいます...")
        loader = TextLoader(file_path, encoding="utf-8")
        documents = loader.load()

        # 2. テキストの分割（チャンクに）
        print("テキストを分割しています...")
        text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)
        docs = text_splitter.split_documents(documents)
        print(f"{len(docs)}個のチャンクに分割しました")

        # 3. ベクトル埋め込みとFAISSインデックス作成
        print("ベクトル埋め込みを作成しています...")
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(docs, embeddings)

        # オプション: ベクトルストアの保存
        # vectorstore.save_local("faiss_index")

        # 4. RAGチェーンの作成
        print("RAGチェーンを作成しています...")
        retriever = vectorstore.as_retriever()
        llm = ChatOpenAI(model_name="gpt-3.5-turbo")  # or "gpt-4"

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=True
        )

        # 5. 質問を投げてみる
        query = "What is FAISS?"
        print(f"\n質問: {query}")
        print("回答を生成しています...\n")

        result = qa_chain(query)

        print("=" * 50)
        print("Question:", query)
        print("Answer:", result["result"])
        print("\nSources:")
        for i, doc in enumerate(result["source_documents"], 1):
            print(f"{i}. {doc.page_content[:100]}...")
        print("=" * 50)

    except Exception as e:
        print(f"エラーが発生しました: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

