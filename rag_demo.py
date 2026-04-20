from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters.character import CharacterTextSplitter


DEFAULT_QUERY = "What is FAISS?"
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_INDEX_PATH = "faiss_index"
DEFAULT_FILE_PATH = "data/sample.txt"
INDEX_METADATA_FILE = "index_metadata.json"


@dataclass
class Settings:
    file_path: Path
    query: str
    model: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    index_path: Path
    rebuild_index: bool
    save_index: bool


def env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def parse_args() -> Settings:
    parser = argparse.ArgumentParser(
        description="Run a small RAG demo backed by a FAISS index."
    )
    parser.add_argument(
        "--file-path",
        default=os.getenv("RAG_FILE_PATH", DEFAULT_FILE_PATH),
        help="Path to the source text file.",
    )
    parser.add_argument(
        "--query",
        default=os.getenv("RAG_QUERY", DEFAULT_QUERY),
        help="Question to ask the RAG pipeline.",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("RAG_MODEL", DEFAULT_MODEL),
        help="Chat model name used for answer generation.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=int(os.getenv("RAG_CHUNK_SIZE", "200")),
        help="Chunk size used by the text splitter.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=int(os.getenv("RAG_CHUNK_OVERLAP", "20")),
        help="Chunk overlap used by the text splitter.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=int(os.getenv("RAG_TOP_K", "4")),
        help="Number of documents retrieved for the answer.",
    )
    parser.add_argument(
        "--index-path",
        default=os.getenv("RAG_INDEX_PATH", DEFAULT_INDEX_PATH),
        help="Directory used to save and load the FAISS index.",
    )
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        default=env_flag("RAG_REBUILD_INDEX", False),
        help="Force rebuilding the FAISS index even if one already exists.",
    )
    parser.add_argument(
        "--no-save-index",
        action="store_true",
        help="Do not save the generated FAISS index to disk.",
    )
    args = parser.parse_args()

    settings = Settings(
        file_path=Path(args.file_path),
        query=args.query,
        model=args.model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        top_k=args.top_k,
        index_path=Path(args.index_path),
        rebuild_index=args.rebuild_index,
        save_index=not args.no_save_index,
    )
    validate_settings(settings)
    return settings


def validate_settings(settings: Settings) -> None:
    if settings.chunk_size <= 0:
        raise ValueError("--chunk-size must be greater than 0.")
    if settings.chunk_overlap < 0:
        raise ValueError("--chunk-overlap must be 0 or greater.")
    if settings.chunk_overlap >= settings.chunk_size:
        raise ValueError("--chunk-overlap must be smaller than --chunk-size.")
    if settings.top_k <= 0:
        raise ValueError("--top-k must be greater than 0.")


def ensure_api_key() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY is not set.")
        print("Example: set OPENAI_API_KEY before running this script.")
        sys.exit(1)


def ensure_input_file(file_path: Path) -> None:
    if not file_path.exists():
        print(f"Error: input file was not found: {file_path}")
        sys.exit(1)


def load_documents(file_path: Path):
    print(f"Loading source file: {file_path}")
    loader = TextLoader(str(file_path), encoding="utf-8")
    return loader.load()


def split_documents(documents, chunk_size: int, chunk_overlap: int):
    print(
        "Splitting documents "
        f"(chunk_size={chunk_size}, chunk_overlap={chunk_overlap})"
    )
    splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    docs = splitter.split_documents(documents)
    print(f"Created {len(docs)} chunks.")
    return docs


def build_vectorstore(settings: Settings) -> FAISS:
    documents = load_documents(settings.file_path)
    chunks = split_documents(
        documents,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    embeddings = OpenAIEmbeddings()
    print("Building FAISS index from document chunks.")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    if settings.save_index:
        save_vectorstore(vectorstore, settings)
    return vectorstore


def save_vectorstore(vectorstore: FAISS, settings: Settings) -> None:
    settings.index_path.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(settings.index_path))
    metadata_path = settings.index_path / INDEX_METADATA_FILE
    metadata = asdict(settings)
    metadata["file_path"] = str(settings.file_path)
    metadata["index_path"] = str(settings.index_path)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved FAISS index to: {settings.index_path}")


def load_vectorstore(settings: Settings) -> FAISS:
    embeddings = OpenAIEmbeddings()
    print(f"Loading existing FAISS index: {settings.index_path}")
    metadata_path = settings.index_path / INDEX_METADATA_FILE
    if metadata_path.exists():
        print(f"Found index metadata: {metadata_path}")
    return FAISS.load_local(
        str(settings.index_path),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def get_vectorstore(settings: Settings) -> FAISS:
    index_exists = settings.index_path.exists()
    if index_exists and not settings.rebuild_index:
        return load_vectorstore(settings)
    if index_exists and settings.rebuild_index:
        print("Rebuilding FAISS index because --rebuild-index was requested.")
    else:
        print("No saved FAISS index found. Building a new one.")
    return build_vectorstore(settings)


def format_docs(documents: Iterable) -> str:
    return "\n\n".join(doc.page_content for doc in documents)


def build_chain(vectorstore: FAISS, model_name: str, top_k: int):
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You answer questions using only the provided context. "
                "If the context is insufficient, say so clearly.",
            ),
            (
                "human",
                "Context:\n{context}\n\nQuestion:\n{question}",
            ),
        ]
    )
    llm = ChatOpenAI(model=model_name)
    chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return retriever, chain


def print_sources(source_documents) -> None:
    print("\nSources:")
    for index, doc in enumerate(source_documents, start=1):
        snippet = doc.page_content[:160].replace("\n", " ")
        print(f"{index}. {snippet}")


def run(settings: Settings) -> int:
    ensure_api_key()
    ensure_input_file(settings.file_path)

    vectorstore = get_vectorstore(settings)
    retriever, chain = build_chain(
        vectorstore,
        model_name=settings.model,
        top_k=settings.top_k,
    )

    print(f"\nQuestion: {settings.query}")
    source_documents = retriever.invoke(settings.query)
    answer = chain.invoke(settings.query)

    print("=" * 60)
    print("Answer:")
    print(answer)
    print_sources(source_documents)
    print("=" * 60)
    return 0


def main() -> int:
    try:
        settings = parse_args()
        return run(settings)
    except ValueError as error:
        print(f"Configuration error: {error}")
        return 1
    except Exception as error:
        print(f"Runtime error: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
