from __future__ import annotations

import argparse
import heapq
import json
import sqlite3
import time
from dataclasses import asdict, dataclass
from hashlib import blake2b
from pathlib import Path

import faiss
import numpy as np


DEFAULT_ARTIFACT_DIR = Path("benchmark_artifacts")
DEFAULT_DB_PATH = DEFAULT_ARTIFACT_DIR / "baseline_vectors.sqlite3"
DEFAULT_REPORT_PATH = DEFAULT_ARTIFACT_DIR / "faiss_vs_sqlite_report.json"

TOP_K = 4


@dataclass
class BenchmarkResult:
    documents: int
    dimension: int
    query_count: int
    sqlite_avg_ms: float
    faiss_avg_ms: float
    speedup: float
    same_top_k_for_all_queries: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare SQLite linear vector scan with FAISS search."
    )
    parser.add_argument(
        "--documents",
        type=int,
        default=12000,
        help="Number of synthetic documents to generate.",
    )
    parser.add_argument(
        "--dimension",
        type=int,
        default=256,
        help="Embedding dimension used in the benchmark.",
    )
    parser.add_argument(
        "--queries",
        type=int,
        default=20,
        help="Number of synthetic queries to evaluate.",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR,
        help="Directory where benchmark artifacts are written.",
    )
    return parser.parse_args()


def hashed_embedding(text: str, dimension: int) -> np.ndarray:
    values: list[float] = []
    counter = 0
    while len(values) < dimension:
        digest = blake2b(f"{text}:{counter}".encode("utf-8"), digest_size=32).digest()
        values.extend(byte / 255.0 for byte in digest)
        counter += 1
    vector = np.array(values[:dimension], dtype=np.float32)
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def build_document_text(doc_id: int) -> str:
    topics = [
        "FAISS similarity search for dense vectors",
        "LangChain retrieval and prompt composition",
        "RAG systems that combine retrieval and generation",
        "SQLite based storage with linear vector scan",
        "Approximate nearest neighbor search",
        "Document chunking for semantic retrieval",
    ]
    suffix = [
        "focuses on retrieval latency and throughput",
        "explains trade-offs between simplicity and scale",
        "shows why indexing matters for larger datasets",
        "compares exact scan with dedicated vector search",
        "highlights the effect of document volume",
    ]
    return (
        f"Document {doc_id}. "
        f"{topics[doc_id % len(topics)]}. "
        f"{suffix[(doc_id * 3) % len(suffix)]}. "
        f"This record is used for benchmark group {doc_id % 97}."
    )


def build_query_text(query_id: int) -> str:
    prompts = [
        "How fast is FAISS for dense vector search?",
        "Explain the impact of linear scan on retrieval latency.",
        "Why does indexing help RAG retrieval performance?",
        "Compare SQLite scan and FAISS search throughput.",
        "How does dataset size affect nearest-neighbor search?",
    ]
    return f"Query {query_id}: {prompts[query_id % len(prompts)]}"


def generate_dataset(documents: int, dimension: int) -> tuple[list[str], np.ndarray]:
    texts = [build_document_text(doc_id) for doc_id in range(documents)]
    vectors = np.vstack([hashed_embedding(text, dimension) for text in texts]).astype(
        np.float32
    )
    return texts, vectors


def create_sqlite_db(db_path: Path, texts: list[str], vectors: np.ndarray) -> None:
    if db_path.exists():
        db_path.unlink()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    try:
        connection.execute(
            """
            CREATE TABLE documents (
                doc_id INTEGER PRIMARY KEY,
                content TEXT NOT NULL,
                embedding BLOB NOT NULL
            )
            """
        )
        rows = [
            (doc_id, text, sqlite3.Binary(vectors[doc_id].tobytes()))
            for doc_id, text in enumerate(texts)
        ]
        connection.executemany(
            "INSERT INTO documents (doc_id, content, embedding) VALUES (?, ?, ?)",
            rows,
        )
        connection.commit()
    finally:
        connection.close()


def sqlite_search(connection: sqlite3.Connection, query_vector: np.ndarray, top_k: int):
    best: list[tuple[float, int]] = []
    cursor = connection.execute("SELECT doc_id, embedding FROM documents")
    for doc_id, embedding_blob in cursor:
        vector = np.frombuffer(embedding_blob, dtype=np.float32)
        score = float(np.dot(query_vector, vector))
        if len(best) < top_k:
            heapq.heappush(best, (score, doc_id))
        else:
            heapq.heappushpop(best, (score, doc_id))
    ordered = sorted(best, reverse=True)
    return [doc_id for _, doc_id in ordered]


def build_faiss_index(vectors: np.ndarray) -> faiss.IndexFlatIP:
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def faiss_search(index: faiss.IndexFlatIP, query_vector: np.ndarray, top_k: int):
    _, indices = index.search(query_vector.reshape(1, -1), top_k)
    return indices[0].tolist()


def benchmark_sqlite(connection: sqlite3.Connection, query_vectors: np.ndarray) -> tuple[float, list[list[int]]]:
    started = time.perf_counter()
    results = [sqlite_search(connection, vector, TOP_K) for vector in query_vectors]
    elapsed = time.perf_counter() - started
    return elapsed, results


def benchmark_faiss(index: faiss.IndexFlatIP, query_vectors: np.ndarray) -> tuple[float, list[list[int]]]:
    started = time.perf_counter()
    results = [faiss_search(index, vector, TOP_K) for vector in query_vectors]
    elapsed = time.perf_counter() - started
    return elapsed, results


def run_benchmark(documents: int, dimension: int, query_count: int, artifact_dir: Path) -> BenchmarkResult:
    texts, vectors = generate_dataset(documents, dimension)
    query_vectors = np.vstack(
        [hashed_embedding(build_query_text(query_id), dimension) for query_id in range(query_count)]
    ).astype(np.float32)

    db_path = artifact_dir / DEFAULT_DB_PATH.name
    create_sqlite_db(db_path, texts, vectors)
    index = build_faiss_index(vectors)

    connection = sqlite3.connect(db_path)
    try:
        sqlite_elapsed, sqlite_results = benchmark_sqlite(connection, query_vectors)
    finally:
        connection.close()

    faiss_elapsed, faiss_results = benchmark_faiss(index, query_vectors)
    same_top_k = sqlite_results == faiss_results

    sqlite_avg_ms = sqlite_elapsed * 1000 / query_count
    faiss_avg_ms = faiss_elapsed * 1000 / query_count
    result = BenchmarkResult(
        documents=documents,
        dimension=dimension,
        query_count=query_count,
        sqlite_avg_ms=sqlite_avg_ms,
        faiss_avg_ms=faiss_avg_ms,
        speedup=sqlite_avg_ms / faiss_avg_ms if faiss_avg_ms > 0 else float("inf"),
        same_top_k_for_all_queries=same_top_k,
    )

    report_path = artifact_dir / DEFAULT_REPORT_PATH.name
    report_path.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
    return result


def main() -> int:
    args = parse_args()
    result = run_benchmark(
        documents=args.documents,
        dimension=args.dimension,
        query_count=args.queries,
        artifact_dir=args.artifact_dir,
    )
    print(json.dumps(asdict(result), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
