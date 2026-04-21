from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from hashlib import blake2b
from pathlib import Path

import faiss
import hnswlib
import numpy as np


DEFAULT_ARTIFACT_DIR = Path("benchmark_artifacts")
DEFAULT_REPORT_PATH = DEFAULT_ARTIFACT_DIR / "faiss_vs_hnswlib_report.json"
TOP_K = 10


@dataclass
class BenchmarkResult:
    documents: int
    dimension: int
    query_count: int
    top_k: int
    hnsw_m: int
    hnsw_ef_construction: int
    hnsw_ef_search: int
    faiss_avg_ms: float
    hnswlib_avg_ms: float
    faiss_speedup_vs_hnswlib: float
    average_overlap_at_k: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare FAISS exact search with hnswlib HNSW search."
    )
    parser.add_argument("--documents", type=int, default=3000)
    parser.add_argument("--dimension", type=int, default=256)
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--hnsw-m", type=int, default=16)
    parser.add_argument("--hnsw-ef-construction", type=int, default=200)
    parser.add_argument("--hnsw-ef-search", type=int, default=20)
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
        "FAISS exact vector search",
        "hnswlib graph based approximate nearest neighbors",
        "RAG retrieval quality and latency",
        "vector index performance benchmark",
        "semantic retrieval over dense embeddings",
    ]
    suffix = [
        "emphasizes low latency lookups",
        "compares indexing and query speed",
        "focuses on repeated top-k search",
        "examines scalability for embeddings",
        "measures practical retrieval cost",
    ]
    return (
        f"Document {doc_id}. "
        f"{topics[doc_id % len(topics)]}. "
        f"{suffix[(doc_id * 7) % len(suffix)]}. "
        f"Group {(doc_id * 11) % 127}."
    )


def build_query_text(query_id: int) -> str:
    prompts = [
        "Which vector index responds faster?",
        "How does approximate search compare with exact search?",
        "What is the top-k retrieval cost?",
        "How does dataset size affect ANN search?",
        "Which approach is better for repeated queries?",
    ]
    return f"Query {query_id}: {prompts[query_id % len(prompts)]}"


def generate_vectors(count: int, dimension: int, text_builder) -> np.ndarray:
    return np.vstack(
        [hashed_embedding(text_builder(idx), dimension) for idx in range(count)]
    ).astype(np.float32)


def build_faiss_index(vectors: np.ndarray) -> faiss.IndexFlatIP:
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def build_hnswlib_index(
    vectors: np.ndarray,
    m: int,
    ef_construction: int,
    ef_search: int,
) -> hnswlib.Index:
    index = hnswlib.Index(space="ip", dim=vectors.shape[1])
    index.init_index(
        max_elements=vectors.shape[0],
        ef_construction=ef_construction,
        M=m,
    )
    index.add_items(vectors, np.arange(vectors.shape[0]))
    index.set_ef(ef_search)
    return index


def benchmark_faiss(index: faiss.IndexFlatIP, queries: np.ndarray, top_k: int):
    started = time.perf_counter()
    _, labels = index.search(queries, top_k)
    elapsed = time.perf_counter() - started
    return elapsed, labels.tolist()


def benchmark_hnswlib(index: hnswlib.Index, queries: np.ndarray, top_k: int):
    started = time.perf_counter()
    labels, _ = index.knn_query(queries, k=top_k)
    elapsed = time.perf_counter() - started
    return elapsed, labels.tolist()


def average_overlap(left: list[list[int]], right: list[list[int]], top_k: int) -> float:
    overlaps = []
    for left_ids, right_ids in zip(left, right):
        overlap = len(set(left_ids) & set(right_ids)) / top_k
        overlaps.append(overlap)
    return float(sum(overlaps) / len(overlaps))


def run_benchmark(args: argparse.Namespace) -> BenchmarkResult:
    vectors = generate_vectors(args.documents, args.dimension, build_document_text)
    queries = generate_vectors(args.queries, args.dimension, build_query_text)

    faiss_index = build_faiss_index(vectors)
    hnsw_index = build_hnswlib_index(
        vectors,
        m=args.hnsw_m,
        ef_construction=args.hnsw_ef_construction,
        ef_search=args.hnsw_ef_search,
    )

    faiss_elapsed, faiss_results = benchmark_faiss(faiss_index, queries, args.top_k)
    hnsw_elapsed, hnsw_results = benchmark_hnswlib(hnsw_index, queries, args.top_k)

    result = BenchmarkResult(
        documents=args.documents,
        dimension=args.dimension,
        query_count=args.queries,
        top_k=args.top_k,
        hnsw_m=args.hnsw_m,
        hnsw_ef_construction=args.hnsw_ef_construction,
        hnsw_ef_search=args.hnsw_ef_search,
        faiss_avg_ms=faiss_elapsed * 1000 / args.queries,
        hnswlib_avg_ms=hnsw_elapsed * 1000 / args.queries,
        faiss_speedup_vs_hnswlib=(
            (hnsw_elapsed / faiss_elapsed) if faiss_elapsed > 0 else float("inf")
        ),
        average_overlap_at_k=average_overlap(faiss_results, hnsw_results, args.top_k),
    )

    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.artifact_dir / DEFAULT_REPORT_PATH.name
    report_path.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
    return result


def main() -> int:
    args = parse_args()
    result = run_benchmark(args)
    print(json.dumps(asdict(result), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
