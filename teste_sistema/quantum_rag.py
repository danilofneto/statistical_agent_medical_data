# ---------------------------------------------------------
# Arquivo: quantum_rag.py
# Descrição: Implementação de Quantum-inspired RAG
#            com logging de scores para análise.
# ---------------------------------------------------------

import json
import os
import time
from typing import List, Tuple
from datetime import datetime

LOG_DIR = "quantum_logs"
os.makedirs(LOG_DIR, exist_ok=True)


def _normalize(scores: List[float]) -> List[float]:
    total = sum(scores)
    if total == 0:
        return [0.0 for _ in scores]
    return [s / total for s in scores]


def quantum_rag_query(
    query: str,
    vectorstore,
    k: int = 5,
    top_n: int = 3,
    alpha: float = 0.5,
    log: bool = True
) -> str:
    """
    Executa Quantum-inspired RAG usando late fusion
    e registra scores para análise posterior.

    Args:
        query (str): pergunta do usuário
        vectorstore: FAISS vectorstore
        k (int): documentos recuperados inicialmente
        top_n (int): documentos retornados após re-ranking
        alpha (float): peso do score semântico
        log (bool): ativa logging

    Returns:
        str: contexto textual re-ranqueado
    """

    # -------------------------------------------------
    # 1. Busca vetorial clássica
    # -------------------------------------------------
    results: List[Tuple] = vectorstore.similarity_search_with_score(
        query,
        k=k
    )

    docs = [r[0] for r in results]
    raw_distances = [r[1] for r in results]

    # -------------------------------------------------
    # 2. Conversão distância → score semântico
    #    (quanto menor a distância, maior o score)
    # -------------------------------------------------
    semantic_scores = [1 / (1 + d) for d in raw_distances]
    semantic_scores_norm = _normalize(semantic_scores)

    # -------------------------------------------------
    # 3. Quantum-inspired fusion
    #    (neste baseline, apenas score semântico,
    #     mas já estruturado para múltiplos sinais)
    # -------------------------------------------------
    quantum_scores = [
        alpha * s
        for s in semantic_scores_norm
    ]

    # -------------------------------------------------
    # 4. Re-ranking
    # -------------------------------------------------
    ranked = sorted(
        zip(docs, quantum_scores, raw_distances),
        key=lambda x: x[1],
        reverse=True
    )

    # -------------------------------------------------
    # 5. Logging (essencial para mestrado)
    # -------------------------------------------------
    if log:
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "query": query,
            "alpha": alpha,
            "k": k,
            "top_n": top_n,
            "results": [
                {
                    "rank": idx + 1,
                    "quantum_score": round(score, 6),
                    "semantic_distance": round(dist, 6),
                    "semantic_score_norm": round(
                        semantic_scores_norm[i], 6
                    ),
                    "content_preview": doc.page_content[:300]
                }
                for idx, (doc, score, dist), i
                in zip(
                    range(len(ranked)),
                    ranked,
                    range(len(semantic_scores_norm))
                )
            ]
        }

        filename = f"{LOG_DIR}/quantum_rag_{int(time.time())}.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(log_entry, f, indent=2, ensure_ascii=False)

    # -------------------------------------------------
    # 6. Construção do contexto final
    # -------------------------------------------------
    context = "\n---\n".join(
        [doc.page_content for doc, _, _ in ranked[:top_n]]
    )

    return context
