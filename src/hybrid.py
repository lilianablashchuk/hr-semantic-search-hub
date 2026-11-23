import os
import re
import json
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

BI_ENCODER_NAME = "intfloat/multilingual-e5-small"
CROSS_ENCODER_NAME = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

def tokenize_uk_en(text: str):
    return re.findall(r"[a-zA-Zа-яА-ЯіІїЇєЄґҐ0-9_-]+", text.lower())

def load_corpus(path="data/corpus.jsonl"):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records

class HybridSearchEngine:
    def __init__(self, records):
        self.records = records
        print(f"Завантаження Bi-Encoder ({BI_ENCODER_NAME}) та Cross-Encoder ({CROSS_ENCODER_NAME})...")
        self.bi_encoder = SentenceTransformer(BI_ENCODER_NAME)
        self.cross_encoder = CrossEncoder(CROSS_ENCODER_NAME)

        passages = [f"passage: [{r['title']} | {r['section']}] {r['text']}" for r in records]
        emb = self.bi_encoder.encode(passages, normalize_embeddings=True, show_progress_bar=False)
        self.embeddings = np.array(emb, dtype="float32")
        self.dense_index = faiss.IndexFlatIP(self.embeddings.shape[1])
        self.dense_index.add(self.embeddings)

        tokenized_corpus = [tokenize_uk_en(f"{r['title']} {r['section']} {r['text']}") for r in records]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def _passes_filter(self, rec: dict, filters: dict) -> bool:
        if not filters:
            return True
        if "type" in filters and rec.get("type") != filters["type"]:
            return False
        if "sensitivity" in filters:
            allowed = filters["sensitivity"]
            if isinstance(allowed, list):
                if rec.get("sensitivity") not in allowed:
                    return False
            elif rec.get("sensitivity") != allowed:
                return False
        if "date_from" in filters and rec.get("date", "") < filters["date_from"]:
            return False
        return True

    def search_bm25(self, query: str, top_k: int = 30, filters: dict = None):
        q_tokens = tokenize_uk_en(query)
        scores = self.bm25.get_scores(q_tokens)
        ranked_indices = np.argsort(scores)[::-1]
        results = []
        for idx in ranked_indices:
            if scores[idx] <= 0:    
                continue
            rec = self.records[idx]
            if self._passes_filter(rec, filters):
                results.append((int(idx), float(scores[idx])))
                if len(results) == top_k:
                    break
        return results

    def search_dense(self, query: str, top_k: int = 30, filters: dict = None):
        q_vec = self.bi_encoder.encode([f"query: {query}"], normalize_embeddings=True, show_progress_bar=False)
        q_vec = np.array(q_vec, dtype="float32")
        scores, indices = self.dense_index.search(q_vec, len(self.records))
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            rec = self.records[idx]
            if self._passes_filter(rec, filters):
                results.append((int(idx), float(score)))
                if len(results) == top_k:
                    break
        return results

    def search_hybrid_rrf(self, query: str, top_k: int = 30, rrf_k: int = 60, filters: dict = None):
        bm25_res = self.search_bm25(query, top_k=top_k, filters=filters)
        dense_res = self.search_dense(query, top_k=top_k, filters=filters)

        rrf_scores = {}
        for rank, (idx, _) in enumerate(bm25_res, start=1):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (rrf_k + rank)
        for rank, (idx, _) in enumerate(dense_res, start=1):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (rrf_k + rank)

        return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    def search_hybrid_rerank(self, query: str, retrieve_k: int = 30, final_k: int = 5, filters: dict = None):
        rrf_candidates = self.search_hybrid_rrf(query, top_k=retrieve_k, rrf_k=60, filters=filters)
        if not rrf_candidates:
            return [], []

        pairs = [(query, f"{self.records[idx]['title']}. {self.records[idx]['text']}") for idx, _ in rrf_candidates]
        ce_scores = self.cross_encoder.predict(pairs)

        reranked = sorted(
            zip([idx for idx, _ in rrf_candidates], [float(s) for s in ce_scores]),
            key=lambda x: x[1],
            reverse=True
        )[:final_k]

        return rrf_candidates[:final_k], reranked

def main():
    os.makedirs("docs", exist_ok=True)
    records = load_corpus()
    engine = HybridSearchEngine(records)

    sample_q = "Як погодити оффер якщо сума перевищує зарплатну вилку на 12 відсотків?"
    rrf_top5, reranked_top5 = engine.search_hybrid_rerank(sample_q, retrieve_k=30, final_k=5)

    print("\n Критерій C2: Вплив Cross-Encoder Reranker на Top-5")
    print(f"Запит: {sample_q}")
    print("До reranking (RRF Top-5):   ", [records[i]["chunk_id"] for i, _ in rrf_top5])
    print("Після reranking (CE Top-5):", [records[i]["chunk_id"] for i, _ in reranked_top5])

    filter_q = "Яка зарплатна вилка для Middle розробника?"
    unfiltered_top5, _ = engine.search_hybrid_rerank(filter_q, final_k=5, filters=None)
    public_only_top5, _ = engine.search_hybrid_rerank(
        filter_q,
        final_k=5,
        filters={"sensitivity": ["public"], "date_from": "2026-09-01"}
    )
    policy_internal_top5, _ = engine.search_hybrid_rerank(
        "Правила роботи з даними та оферами",
        final_k=5,
        filters={"type": "policy", "sensitivity": ["internal"], "date_from": "2026-08-01"}
    )

    print("\nКритерій C3: Pre-filter за метаданими")
    print("Без фільтра:", [(records[i]["chunk_id"], records[i]["sensitivity"]) for i, _ in unfiltered_top5[:3]])
    print("Фільтр sensitivity=['public'], date>='2026-09-01':", [(records[i]["chunk_id"], records[i]["sensitivity"], records[i]["date"]) for i, _ in public_only_top5[:3]])
    print("Фільтр type='policy', sensitivity=['internal'], date>='2026-08-01':", [(records[i]["chunk_id"], records[i]["type"], records[i]["sensitivity"]) for i, _ in policy_internal_top5[:3]])

    case_studies = [
        {
            "label": "ВИГРАШ 1 (Точний код ATS)",
            "query": "Що означає код REJ-COMP в ATS?",
            "expected": "doc25",
            "reason": "BM25 знаходить точний лексичний збіг коду `REJ-COMP`, піднімаючи `doc25` у топ через RRF."
        },
        {
            "label": "ВИГРАШ 2 (Рідкісний оператор X-Ray + контекст)",
            "query": "Оператор X-Ray пошуку site:linkedin.com/in/ для Embedded STM32",
            "expected": "doc08",
            "reason": "Поєднання точного рядка `site:linkedin.com/in/` (BM25) та семантики пошуку інженерів мікроконтролерів (Dense) виводить `doc08` на 1-шу позицію."
        },
        {
            "label": "ВИГРАШ 3 (Абревіатура OAR + семантика бенчмарку)",
            "query": "Цільовий показник Offer Acceptance Rate OAR у воронці",
            "expected": "doc15",
            "reason": "BM25 фіксує точну абревіатуру `OAR`, а Dense пов'язує «цільовий показник» із «бенчмарк 85%» у `doc15`."
        },
        {
            "label": "ПРОГРАШ 1 (Лексичний шум частотних доменних слів)",
            "query": "Як кандидату не хвилюватися і впевнено почуватися на першому робочому дні?",
            "expected": "doc26",
            "reason": "Слово «кандидату» часто зустрічається в документах про інтерв'ю та відмови (`doc05`, `doc09`), тому BM25 зашумлює RRF, тоді як у `doc26` вжито слово «новачок» і чистий Dense знаходить його точніше."
        }
    ]

    cases_md = ""
    print("\n3 виграші та 1 програш гібридного пошуку")
    for cs in case_studies:
        d_top3 = [records[i]["doc_id"] for i, _ in engine.search_dense(cs["query"], top_k=3)]
        b_top3 = [records[i]["doc_id"] for i, _ in engine.search_bm25(cs["query"], top_k=3)]
        h_top3 = [records[i]["doc_id"] for i, _ in engine.search_hybrid_rrf(cs["query"], top_k=3)]
        print(f"{cs['label']} | Очікується: {cs['expected']} | BM25: {b_top3} | Dense: {d_top3} | Hybrid RRF: {h_top3}")
        cases_md += f"### {cs['label']}\n- **Запит:** `{cs['query']}`\n- **Очікуваний документ:** `{cs['expected']}`\n- **Top-3:** BM25 = `{b_top3}`, Dense = `{d_top3}`, Hybrid RRF = `{h_top3}`\n- **Пояснення:** {cs['reason']}\n\n"

    rrf_ids = [f"`{records[i]['chunk_id']}` ({s:.4f})" for i, s in rrf_top5]
    ce_ids = [f"`{records[i]['chunk_id']}` ({s:.4f})" for i, s in reranked_top5]

    report_md = f"""# Звіт про гібридний пошук, RRF та Reranking (Результат 4)

## 1. Архітектура гібридного пошуку 
- **Sparse (BM25Okapi)** + **Dense (`{BI_ENCODER_NAME}` + FAISS)** + **RRF ($k=60$)**.

## 2. Cross-Encoder Reranker для Top-30 -> Top-5 
- **Модель:** `{CROSS_ENCODER_NAME}`
- **Запит:** `{sample_q}`
- **До reranking (RRF Top-5):** {", ".join(rrf_ids)}
- **Після reranking (CE Top-5):** {", ".join(ce_ids)}

## 3. Pre-filter за метаданими 
- Без фільтра: `{[(records[i]['chunk_id'], records[i]['sensitivity']) for i, _ in unfiltered_top5[:3]]}`
- Фільтр `sensitivity=['public'], date >= '2026-09-01'`: `{[(records[i]['chunk_id'], records[i]['sensitivity'], records[i]['date']) for i, _ in public_only_top5[:3]]}`
- Фільтр `type='policy', sensitivity=['internal'], date >= '2026-08-01'`: `{[(records[i]['chunk_id'], records[i]['type'], records[i]['sensitivity']) for i, _ in policy_internal_top5[:3]]}`

## 4. 3 приклади виграшу та 1 приклад програшу 
{cases_md}"""

    with open("docs/hybrid_report.md", "w", encoding="utf-8") as f:
        f.write(report_md)
    print("\nЗвіт збережено у docs/hybrid_report.md")

if __name__ == "__main__":
    main()