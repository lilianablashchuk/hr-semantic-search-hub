import os
import time
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

def load_chunks(path="data/chunks_selected.jsonl"):
    if not os.path.exists(path):
        path = "data/corpus.jsonl"
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks

EVAL_QUERIES_20 = [
    ("Який SLA первинного розгляду резюме рекрутером?", "doc01", "uk"),
    ("Протягом скількох годин потрібно видалити персональні дані кандидата за GDPR?", "doc02", "uk"),
    ("Хто затверджує оффер якщо компенсація перевищує Salary Band більше ніж на 10%?", "doc03", "uk+en"),
    ("Чи дозволено телефонувати поточному роботодавцю під час Reference Check?", "doc04", "uk+en"),
    ("Що означає принцип Zero Ghosting Policy при відмові кандидату?", "doc05", "uk+en"),
    ("Який бонус виплачується за рекомендацію Middle розробника за реферальною програмою?", "doc06", "uk"),
    ("Як забезпечується сліпий скринінг тестових завдань для уникнення упереджень?", "doc07", "uk"),
    ("Boolean search query string for FastAPI and Docker backend developer", "doc08", "en"),
    ("Як розшифровується модель STAR для поведінкового інтерв'ю?", "doc09", "uk+en"),
    ("Що оцінює інтерв'юер під час live-coding сесії крім синтаксису?", "doc10", "uk+en"),
    ("Поясніть різницю між Halo Effect та Affinity Bias на співбесіді", "doc11", "uk+en"),
    ("Скільки пунктів має бути у списку Must-have в описі вакансії?", "doc12", "uk+en"),
    ("Скільки follow-up повідомлень дозволено надсилати пасивному кандидату?", "doc13", "uk+en"),
    ("Що таке Total Rewards і як використовувати їх у переговорах про зарплату?", "doc14", "uk+en"),
    ("How is Offer Acceptance Rate calculated and what is the target benchmark?", "doc15", "en"),
    ("Які вимоги до баз даних для вакансії Middle Python Backend Developer?", "doc16", "uk+en"),
    ("Які інструменти тестування API потрібні для Junior QA Engineer?", "doc17", "uk+en"),
    ("Вимоги до мікроконтролерів та інтерфейсів для Embedded Firmware інженера", "doc20", "uk+en"),
    ("Яка зарплатна вилка для Senior QA Automation або Lead Data Analyst?", "doc23", "uk+en"),
    ("Що означає код закриття картки кандидата REJ-COMP в системі ATS?", "doc25", "uk+en"),
]

class HNSWVectorStore:
    def __init__(self, model_name: str, use_e5_prefix: bool = False, M: int = 16, ef_construction: int = 64, ef_search: int = 32):
        self.model_name = model_name
        self.use_e5_prefix = use_e5_prefix
        self.M = M
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        self.index = None
        self.records = []
        self.deleted_ids = set()
        self.chunk_id_to_pos = {}

    def _format_passages(self, texts):
        return [f"passage: {t}" for t in texts] if self.use_e5_prefix else texts

    def _format_queries(self, texts):
        return [f"query: {t}" for t in texts] if self.use_e5_prefix else texts

    def build(self, chunks):
        t0 = time.perf_counter()
        self.records = list(chunks)
        self.deleted_ids = set()
        self.chunk_id_to_pos = {c["chunk_id"]: i for i, c in enumerate(self.records)}

        texts = self._format_passages([c["text"] for c in self.records])
        vectors = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        vectors = np.array(vectors, dtype="float32")

        self.index = faiss.IndexHNSWFlat(self.dim, self.M, faiss.METRIC_INNER_PRODUCT)
        self.index.hnsw.efConstruction = self.ef_construction
        self.index.hnsw.efSearch = self.ef_search
        self.index.add(vectors)

        return time.perf_counter() - t0

    def search(self, query: str, k: int = 5):
        t0 = time.perf_counter()
        q_vec = self.model.encode(self._format_queries([query]), normalize_embeddings=True, show_progress_bar=False)
        q_vec = np.array(q_vec, dtype="float32")

        fetch_k = min(len(self.records), k + len(self.deleted_ids) + 5)
        scores, indices = self.index.search(q_vec, fetch_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            rec = self.records[idx]
            if rec["chunk_id"] in self.deleted_ids:
                continue
            results.append((float(score), rec))
            if len(results) == k:
                break

        latency_ms = (time.perf_counter() - t0) * 1000.0
        return results, latency_ms

    def delete_by_doc_id(self, doc_id: str) -> int:
        removed = 0
        for rec in self.records:
            if rec["doc_id"] == doc_id and rec["chunk_id"] not in self.deleted_ids:
                self.deleted_ids.add(rec["chunk_id"])
                removed += 1
        return removed

    def upsert_chunks(self, new_chunks: list) -> int:
        for ch in new_chunks:
            if ch["chunk_id"] in self.chunk_id_to_pos:
                self.deleted_ids.add(ch["chunk_id"])

        texts = self._format_passages([c["text"] for c in new_chunks])
        vectors = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        vectors = np.array(vectors, dtype="float32")

        start_pos = len(self.records)
        self.index.add(vectors)
        for offset, ch in enumerate(new_chunks):
            self.records.append(ch)
            self.chunk_id_to_pos[ch["chunk_id"]] = start_pos + offset

        return len(new_chunks)

def evaluate_store(store: HNSWVectorStore, queries):
    hits = 0
    latencies = []
    lang_hits = {"uk": [0, 0], "en": [0, 0], "uk+en": [0, 0]}

    for q_text, expected_doc, lang in queries:
        res, lat_ms = store.search(q_text, k=5)
        latencies.append(lat_ms)
        retrieved_docs = {r[1]["doc_id"] for r in res}
        is_hit = int(expected_doc in retrieved_docs)
        hits += is_hit
        lang_hits[lang][0] += is_hit
        lang_hits[lang][1] += 1

    recall_5 = hits / len(queries)
    avg_lat = float(np.mean(latencies))
    p95_lat = float(np.percentile(latencies, 95))
    return recall_5, avg_lat, p95_lat, lang_hits

def main():
    os.makedirs("docs", exist_ok=True)
    chunks = load_chunks()
    print(f"Завантажено {len(chunks)} фрагментів для індексації в HNSW.")

    M = 16
    EF_CONSTRUCTION = 64
    EF_SEARCH = 32

    models_cfg = [
        ("intfloat/multilingual-e5-small", True),
        ("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", False),
    ]

    summary = []
    stores = {}

    for m_name, use_prefix in models_cfg:
        print(f"\n Побудова HNSW індексу для {m_name}")
        store = HNSWVectorStore(m_name, use_e5_prefix=use_prefix, M=M, ef_construction=EF_CONSTRUCTION, ef_search=EF_SEARCH)
        build_sec = store.build(chunks)
        r5, avg_ms, p95_ms, lang_stats = evaluate_store(store, EVAL_QUERIES_20)
        summary.append((m_name, store.dim, build_sec, r5, avg_ms, p95_ms, lang_stats))
        stores[m_name] = store
        print(f"Час побудови: {build_sec:.2f} с | Recall@5: {r5:.2f} | Avg latency: {avg_ms:.2f} мс | P95: {p95_ms:.2f} мс")

    test_store = stores["intfloat/multilingual-e5-small"]
    target_docs = ["doc01", "doc05", "doc10", "doc16", "doc23"]
    deleted_total = sum(test_store.delete_by_doc_id(d) for d in target_docs)

    res_after_del, _ = test_store.search("Яка зарплатна вилка для Senior QA Automation?", k=5)
    found_doc23_after_del = any(r[1]["doc_id"] == "doc23" for r in res_after_del)

    updated_chunks = [
        {
            "chunk_id": "doc01_upd_001",
            "doc_id": "doc01",
            "title": "Регламент етапів воронки рекрутингу та SLA найму (Оновлено)",
            "section": "Оновлений SLA",
            "text": "[Оновлення SLA] Максимальний час розгляду резюме скорочено до 1 робочого дня."
        },
        {
            "chunk_id": "doc05_upd_001",
            "doc_id": "doc05",
            "title": "Стандарти комунікації при відмові (Оновлено)",
            "section": "Zero Ghosting",
            "text": "[Zero Ghosting] Обов'язковий лист-відмова надсилається протягом 48 годин."
        },
        {
            "chunk_id": "doc10_upd_001",
            "doc_id": "doc10",
            "title": "Гайд із технічного скринінгу (Оновлено)",
            "section": "Live-coding",
            "text": "[Live-coding] Дозволено використання офіційної документації мови під час задачі."
        },
        {
            "chunk_id": "doc16_upd_001",
            "doc_id": "doc16",
            "title": "Профіль: Middle Python Backend Developer (Оновлено)",
            "section": "Бази даних",
            "text": "[Must-have] Обов'язковий досвід роботи з PostgreSQL 16, PostGIS та асинхронним SQLAlchemy."
        },
        {
            "chunk_id": "doc23_upd_001",
            "doc_id": "doc23",
            "title": "Зарплатні вилки 2026 (Оновлено)",
            "section": "QA Automation",
            "text": "[Salary Bands Q4 2026] Оновлена вилка для Senior QA Automation становить 3 800 – 5 100 USD Net."
        }
    ]
    upserted_count = test_store.upsert_chunks(updated_chunks)
    res_after_upd, _ = test_store.search("Яка зарплатна вилка для Senior QA Automation?", k=5)
    top1_after_upd = res_after_upd[0][1]["chunk_id"] if res_after_upd else "none"

    print(f"\nПеревірка Upsert/Delete на 5 документах ({target_docs}):")
    print(f"- Видалено старих чанків (tombstones): {deleted_total}")
    print(f"- Чи знайдено doc23 одразу після delete: {found_doc23_after_del} (очікується False)")
    print(f"- Додано оновлених чанків (upsert): {upserted_count}")
    print(f"- Top-1 чанк після upsert для запиту про вилку Senior QA: {top1_after_upd}")

    rows = ""
    for m_name, dim, b_sec, r5, avg_ms, p95_ms, l_stats in summary:
        uk_acc = f"{l_stats['uk'][0]}/{l_stats['uk'][1]}"
        en_acc = f"{l_stats['en'][0]}/{l_stats['en'][1]}"
        mix_acc = f"{l_stats['uk+en'][0]}/{l_stats['uk+en'][1]}"
        rows += f"| `{m_name}` | {dim} | {b_sec:.2f} с | **{r5:.2f}** ({int(r5*20)}/20) | {uk_acc} / {en_acc} / {mix_acc} | {avg_ms:.2f} мс | {p95_ms:.2f} мс |\n"

    report_md = f"""# Порівняння моделей Embeddings та векторного індексу HNSW

## 1. Конфігурація векторного індексу FAISS HNSW (Критерій C1, C2)
- **Тип індексу:** `faiss.IndexHNSWFlat` (метрика `METRIC_INNER_PRODUCT` поверх L2-нормалізованих векторів).
- **Зафіксовані гіперпараметри:** `M = {M}`, `efConstruction = {EF_CONSTRUCTION}`, `efSearch = {EF_SEARCH}`.
- **Кількість проіндексованих фрагментів:** {len(chunks)}.

## 2. Порівняльна таблиця якості та затримки на 20 запитах 

| Модель Embeddings | Розмірність | Час побудови | Recall@5 (20 запитів) | Мови (UK / EN / Mix) | Avg Latency | P95 Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
{rows}

## 3. Процедура оновлення індексу (Upsert / Delete) та перевірка
- **Delete (`delete_by_doc_id`):** логічне видалення через множину `deleted_ids` (tombstone) з фільтрацією на етапі пошуку.
- **Upsert (`upsert_chunks`):** позначення старих чанків у `deleted_ids` та інкрементальне додавання нових векторів через `index.add()`.
- **Перевірка на 5 документах (`doc01`, `doc05`, `doc10`, `doc16`, `doc23`):** видалено {deleted_total} старих фрагментів (`found_doc23_after_del = {found_doc23_after_del}`), додано {upserted_count} оновлених фрагментів, Top-1 результат після оновлення — `{top1_after_upd}`.
"""
    with open("docs/embeddings_cmp.md", "w", encoding="utf-8") as f:
        f.write(report_md)
    print("\nЗвіт збережено у docs/embeddings_cmp.md")

if __name__ == "__main__":
    main()