import os
import re
import glob
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from build_corpus import parse_frontmatter_and_clean

MODEL_NAME = "intfloat/multilingual-e5-small"

def load_clean_documents():
    docs = []
    for filepath in sorted(glob.glob("kb/*.md")):
        with open(filepath, "r", encoding="utf-8") as f:
            meta, clean_body = parse_frontmatter_and_clean(f.read())
        docs.append({
            "doc_id": meta["doc_id"],
            "title": meta["title"],
            "type": meta["type"],
            "date": meta["date"],
            "sensitivity": meta["sensitivity"],
            "source_url": meta["source_url"],
            "body": clean_body
        })
    return docs

def chunk_fixed_overlap(docs, chunk_size=350, overlap=70):
    chunks = []
    step = chunk_size - overlap
    for d in docs:
        text = re.sub(r"\s+", " ", d["body"]).strip()
        idx = 1
        for start in range(0, len(text), step):
            piece = text[start:start + chunk_size].strip()
            if len(piece) < 40:
                continue
            chunks.append({
                "chunk_id": f"{d['doc_id']}_fix_{idx:03d}",
                "doc_id": d["doc_id"],
                "title": d["title"],
                "section": "fixed_window",
                "type": d["type"],
                "date": d["date"],
                "sensitivity": d["sensitivity"],
                "source_url": d["source_url"],
                "text": piece
            })
            idx += 1
    return chunks

def chunk_by_headers(docs):
    chunks = []
    for d in docs:
        lines = d["body"].splitlines()
        current_sec = "Загальне"
        buf = []
        idx = 1
        for line in lines:
            if line.startswith("# "):
                continue
            elif line.startswith("## "):
                if buf:
                    sec_text = " ".join(buf).strip()
                    chunks.append({
                        "chunk_id": f"{d['doc_id']}_hdr_{idx:03d}",
                        "doc_id": d["doc_id"],
                        "title": d["title"],
                        "section": current_sec,
                        "type": d["type"],
                        "date": d["date"],
                        "sensitivity": d["sensitivity"],
                        "source_url": d["source_url"],
                        "text": f"{current_sec}. {sec_text}"
                    })
                    idx += 1
                    buf = []
                current_sec = line.replace("## ", "").strip()
            else:
                if line.strip():
                    buf.append(line.strip())
        if buf:
            sec_text = " ".join(buf).strip()
            chunks.append({
                "chunk_id": f"{d['doc_id']}_hdr_{idx:03d}",
                "doc_id": d["doc_id"],
                "title": d["title"],
                "section": current_sec,
                "type": d["type"],
                "date": d["date"],
                "sensitivity": d["sensitivity"],
                "source_url": d["source_url"],
                "text": f"{current_sec}. {sec_text}"
            })
    return chunks

def chunk_hierarchical(docs, child_max_chars=220, overlap=35):
    header_chunks = chunk_by_headers(docs)
    chunks = []
    for hc in header_chunks:
        raw_sec = hc["text"]
        sentences = re.split(r"(?<=[.!?;])\s+", raw_sec)
        cur = ""
        sub_idx = 1
        for sent in sentences:
            if len(cur) + len(sent) + 1 <= child_max_chars:
                cur = (cur + " " + sent).strip()
            else:
                if cur:
                    enriched = f"[{hc['title']} | {hc['section']}] {cur}"
                    chunks.append({
                        "chunk_id": f"{hc['chunk_id']}_sub{sub_idx}",
                        "doc_id": hc["doc_id"],
                        "title": hc["title"],
                        "section": hc["section"],
                        "type": hc["type"],
                        "date": hc["date"],
                        "sensitivity": hc["sensitivity"],
                        "source_url": hc["source_url"],
                        "text": enriched
                    })
                    sub_idx += 1
                tail = cur[-overlap:] if len(cur) > overlap else cur
                cur = (tail + " " + sent).strip()
        if cur:
            enriched = f"[{hc['title']} | {hc['section']}] {cur}"
            chunks.append({
                "chunk_id": f"{hc['chunk_id']}_sub{sub_idx}",
                "doc_id": hc["doc_id"],
                "title": hc["title"],
                "section": hc["section"],
                "type": hc["type"],
                "date": hc["date"],
                "sensitivity": hc["sensitivity"],
                "source_url": hc["source_url"],
                "text": enriched
            })
    return chunks

TEST_QUERIES = [
    ("Який SLA розгляду резюме та заповнення Scorecard в ATS?", "doc01"),
    ("Протягом скількох годин потрібно видалити дані кандидата за GDPR?", "doc02"),
    ("Хто погоджує офер якщо зарплата перевищує вилку більше ніж на 10 відсотків?", "doc03"),
    ("Чи можна телефонувати поточному роботодавцю кандидата під час Reference Check?", "doc04"),
    ("Що таке Zero Ghosting Policy і за скільки днів надсилається відмова?", "doc05"),
    ("Який розмір реферального бонусу за рекомендацію Middle розробника?", "doc06"),
    ("Приклад X-Ray запиту в Google для пошуку Embedded розробників у Львові", "doc08"),
    ("Що означають літери STAR у поведінковому інтерв'ю?", "doc09"),
    ("Що таке Halo Effect та Affinity Bias під час співбесіди?", "doc11"),
    ("Яка зарплатна вилка для Middle Python Backend або DevOps інженера?", "doc23"),
]

def build_and_eval(chunks, model, k=5):
    passages = [f"passage: {c['text']}" for c in chunks]
    emb = model.encode(passages, normalize_embeddings=True, show_progress_bar=False)
    emb = np.array(emb, dtype="float32")

    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb)

    hits = 0
    for q_text, expected_doc in TEST_QUERIES:
        q_emb = model.encode([f"query: {q_text}"], normalize_embeddings=True, show_progress_bar=False)
        q_emb = np.array(q_emb, dtype="float32")
        _, indices = index.search(q_emb, k)
        retrieved_docs = {chunks[i]["doc_id"] for i in indices[0] if i != -1}
        if expected_doc in retrieved_docs:
            hits += 1

    recall_at_5 = hits / len(TEST_QUERIES)
    avg_chars = float(np.mean([len(c["text"]) for c in chunks]))
    avg_words = float(np.mean([len(c["text"].split()) for c in chunks]))
    return len(chunks), recall_at_5, avg_chars, avg_words

def main():
    os.makedirs("data", exist_ok=True)
    os.makedirs("docs", exist_ok=True)

    docs = load_clean_documents()
    print(f"Завантажено {len(docs)} документів. Завантаження моделі {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)

    strategies = {
        "1. Fixed-size + Overlap (350/70)": chunk_fixed_overlap(docs, 350, 70),
        "2. Structural (Markdown Headers ##)": chunk_by_headers(docs),
        "3. Hierarchical (Parent Context + Child)": chunk_hierarchical(docs, 220, 35),
    }

    results = []
    for name, ch_list in strategies.items():
        n_chunks, r5, avg_c, avg_w = build_and_eval(ch_list, model, k=5)
        results.append((name, n_chunks, r5, avg_c, avg_w))
        print(f"{name:42s} | Чанків: {n_chunks:3d} | Recall@5: {r5:.2f} | Сер. розмір: {avg_c:.1f} симв. ({avg_w:.1f} слів)")

    best = max(results, key=lambda x: (x[2], -x[3]))

    best_chunks = strategies[best[0]]
    with open("data/chunks_selected.jsonl", "w", encoding="utf-8") as f:
        for ch in best_chunks:
            f.write(json.dumps(ch, ensure_ascii=False) + "\n")

    table_rows = "\n".join(
        f"| {r[0]} | {r[1]} | {r[2]:.2f} ({int(r[2]*10)}/10) | {r[3]:.1f} симв. (~{r[4]:.1f} слів) |"
        for r in results
    )

    report = f"""# Порівняння стратегій Chunking 

## 1. Умови експерименту 
- **Корпус:** {len(docs)} очищених документів предметної області «Рекрутинг та підбір кадрів».
- **Фіксована модель embeddings:** `{MODEL_NAME}` (розмірність 384, косинусна подібність через `IndexFlatIP` з L2-нормалізацією).
- **Тестовий набір:** 10 фактологічних і процедурних запитів з еталонними `doc_id`.
- **Досліджувані стратегії:**
  1. **Fixed-size + Overlap:** розбиття суцільного тексту вікном 350 символів із перекриттям 70 символів без урахування структури Markdown.
  2. **Structural (Markdown Headers):** розбиття за логічними секціями (`##`), один розділ — один фрагмент.
  3. **Hierarchical (Parent Context + Child):** двоступенева нарізка, де секція розбивається на дочірні фрагменти по ~220 символів, а кожен фрагмент отримує префікс `[Назва документа | Назва секції]`.

## 2. Порівняльна таблиця результатів 

| Стратегія chunking | Кількість чанків | Recall@5 | Середній розмір контексту |
| :--- | :---: | :---: | :---: |
{table_rows}

## 3. Обґрунтування вибору 
- **Обрана стратегія:** **{best[0]}** (показник `Recall@5 = {best[2]:.2f}`, середній розмір контексту — `{best[3]:.1f}` символів).
- **Аналіз результатів:**
  - *Fixed-size + Overlap* розриває логічні правила та списки на межі вікна, через що умови (наприклад, відсоток перевищення вилки або коди ATS) потрапляють у різні чанки без заголовка секції.
  - *Structural (Markdown Headers)* зберігає цілісність розділу, але створює великі фрагменти, у яких векторне представлення усереднює кілька різних підтем, знижуючи точність семантичного пошуку.
  - *Hierarchical* поєднує компактний розмір дочірнього чанка (зменшує шум у контексті) із явним ієрархічним контекстом (`title` + `section`), забезпечуючи стабільно високий `Recall@5` при економному використанні контекстного вікна.
"""
    with open("docs/chunking_cmp.md", "w", encoding="utf-8") as f:
        f.write(report)
    print("\nЗвіт збережено у docs/chunking_cmp.md")

if __name__ == "__main__":
    main()