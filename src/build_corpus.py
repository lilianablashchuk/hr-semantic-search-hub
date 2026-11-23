import os
import re
import json
import glob
from collections import Counter

def parse_frontmatter_and_clean(raw_text: str):
    meta = {}
    fm_match = re.match(r"^<!--.*?-->\s*---\n(.*?)\n---\n(.*)$", raw_text, flags=re.DOTALL)
    if fm_match:
        fm_block, body = fm_match.group(1), fm_match.group(2)
        for line in fm_block.strip().splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                meta[k.strip()] = v.strip()
    else:
        body = raw_text

    body = re.sub(r"<!--.*?-->", "", body, flags=re.DOTALL)
    body = re.sub(r"^\[Навігація:.*?\]\s*$", "", body, flags=re.MULTILINE)
    body = re.sub(r"^---\s*\nPage \d+ of \d+.*$", "", body, flags=re.MULTILINE | re.DOTALL)
    body = re.sub(r"\n{3,}", "\n\n", body).strip()

    return meta, body

def split_into_initial_chunks(body: str, max_chars: int = 125, overlap: int = 25):
    lines = body.splitlines()
    current_section = "Вступ"
    sections = []
    buffer = []

    for line in lines:
        if line.startswith("# "):
            continue
        elif line.startswith("## "):
            if buffer:
                sections.append((current_section, "\n".join(buffer).strip()))
                buffer = []
            current_section = line.replace("## ", "").strip()
        else:
            if line.strip():
                buffer.append(line.strip())

    if buffer:
        sections.append((current_section, "\n".join(buffer).strip()))

    chunks = []
    for sec_title, sec_text in sections:
        units = re.split(r"(?:\n+|(?<=[.!?;])\s+)", sec_text)
        current_chunk = ""
        for unit in units:
            unit = unit.strip()
            if not unit:
                continue
            if len(current_chunk) + len(unit) + 1 <= max_chars:
                current_chunk = (current_chunk + " " + unit).strip()
            else:
                if current_chunk:
                    chunks.append((sec_title, current_chunk))
                tail = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                current_chunk = (tail + " " + unit).strip() if tail else unit
        if current_chunk:
            chunks.append((sec_title, current_chunk))

    return chunks

def main():
    os.makedirs("data", exist_ok=True)
    os.makedirs("docs", exist_ok=True)

    md_files = sorted(glob.glob("kb/*.md"))
    all_records = []
    sensitivity_counts = Counter()
    type_counts = Counter()

    for filepath in md_files:
        with open(filepath, "r", encoding="utf-8") as f:
            raw_text = f.read()

        meta, clean_body = parse_frontmatter_and_clean(raw_text)
        doc_chunks = split_into_initial_chunks(clean_body, max_chars=125, overlap=25)

        for idx, (section, text) in enumerate(doc_chunks, start=1):
            record = {
                "chunk_id": f"{meta['doc_id']}_ch{idx:03d}",
                "doc_id": meta["doc_id"],
                "title": meta["title"],
                "section": section,
                "type": meta["type"],
                "date": meta["date"],
                "sensitivity": meta["sensitivity"],
                "source_url": meta["source_url"],
                "text": text
            }
            all_records.append(record)
            sensitivity_counts[meta["sensitivity"]] += 1
            type_counts[meta["type"]] += 1

    corpus_path = "data/corpus.jsonl"
    with open(corpus_path, "w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Оброблено документів: {len(md_files)}")
    print(f"Збережено фрагментів у {corpus_path}: {len(all_records)}")
    print(f"Розподіл за sensitivity: {dict(sensitivity_counts)}")
    print(f"Розподіл за type: {dict(type_counts)}")

    card_md = f"""# Картка корпусу (Corpus Card): Рекрутинг та підбір кадрів

## 1. Загальна інформація
- **Назва корпусу:** Domain Knowledge Hub — Recruiting & Talent Acquisition KB
- **Тематична область:** Рекрутинг та підбір кадрів (організація знань про пошук, оцінювання та відбір кандидатів)
- **Кількість документів:** {len(md_files)}
- **Кількість фрагментів (чанків):** {len(all_records)}
- **Основна мова:** Українська (`uk`) із використанням усталеної англомовної індустріальної термінології (`en`: ATS, GDPR, Boolean Search, STAR, SLA, EVP, DevOps, QA, Embedded тощо).
- **Часове охоплення документів:** травень – жовтень 2026 року (`2026-05-10` — `2026-10-25`).

## 2. Джерела та ліцензії 
- **Походження даних:** Синтезовані та верифіковані навчально-методичні документи предметної області HR/Recruiting, розроблені в межах курсу «Сховища знань та управління знаннями» (ЛНУ ім. Івана Франка, 2026) на основі відкритих стандартів SHRM, GDPR (Регламент ЄС 2016/679) та практик IT-рекрутингу.
- **Ліцензія:** CC BY-NC-SA 4.0 (Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International) — вільне використання для навчальних та дослідницьких цілей із зазнам авторства.
- **Персональні дані:** Відсутні (усі імена, ставки та політики є узагальненими моделями; реальні персональні дані кандидатів не використовувалися).

## 3. Структура метаданих та рівні доступу `sensitivity` (Критерій C4)
Кожен запис у `data/corpus.jsonl` містить обов'язкові поля: `chunk_id`, `doc_id`, `title`, `section`, `type`, `date`, `sensitivity`, `source_url`, `text`.

| Рівень `sensitivity` | Опис рівня доступу | Кількість фрагментів |
| :--- | :--- | :---: |
| `public` | Публічні гайди, профілі вакансій, глосарій, політика DEI та стандарти фідбеку | {sensitivity_counts['public']} |
| `internal` | Внутрішні регламенти воронки, GDPR-процедури, скрипти аутрічу, матриця грейдів, Scorecard | {sensitivity_counts['internal']} |
| `restricted` | Конфіденційні фінансові документи: зарплатні вилки (`Salary Bands 2026`) та регламент погодження оферів | {sensitivity_counts['restricted']} |

**Розподіл за типом документа (`type`):**
- `guideline` (методичні гайди): {type_counts['guideline']} фрагментів
- `policy` (внутрішні регламенти та політики): {type_counts['policy']} фрагментів
- `job_profile` (профілі посад та вимоги): {type_counts['job_profile']} фрагментів
- `reference` (довідники, грейди, вилки, коди ATS): {type_counts['reference']} фрагментів

## 4. Правила очищення від шуму 
Під час побудови корпусу з вихідних файлів `kb/*.md` автоматично застосовано такі детерміновані правила очищення (реалізовані в `src/build_corpus.py`):
1. **Видалення технічних HTML-коментарів:** регулярний вираз `<!--.*?-->` прибирає службові теги шаблонізатора.
2. **Видалення навігаційних елементів (Breadcrumbs):** рядки формату `^[Навігація:.*?]$` видаляються, щоб навігаційні слова («Головна», «База знань») не спотворювали частоти термінів у BM25 та векторних представленнях.
3. **Видалення нижніх колонтитулів (Footers):** блок `^---\\s*\\nPage \\d+ of \\d+.*$` очищає номери сторінок, копірайти та кнопки друку (`Print this page`).
4. **Нормалізація пробільних символів:** послідовності з трьох і більше переносів рядка (`\\n{{3,}}`) зводяться до стандартного розділювача абзаців (`\\n\\n`).

## 5. Відомі обмеження та прогалини корпусу
- **Доменна спеціалізація:** Профілі посад та матриці грейдів орієнтовані переважно на IT-сектор (Backend, QA, DevOps, Data, Embedded) і не охоплюють специфіку масового найму (Blue-collar) або державної служби.
- **Синтетичний характер зарплатних вилок:** Числові діапазони в `doc23` є модельними оцінками станом на осінь 2026 року і не є реальною комерційною таємницею конкретної компанії.
- **Відсутність мультимодальних даних:** Корпус містить лише текстові Markdown-документи без PDF-сканів резюме чи таблиць у форматі XLSX.
"""
    with open("docs/corpus_card.md", "w", encoding="utf-8") as f:
        f.write(card_md)
    print("Картку корпусу збережено у docs/corpus_card.md")

if __name__ == "__main__":
    main()