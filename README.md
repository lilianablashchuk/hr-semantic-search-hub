# Лабораторна робота 10
**Тематична область:** Рекрутинг та підбір кадрів (пошук, оцінювання та відбір кандидатів).  
**Тема:** Векторне сховище та семантичний пошук 

## 1. Середовище та залежності
- **Python:** 3.11 / 3.12 (`seed = 42`)
- **Бібліотеки:** `sentence-transformers`, `faiss-cpu`, `rank_bm25`, `scikit-learn`, `numpy`, `pandas`

Встановлення залежностей:
```bash
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install sentence-transformers faiss-cpu rank_bm25 scikit-learn numpy pandas
