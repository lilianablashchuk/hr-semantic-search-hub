<!-- NAV_HEADER: Home | Knowledge Base | Recruiting Hub | Login -->
---
doc_id: doc08
title: Гайд із Boolean Search та X-Ray сорсингу кандидатів
type: guideline
date: 2026-09-20
sensitivity: public
source_url: https://hub.lnu.edu.ua/recruitment/kb/08_boolean_xray_sourcing_guide.md
---

[Навігація: Головна > База знань > Рекрутинг та підбір кадрів]

# Гайд із Boolean Search та X-Ray сорсингу кандидатів

## 1. Базові оператори Boolean Search
Boolean Search дозволяє формувати точні пошукові запити в LinkedIn Recruiter, GitHub та пошукових системах за допомогою логічних операторів. Оператор AND вимагає наявності обох термінів; OR розширює пошук синонімами; NOT (або мінус у Google) виключає нерелевантні слова; лапки "" фіксують точну фразу; дужки () визначають пріоритет виконання операцій.

## 2. Приклади пошукових рядків для технічних ролей
Для пошуку Python-розробника з досвідом асинхронних фреймворків та контейнеризації використовується запит: `("Python Developer" OR "Backend Engineer") AND (FastAPI OR Django) AND (Docker OR Kubernetes) NOT (Recruiter OR HR OR Intern)`. Для пошуку QA Automation спеціаліста: `("QA Automation" OR SDET) AND (Playwright OR Selenium OR Cypress) AND (Python OR TypeScript)`.

## 3. Техніка X-Ray пошуку через Google
X-Ray пошук використовує оператор `site:` для індексації публічних профілів без обмежень внутрішніх фільтрів платформи. Приклад X-Ray запиту для пошуку розробників у Львові в LinkedIn: `site:linkedin.com/in/ ("Lviv" OR "Львів") ("Embedded" OR "STM32" OR "Firmware") -intitle:jobs`. Для пошуку на GitHub за мовою програмування та локацією: `site:github.com "Lviv" "Python" "repositories" -site:github.com/topics`.

---
Page 1 of 1 | Internal Knowledge Hub LNU | © 2026 All rights reserved | Print this page
