# Ответы государственных органов на обращения граждан

Работа выполнена в рамках прохождения курса [Natural Language Processing course (stream 6, spring 2024)](https://ods.ai/tracks/nlp-course-spring-2024).

- [Jupyter notebook](project/gpt2_small.ipynb)
- [Основной отчёт о ходе выполнения проекта в формате pdf](https://github.com/AntonSHBK/NLP_course/tree/main/docs/NLP_Course.pdf)
- [Директория проекта](https://github.com/AntonSHBK/NLP_course/tree/main/project)

## Цель проекта

Цель данного проекта — разработать и обучить модель на базе GPT-2 для автоматизации процесса создания ответов от представителей государственных органов на обращения граждан. Модель должна обеспечить высокую точность и релевантность ответов, учитывая контекст запроса и специфику вопроса.

## Описание проекта

Проект включает в себя разработку модифицированной модели GPT-2, которая дополнена возможностью учета типа запроса, его категории и других специфических параметров, обеспечивая тем самым более точное и осмысленное взаимодействие с пользователем. Модель обучается на специально подготовленном наборе данных, содержащем реальные обращения граждан и ответы органов власти.

### Основные характеристики:
- **Базовая модель**: GPT-2 (ai-forever/rugpt3small_based_on_gpt2)
- **Методы генерации**: Temperature Sampling, Top-k Sampling, Top-p Sampling
- **Основные метрики валидации**: BLEU, ROUGE, METEOR

### Описание проектируемых моделей:
- [модель](project/model.py);
- [тренер](project/trainer.py);
- [метрики валидации](project/evaluator.py);
- [датасет](project/custom_data.py);
- [утилиты](project/utils.py).

## Датасет

Для доступа к датасету необходимо обратиться с соответствующим сообщением по электронной почте (`anton42@yandex.ru`).

## Зависимости

Для работы проекта требуется Python версии 3.10. Все необходимые библиотеки и их версии перечислены в файле `requirements.txt`, который можно использовать для установки зависимостей.

### Установка зависимостей

Для установки всех зависимостей выполните следующую команду в вашем терминале:

```bash
pip install -r requirements.txt
```
