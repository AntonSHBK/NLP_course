# 1. NLP_course
Description

Создание виртуального окружения:
```bush
pip install virtualenv
virtualenv .venv -p python3.10
.venv\scripts\activate
pip install -r requirements.txt
```
Альтернативный вариант:
```bush
python -m venv .venv
```

Если команда activate генерирует сообщение "Activate.ps1 не имеет цифровой подписи. Вы не можете запустить этот скрипт в текущей системе.", затем вам необходимо временно изменить политику выполнения PowerShell, чтобы разрешить запуск сценариев (см. О политиках выполнения в документации PowerShell):
```bush
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
```

установить права доступа для сценариев, указать Да
```bush
Set-ExecutionPolicy RemoteSigned
```


Video course Autumn:
- [1 Lecture](https://ods.ai/tracks/nlp-course-autumn-23/blocks/d43f510d-dda4-45fc-9460-28c23f0790f6);
- [2 Lecture](https://ods.ai/tracks/nlp-course-autumn-23/blocks/9e3cb83a-eac5-4da1-aa72-dbc3dfd78e9d);
- [3 Lecture](https://ods.ai/tracks/nlp-course-autumn-23/blocks/7b6bbe9b-3fdb-44ba-bca7-910f005e5a5a);
- [4 Lecture](https://ods.ai/tracks/nlp-course-autumn-23/blocks/51ea81de-102e-4288-ad08-8fca75b875d3);
- [5 Lecture](https://ods.ai/tracks/nlp-course-autumn-23/blocks/d6ed57bb-a97f-4c6b-a55d-64aea4b54d0a);
- [6 Lecture](https://ods.ai/tracks/nlp-course-autumn-23/blocks/f9d494e9-1ec4-4d35-897c-2142f36a7454);
- []();
- [8 Lecture](https://www.youtube.com/watch?v=B17UT63YWZc&t=3s);
- []();
- []();

Data:
* []()
* []()




[Latex devcontainer:](https://github.com/AntonSHBK/latex_docker_base)
```
git clone https://github.com/AntonSHBK/latex_docker_base
```


# Ответы государственных органов на обращения граждан

## Цель проекта

Цель данного проекта — разработать и обучить модель на базе GPT-2 для автоматизации процесса создания ответов от представителей государственных органов на обращения граждан. Модель должна обеспечить высокую точность и релевантность ответов, учитывая контекст запроса и специфику вопроса.

## Описание проекта

Проект включает в себя разработку модифицированной модели GPT-2, которая дополнена возможностью учета типа запроса, его категории и других специфических параметров, обеспечивая тем самым более точное и осмысленное взаимодействие с пользователем. Модель обучается на специально подготовленном наборе данных, содержащем реальные обращения граждан и ответы органов власти.

### Основные характеристики:
- **Базовая модель**: GPT-2 (ai-forever/rugpt3small_based_on_gpt2)
- **Методы генерации**: Argmax, Temperature Sampling, Top-k Sampling, Top-p Sampling
- **Основные метрики валидации**: BLEU, ROUGE, METEOR

## Интерактивные элементы

Для демонстрации работы модели предлагается интерактивный блок, где пользователи могут вводить свои запросы, и модель будет генерировать ответы в реальном времени.

```html
<!-- Пример интерактивного блока для ввода запроса -->
Введите ваш запрос:
<input type="text" id="user_query">
<button onclick="generateResponse()">Получить ответ</button>
<div id="model_response"></div>
<script>
function generateResponse() {
    const userQuery = document.getElementById('user_query').value;
    // Здесь будет код для обращения к модели и получения ответа
    document.getElementById('model_response').innerText = 'Здесь будет ответ модели...';
}
</script>
