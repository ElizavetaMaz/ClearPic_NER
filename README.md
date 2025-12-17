# ClearPic_NER

## 📌 Краткое описание проекта
ClearPic NER Preprocessing классы и методы для NER обработки СМИ на азербайджанском языке.

## 📂 Структура проекта
```
ClearPic_NER/
├──az_stemming/
│   └── lemmatizator.py         # Класс лемматизор для азербайджанского языка
├── az_ner_news.py              # Основной класс ExtractedEntities
├── ExtractEntitiesToJson.py    # Скрипт для обработки статей из MongoDB c сохранением JSON в папку output/
├── requirements.txt            # Зависимости проекта
└── config/                     # Конфигурационные файлы
    ├── labels_mapping.json     # Маппинг меток NER модели
    ├── types_city_country.json # Типы локаций (город/страна и т.д.)
    └── types_org.json          # Типы организаций

```
## 🚀 Быстрый старт (проекта)

  1. Клонировать репозиторий: git clone https://github.com/ElizavetaMaz/ClearPic_NER
  2. Перейти в репозиторий: cd ClearPic_NER
  4. Установка зависимостей: pip install -r requirements.txt
  5. Настройка конфигурации: cоздайте файл .env в корне проекта .env
      * MONGO_URI=mongodb+srv://<username>:<password>@cluster.mongodb.net/ (обязательно)
      * NER_PATH (необязательно)
      * TYPES_LOC_PATH (необязательно)
      * LABELS_PATH (необязательно)
      * ORGS_TYPES_PATH (необязательно)
      * OUTPUT_PATH (необязательно)
  6. Подготовка модели NER: модель, которая использовалась в этом проекте (https://disk.360.yandex.ru/d/vVpFqLGsOYLgwA). Нужно распокавать архив в корне проекта.
  7. Запуск: python ExtractEntitiesToJson.py (c загрузкой в Json) или python ExtractEntities.py (c загрузкой в MongoDB)

## 📖 Использование

```
from az_ner_news import ExtractedEntities
import json

# Инициализация экстрактора
extractor = ExtractedEntities(
    ner_model_path="models/xlm-roberta-ner",
    labels_path="config/labels_mapping.json",
    types_loc_path="config/location_types.json",
    org_types_path="config/organization_types.json"
)
  
# Обработка текста
text = "Prezident İlham Əliyev Bakı şəhərində yeni zavodun açılışında iştirak edib."
entities, remaining_text = extractor.extract_from_text(text)

print(json.dumps(entities, indent=4, ensure_ascii=False))
```

