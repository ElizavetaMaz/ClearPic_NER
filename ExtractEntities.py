import pandas as pd
import pymongo
from dotenv import load_dotenv
import os
from az_ner_news import ExtractedEntities
from datetime import datetime

load_dotenv()

# Загрузка переменных окружения
MONGO_URI = os.getenv("MONGO_URI")
NER_PATH = os.getenv("NER_PATH", "./ner_azerbaijan_local")
TYPES_LOC_PATH = os.getenv("TYPES_LOC_PATH", "config/types_city_country.json")
LABELS_PATH = os.getenv("LABELS_PATH", "config/label_mapping.json")
ORGS_TYPES_PATH = os.getenv("ORGS_TYPES_PATH", "config/types_org.json")

# Настройки
SOURCE_COLLECTION = "articles"  # Исходная коллекция
TARGET_COLLECTION = "articles_with_ner"  # Новая коллекция для обработанных статей
BATCH_SIZE = 100  # Размер батча для обработки


def create_processed_collection(client, db_name="az_articles", collection_name=TARGET_COLLECTION):
    """
    Создает новую коллекцию для обработанных статей, избегая проверки listCollections.
    """
    db = client[db_name]
    collection = db[collection_name]

    # Пытаемся создать индекс. Если коллекции не существует, она создастся при первой вставке.
    # Создание индекса помогает убедиться, что коллекция правильно инициализирована.
    try:
        collection.create_index([("processed_date", pymongo.ASCENDING)])
        print(f"Используется существующая или новая коллекция: {collection_name}")
    except pymongo.errors.OperationFailure as e:
        # Эта ошибка означает, что у пользователя также нет прав createIndex.
        # Коллекция все равно будет создана при первой вставке данных.
        print(f"Примечание: Не удалось создать индекс. Коллекция будет создана при первой вставке. Ошибка: {e}")

    return collection

def save_batch_to_mongodb(collection, articles):
    """
    Сохраняет батч статей в MongoDB.

    Args:
        collection: Коллекция MongoDB
        articles: Список статей для сохранения
    """
    try:
        if articles:
            result = collection.insert_many(articles, ordered=False)
            print(f"  Сохранено {len(result.inserted_ids)} статей в MongoDB")
    except pymongo.errors.BulkWriteError as e:
        print(f"  Предупреждение: некоторые документы не сохранены: {str(e)}")
    except Exception as e:
        print(f"  Ошибка при сохранении в MongoDB: {str(e)}")

def process_and_save_articles(client, extractor, limit=None):
    """
    Обрабатывает статьи и сохраняет в новую коллекцию.

    Args:
        client: Клиент MongoDB
        extractor: Экстрактор сущностей
        limit: Ограничение количества статей (None для всех)

    Returns:
        dict: Статистика обработки
    """
    db = client["az_articles"]

    # Получаем статьи для обработки
    query = {}
    if limit:
        all_articles = list(db[SOURCE_COLLECTION].find(query).limit(limit))
    else:
        all_articles = list(db[SOURCE_COLLECTION].find(query))

    print(f"Найдено {len(all_articles)} статей для обработки")

    # Создаем или получаем целевую коллекцию
    target_collection = create_processed_collection(client)

    # Статистика
    stats = {
        "total_articles": len(all_articles),
        "processed_articles": 0,
        "skipped_articles": 0,
        "total_persons": 0,
        "total_organisations": 0,
        "total_locations": 0,
        "processing_time": None
    }

    # Создаем DataFrame для удобства
    df = pd.DataFrame(all_articles)
    start_time = datetime.now()
    processed_articles = []

    for i, (_, row) in enumerate(df.iterrows(), 1):
        try:
            print(f"\n[{i}/{len(all_articles)}] Обработка статьи {i}...")

            # Проверяем, есть ли текст
            if "text" not in row or not row["text"]:
                print(f"  Статья {i} пропущена: нет текста")
                stats["skipped_articles"] += 1
                continue

            # Предобработка текста
            text = ExtractedEntities.preprocess_text(row["text"])

            if not text or len(text.strip()) < 50:  # Минимальная длина текста
                print(f"  Статья {i} пропущена: текст слишком короткий")
                stats["skipped_articles"] += 1
                continue

            # Извлечение сущностей
            extracted_entities = extractor.extract_from_text(text)

            # Создаем документ для новой коллекции
            processed_article = {
                "_id": row.get("_id"),  # Сохраняем оригинальный ID
                "source": row.get("source"),
                "url": row.get("url"),
                "title": row.get("title", ""),
                "author":row.get("author", ""),
                "parse_date": row.get("parse_date"),
                "article_date": row.get("article_date"),
                "text": text,
                "section": row.get("section"),
                "tags": row.get("tags"),
                "extracted_entities": extracted_entities,
            }

            processed_articles.append(processed_article)

            # Обновляем статистику
            stats["processed_articles"] += 1
            stats["total_persons"] += len(extracted_entities["persons"])
            stats["total_organisations"] += len(extracted_entities["organisations"])
            stats["total_locations"] += len(extracted_entities["locations"])

            print(f"  Извлеченные сущности:")
            print(f"    • Персоны: {len(extracted_entities['persons'])}")
            print(f"    • Организации: {len(extracted_entities['organisations'])}")
            print(f"    • Локации: {len(extracted_entities['locations'])}")

            # Сохраняем батчами для эффективности
            if len(processed_articles) >= BATCH_SIZE:
                save_batch_to_mongodb(target_collection, processed_articles)
                processed_articles = []

        except Exception as e:
            print(f"  Ошибка при обработке статьи {i}: {str(e)}")
            stats["skipped_articles"] += 1
            continue

    # Сохраняем оставшиеся статьи
    if processed_articles:
        save_batch_to_mongodb(target_collection, processed_articles)

    # Рассчитываем время обработки
    end_time = datetime.now()
    stats["processing_time"] = str(end_time - start_time)

    return stats

def main():

    # Инициализация экстрактора сущностей
    extractor = ExtractedEntities(
        ner_model_path=NER_PATH,
        labels_path=LABELS_PATH,
        types_loc_path=TYPES_LOC_PATH,
        org_types_path=ORGS_TYPES_PATH
    )

    # Подключение к MongoDB
    print("Подключение к MongoDB...")
    client = pymongo.MongoClient(MONGO_URI)

    try:
        # Проверяем соединение
        client.admin.command('ping')
        print("Успешное подключение к MongoDB")
    except Exception as e:
        print(f"Ошибка подключения к MongoDB: {e}")
        return

    # Обрабатываем статьи и сохраняем в новую коллекцию
    print("\nНачало обработки статей...")
    stats = process_and_save_articles(client, extractor, limit=None)  # limit=None для всех статей

    # Выводим статистику
    print("\n" + "=" * 50)
    print("СТАТИСТИКА ОБРАБОТКИ")
    print("=" * 50)
    print(f"Всего статей: {stats['total_articles']}")
    print(f"Обработано: {stats['processed_articles']}")
    print(f"Пропущено: {stats['skipped_articles']}")
    print(f"Всего персон: {stats['total_persons']}")
    print(f"Всего организаций: {stats['total_organisations']}")
    print(f"Всего локаций: {stats['total_locations']}")
    print(f"Время обработки: {stats['processing_time']}")

    # Закрываем соединение
    client.close()
    print("\nОбработка завершена!")


if __name__ == "__main__":
    main()