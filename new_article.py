import pymongo
from dotenv import load_dotenv
import os
from az_ner_news import ExtractedEntities
from datetime import datetime
import logging
from bson import ObjectId

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
NER_PATH = os.getenv("NER_PATH", "./ner_azerbaijan_local")
TYPES_LOC_PATH = os.getenv("TYPES_LOC_PATH", "config/types_city_country.json")
LABELS_PATH = os.getenv("LABELS_PATH", "config/label_mapping.json")
ORGS_TYPES_PATH = os.getenv("ORGS_TYPES_PATH", "config/types_org.json")

# Настройки
SOURCE_COLLECTION = "articles"  # Исходная коллекция
PREPARED_COLLECTION = "articles_with_ner"  # Новая коллекция для обработанных статей

def add_new_article_in_db(url, title, text, source=None, author=None, article_date=None, section=None, tags=None):
    """
    Загрузка новой статьи в базу данных.

    Args:
        url: Ссылка на статью (обязательное)
        title: Заголовок статьи (обязательное)
        text: Текст статьи (обязательное)
        source: Источник статьи (необязательное)
        author: Автор статьи (необязательное)
        article_date: Дата публикации статьи (необязательное)
        section: Раздел (необязательное)
        tags: Тэги (необязательное)
    """

    # Проверка, что поля все есть и не пустые
    if not url or not title or not text:
        logging.error("Не заполнены обязательные поля")
        return

    # Подключение к MongoDB
    print("Подключение к MongoDB...")
    client = pymongo.MongoClient(MONGO_URI)
    db = client["az_articles"]

    # Генерируем единый ObjectId
    article_id = ObjectId()

    article = {
        "_id": article_id,
        "source": source,
        "url": url,
        "title": title,
        "author": author,
        "parse_date": datetime.now().isoformat(),
        "article_date": article_date,
        "text": text,
        "section": section,
        "tags": tags,
    }

    # Вставляем в исходную коллекцию
    try:
        db[SOURCE_COLLECTION].insert_one(article)
        print(f"Статья сохранена в коллекцию {SOURCE_COLLECTION} с ID: {article_id}")
    except Exception as e:
        logging.error(f"Ошибка при сохранении в {SOURCE_COLLECTION}: {str(e)}")
        client.close()
        return None

    # Инициализация экстрактора сущностей
    extractor = ExtractedEntities(
        ner_model_path=NER_PATH,
        labels_path=LABELS_PATH,
        types_loc_path=TYPES_LOC_PATH,
        org_types_path=ORGS_TYPES_PATH
    )

    print("\nОбработка статьи...")
    # Предобработка текста
    text = ExtractedEntities.preprocess_text(text)

    start_time = datetime.now()

    # Извлечение сущностей
    extracted_entities = extractor.extract_from_text(text)

    # Собираем обработанный текст
    processed_article = article.copy()
    processed_article["text"] = text
    processed_article["extracted_entities"] = extracted_entities

    # Вставляем в коллекцию c NER
    try:
        db[PREPARED_COLLECTION].insert_one(article)
        print(f"Обработанная статья сохранена в коллекцию {PREPARED_COLLECTION} с ID: {article_id}")
    except Exception as e:
        logging.error(f"Ошибка при сохранении в {PREPARED_COLLECTION}: {str(e)}")
        client.close()
        return None

    # Закрываем соединение
    client.close()

    # Рассчитываем время обработки
    end_time = datetime.now()
    print(f"\nОбработка завершена! Обработка длилась {str(end_time - start_time)}")




add_new_article_in_db(
    url="https://apa.az/senaye-ve-energetika/nyu-york-birjasinda-tebii-qaz-bahalasir-926630",
    title='',
    text='Цена фьючерсных котировок на природный газ на товарной бирже NYMEX в Нью-Йорке выросла. По данным Economists, цена фьючерсов на природный газ составляет 1 миллион долларов, с учётом поставки в декабре. Британская тепловая единица (BTU) выросла на 0,18% до $4,35.'
)
