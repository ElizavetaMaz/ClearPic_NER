import pandas as pd
import pymongo
from dotenv import load_dotenv
import os
import json
from az_ner_news import ExtractedEntities
from tqdm import tqdm

load_dotenv()

# Загрузка переменных окружения
MONGO_URI = os.getenv("MONGO_URI")
NER_PATH = os.getenv("NER_PATH", "./ner_azerbaijan_local")
TYPES_LOC_PATH = os.getenv("TYPES_LOC_PATH", "config/types_city_country.json")
LABELS_PATH = os.getenv("LABELS_PATH", "config/label_mapping.json")
ORGS_TYPES_PATH = os.getenv("ORGS_TYPES_PATH", "config/types_org.json")
OUTPUT_PATH = os.getenv("OUTPUT_PATH", "output/output_all.json")

# Настройки
SOURCE_COLLECTION = "articles"  # Исходная коллекция
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

    db = client["az_articles"]
    all_articles = list(db[SOURCE_COLLECTION].find())

    df = pd.DataFrame(all_articles)

    # Предобработка текстов
    df["text"] = df["text"].apply(ExtractedEntities.preprocess_text)

    # Извлечение сущностей для первых 10 статей
    results = []
    for i in tqdm(range(len(df)), total=len(df), desc="Обработка статей"):

        text = df["text"][i]
        extracted_entities, remaining_text = extractor.extract_from_text(text)

        # Обновляем статью с извлеченными сущностями
        article = all_articles[i]
        article["extracted_entities"] = extracted_entities
        article['_id'] = str(article['_id'])

        results.append(article)

    os.makedirs("output", exist_ok=True)
    # Сохранение результатов
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"Результаты сохранены в {OUTPUT_PATH}")

    # Закрываем соединение
    client.close()
    print("\nОбработка завершена!")


if __name__ == "__main__":
    main()
