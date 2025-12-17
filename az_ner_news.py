import pandas as pd
import re
import json
from transformers import XLMRobertaTokenizerFast, XLMRobertaForTokenClassification, pipeline
from az_stemming.lemmatizator import AzerbaijaniLemmatizer


class ExtractedEntities:
    """Класс для извлечения сущностей из текста."""

    def __init__(self, ner_model_path, labels_path, types_loc_path, org_types_path):
        """
        Инициализация класса для извлечения сущностей.

        Args:
            ner_model_path: Путь к модели NER
            labels_path: Путь к файлу маппинга лейблов
            types_loc_path: Путь к файлу типов локаций
            org_types_path: Путь к файлу типов организаций
        """
        self.ner_path = ner_model_path
        self.labels_path = labels_path
        self.types_loc_path = types_loc_path
        self.org_types_path = org_types_path

        # Загрузка конфигурационных файлов
        with open(self.labels_path, 'r', encoding='utf-8') as f:
            self.label_mapping = json.load(f)

        with open(self.types_loc_path, 'r', encoding='utf-8') as f:
            self.types_city_country = json.load(f)

        with open(self.org_types_path, 'r', encoding='utf-8') as f:
            self.org_types = json.load(f)

        # Инициализация моделей
        self.nlp_local = None
        self.lemmatizer = None
        self._init_models()

        # Список стоп-слов для фильтрации
        self.common_words_az = {
            'mən', 'sən', 'o', 'biz', 'siz', 'onlar',
            'bu', 'həmin', 'belə', 'kim', 'nə', 'harada',
            'necə', 'niyə', 'nə üçün', 'azərbaycan',
            'respublika', 'dövlət', 'şəhər', 'rayon',
            'universitet', 'bank', 'şirkət', 'kompaniya',
            'ölkədə', 'ölkə', 'şəhərdə', 'rayonda'
        }

    def _init_models(self):
        """Инициализация моделей NER и лемматизатора."""
        if self.nlp_local is None:
            tokenizer = XLMRobertaTokenizerFast.from_pretrained(self.ner_path)
            model = XLMRobertaForTokenClassification.from_pretrained(self.ner_path)
            self.nlp_local = pipeline(
                "ner",
                model=model,
                tokenizer=tokenizer,
                aggregation_strategy="simple"
            )

        if self.lemmatizer is None:
            self.lemmatizer = AzerbaijaniLemmatizer()

    def _is_proper_name(self, text):
        """
        Проверяет, является ли текст именем собственным.

        Args:
            text: Текст для проверки

        Returns:
            bool: True если вероятно это имя собственное
        """
        if not text or len(text.strip()) < 2:
            return False

        text = text.strip()

        # 1. Должен начинаться с заглавной буквы
        if not text[0].isupper():
            return False

        # 2. Не должен быть в стоп-словах
        if text.lower() in self.common_words_az:
            return False

        # 3. Не должен содержать только цифры или специальные символы
        if re.search(r'^\d+$', text):
            return False

        # 4. Должен содержать хотя бы одну букву
        if not any(c.isalpha() for c in text):
            return False

        # # 5. Проверка для составных имен (через пробел или дефис)
        # if ' ' in text or '-' in text:
        #     parts = re.split(r'[\s-]+', text)
        #     # Каждая часть должна начинаться с заглавной буквы
        #     for part in parts:
        #         if part and not part[0].isupper():
        #             return False

        return True

    def _prepare_entities_df(self, entities):
        """
        Преобразует результаты NER в DataFrame с правильными лейблами.

        Args:
            entities: Список сущностей из модели NER

        Returns:
            pd.DataFrame: DataFrame с сущностями
        """
        for i in range(len(entities)):
            entity_group = entities[i]['entity_group']
            # Конвертируем числовой лейбл в текстовый
            entities[i]["entity_group"] = self.label_mapping.get(
                str(entity_group.split('_')[-1]),
                entity_group
            )

        return pd.DataFrame(entities)

    def _is_part_of_existing_name(self, new_name, existing_links):
        """
        Проверяет, является ли новое имя частью существующего.

        Args:
            new_name: Новое имя для проверки
            existing_links: Список существующих сущностей

        Returns:
            bool: True если имя уже содержится в существующих сущностях
        """
        for link in existing_links:
            if (new_name in link.get('name', '') or
                    link.get('name', '') in new_name):
                return True
        return False

    def _link_person_position(self, df_ner):
        """
        Связывает персону с должностью на основе порядка в тексте.

        Args:
            df_ner: DataFrame с распознанными сущностями

        Returns:
            list: Список словарей с найденными связями персон и должностей
        """
        links = []

        # Находим все персоны и должности
        persons = df_ner[df_ner["entity_group"] == "PERSON"]
        positions = df_ner[df_ner["entity_group"] == "POSITION"]

        for idx, person in persons.iterrows():
            person_text = person["word"].strip()

            # Проверяем, является ли это именем собственным
            if not self._is_proper_name(person_text):
                continue

            person_lemmatized = self.lemmatizer.lemmatize(person_text, name=True)

            # Проверяем, не является ли это частью уже найденного имени
            if self._is_part_of_existing_name(person_lemmatized, links):
                continue

            person_index = idx
            position = "unknown"
            start = person["start"]
            end = person["end"]

            # Ищем ближайшую должность ДО этой персоны
            candidate_positions = positions[positions.index < person_index]

            if not candidate_positions.empty:
                # Берем самую близкую должность
                closest_position = candidate_positions.iloc[-1]
                closest_position_index = candidate_positions.index[-1]

                # Проверяем расстояние между должностью и именем
                distance = person_index - closest_position_index

                if distance <= 2:  # Максимум 1 сущность между ними
                    position = self.lemmatizer.lemmatize(closest_position["word"])
                    start = closest_position["start"]
                    end = closest_position["end"]

            links.append({
                "name": person_lemmatized,
                "original_name": person_text,
                "position": position,
                "mentions": {
                    "start_char": str(start),
                    "end_char": str(end)
                }
            })

        return links

    def _prepare_locations(self, df_ner):
        """
        Обрабатывает локации с поиском типа локации.

        Args:
            df_ner: DataFrame с распознанными сущностями

        Returns:
            list: Список словарей с найденными локациями
        """
        entities_locations = []

        # Находим все локации и страны
        locations = df_ner[df_ner['entity_group'] == "LOCATION"]
        gpe = df_ner[df_ner['entity_group'] == 'GPE']

        for _, loc in pd.concat([locations, gpe], axis=0).iterrows():
            loc_text = loc["word"].replace('"', '').strip()

            # Проверка на имя собственное
            if not self._is_proper_name(loc_text):
                continue

            start = loc["start"]
            end = loc["end"]
            loc_type = "COUNTRY"

            # Ищем тип локации в словаре
            for loc_type_key, loc_list in self.types_city_country.items():
                if loc_text in loc_list:
                    loc_type = loc_type_key
                    break

            entities_locations.append({
                "name": loc_text,
                "type": loc_type,
                "mentions": {
                    "start_char": str(start),
                    "end_char": str(end)
                }
            })

        return entities_locations

    def _extract_clean_org_name(self, org_text):
        """
        Извлекает чистое название организации.

        Args:
            org_text: Текст с названием организации

        Returns:
            str: Очищенное название организации
        """
        # Убираем лишние пробелы
        org_text = re.sub(r'\s+', ' ', org_text).strip()

        # Паттерны для извлечения
        patterns = [
            r'"([^"]+)"\s*(?:\([^)]+\))?\s*(.*)',  # "Название" (расшифровка)
            r'([^(]+)\s*\(([^)]+)\)',  # Название (сокращение)
            r'(\b[A-Z]+\b[\+\-]?)\s*[-–]\s*(.+)',  # Сокращение - полное название
            r'"([^"]+)"'  # Просто в кавычках
        ]

        for pattern in patterns:
            match = re.match(pattern, org_text)
            if match:
                if match.lastindex >= 1:
                    clean = match.group(1).strip()
                    # Если есть расшифровка, можно её добавить
                    if match.lastindex >= 2 and match.group(2):
                        extension = match.group(2).strip()
                        if len(extension) > 3 and not extension.lower().startswith('bey'):
                            clean = f"{clean} ({extension})"
                    return clean


        # Если ничего не подошло, возвращаем как есть (но чистим)
        return org_text.replace('"', '').strip()


    def _prepare_organisations(self, df_ner):
        """
        Обрабатывает организации с поиском типа организации.

        Args:
            df_ner: DataFrame с распознанными сущностями

        Returns:
            list: Список словарей с найденными организациями
        """
        entities_organisations = []

        # Находим все организации
        organisations = df_ner[df_ner['entity_group'] == "ORGANISATION"]

        for _, org in organisations.iterrows():
            org_text = org["word"].strip()
            org_name = self._extract_clean_org_name(org_text)
            # Проверка на имя собственное
            if not self._is_proper_name(org_name):
                continue

            # Проверяем, не является ли это частью уже найденной организации
            if self._is_part_of_existing_name(org_name, entities_organisations):
                continue

            start = org["start"]
            end = org["end"]
            org_type = 'COMPANY'
            org_lower = org_text.lower()

            # Определяем тип организации
            for org_type_key, org_list in self.org_types.items():
                if org_lower in org_list:
                    org_type = org_type_key
                    break

            # Проверка суффиксов компаний
            if any(org_lower.endswith(suffix) for suffix in [' mmc', ' asc', ' mq', ' ik']):
                org_type = 'COMPANY'

            entities_organisations.append({
                "name": org_name,
                "type": org_type,
                "mentions": {
                    "start_char": str(start),
                    "end_char": str(end)
                }
            })

        return entities_organisations

    def extract_from_text(self, text):
        """
        Основной метод для извлечения сущностей из текста.

        Args:
            text: Текст для обработки

        Returns:
            dict: Словарь с извлеченными сущностями и DataFrame с оставшимся текстом
        """
        # Выполняем NER
        ner_results = self.nlp_local(text)

        # Преобразуем в DataFrame
        df_ner = self._prepare_entities_df(ner_results)

        # Извлекаем сущности
        extracted_entities = {
            "persons": self._link_person_position(df_ner),
            "organisations": self._prepare_organisations(df_ner),
            "locations": self._prepare_locations(df_ner)
        }

        # Получаем оставшийся текст (не сущности)
        remaining_text = df_ner[df_ner['entity_group'] == 'O']

        return extracted_entities, remaining_text

    @staticmethod
    def preprocess_text(text):
        """
        Предобработка текста перед извлечением сущностей.

        Args:
            text: Исходный текст

        Returns:
            str: Обработанный текст
        """
        if not text:
            return ""

        # Заменяем переносы строк и специальные символы
        text = text.replace("\n", " ").replace("«", '"').replace("»", '"')
        text = text.replace("“", '"').replace("”", '"').replace("•", "")

        # Заменяем сокращения
        text = text.replace("mln.", "milyon").replace("mlrd.", "milyard")

        # Убираем лишние пробелы
        text = re.sub(r"\s+", " ", text).strip()

        return text