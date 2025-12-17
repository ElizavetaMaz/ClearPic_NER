import re

class AzerbaijaniLemmatizer:
    """Лемматизатор для азербайджанского на основе правил"""

    def __init__(self):
        # Правила удаления суффиксов (от самых длинных к коротким)
        self.suffix_rules = [
            # Формы с mə (инфинитив/деепричастие)
            ('məsində', 8), ('məsinə', 7), ('məsi', 6), ('mə', 3),

            # Падежные окончания
            ('sində', 6), ('sindən', 7), ('sından', 7), ('sündən', 7),
            ('dən', 3), ('dan', 3), ('də', 2), ('da', 2),
            ('nə', 3), ('na', 3), ('yə', 3), ('ya', 3),
            ('ni', 3), ('nı', 3), ('nü', 3), ('nu', 3),
            ('nin', 4), ('nın', 4), ('nün', 4), ('nun', 4),

            # Притяжательные + падеж
            ('ımda', 4), ('imdə', 4), ('umda', 4), ('ümdə', 4),
            ('ında', 4), ('ində', 4), ('unda', 4), ('ündə', 4),
            ('ımdan', 5), ('imdən', 5), ('umdan', 5), ('ümdən', 5),
            ('ından', 5), ('indən', 5), ('undan', 5), ('ündən', 5),

            # Множественное число + притяжательность
            ('ları', 4), ('ləri', 4), ('ların', 5), ('lərin', 5),
            ('larım', 5), ('lərim', 5), ('larımız', 7), ('lərimiz', 7),

            # Простые притяжательные
            ('ım', 2), ('im', 2), ('um', 2), ('üm', 2),
            ('ın', 2), ('in', 2), ('un', 2), ('ün', 2),

            # Простые окончания
            ('ı', 1), ('i', 1), ('u', 1), ('ü', 1),
            ('si', 2), ('sı', 2), ('su', 2), ('sü', 2),

            # Множественное число
            ('lar', 3), ('lər', 3),

            # Глагольные окончания
            ('ır', 2), ('ir', 2), ('ur', 2), ('ür', 2),
            ('ar', 2), ('ər', 2),
            ('mış', 3), ('miş', 3), ('muş', 3), ('müş', 3),
        ]

        # Словарь исключений
        self.exceptions = {
            'mənimsənilməsində': 'mənimsən',
            'mənimsənilməsi': 'mənimsən',
            'olunur': 'ol',
            'edir': 'et',
            'gedir': 'get',
            'görür': 'gör',
            'deyir': 'de',
            'alır': 'al',
            'verir': 'ver',
            'gəlir': 'gəl',
            'oxuyur': 'oxu',
            'yazır': 'yaz',
            'işləyir': 'işlə',
            'demək': 'de',
            'görmək': 'gör',
            'almaq': 'al',
        }

        # Слова, которые не нужно менять (неизменяемые)
        self.unchangeable = {
            'var', 'yox', 'çox', 'az', 'bəli', 'xeyr',
            'hə', 'yox', 'bəlkə', 'olası', 'mümkün'
        }

    def lemmatize(self, word, name=False):
        """Приведение слова к начальной форме"""
        if not word:
            return word

        if not name:
            word_lower = word.lower()
        else:
            word_lower = word

        # Проверяем исключения
        if word_lower in self.exceptions:
            return self.exceptions[word_lower]

        # Проверяем неизменяемые слова
        if word_lower in self.unchangeable:
            return word_lower

        # Пробуем удалять суффиксы по правилам
        for suffix, min_length in self.suffix_rules:
            if word_lower.endswith(suffix) and len(word_lower) > min_length:
                stem = word_lower[:-len(suffix)]

                # Проверяем минимальную длину основы
                if len(stem) >= 2:
                    # Проверяем гармонию гласных (упрощённо)
                    if self._check_harmony(stem, suffix):
                        return stem

        # Если ничего не подошло - возвращаем слово как есть
        return word_lower

    def lemmatize_text(self, text):
        """Лемматизация всего текста"""
        words = re.findall(r'\b[\wəğıöüşç]+\b', text, re.IGNORECASE)
        lemmas = [self.lemmatize(word) for word in words]
        return ' '.join(lemmas)

    def batch_lemmatize(self, words_list):
        """Пакетная лемматизация списка слов"""
        return [self.lemmatize(word) for word in words_list]

    def _check_harmony(self, stem, suffix):
        """Упрощённая проверка гармонии гласных"""
        back_vowels = {'a', 'ı', 'o', 'u'}
        front_vowels = {'ə', 'e', 'i', 'ö', 'ü'}

        # Находим последнюю гласную в основе
        last_vowel = None
        for char in reversed(stem):
            if char in back_vowels or char in front_vowels:
                last_vowel = char
                break

        if not last_vowel:
            return True

        # Определяем тип основы
        is_back = last_vowel in back_vowels

        # Проверяем гласные в суффиксе
        for char in suffix:
            if char in back_vowels:
                if not is_back:
                    return False
            elif char in front_vowels:
                if is_back:
                    return False

        return True