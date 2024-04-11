import re
import nltk
import pymorphy2
from nltk.corpus import stopwords

# Загрузка русских стоп-слов
nltk.download('stopwords')
russian_stopwords = stopwords.words('russian')

# Инициализация анализатора pymorphy2
morph = pymorphy2.MorphAnalyzer()

def preprocess_text(text):
    # Удаление лишних символов и нормализация
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    
    # Токенизация
    words = nltk.word_tokenize(text, language="russian")
    
    # Удаление стоп-слов и лемматизация
    words_lemmatized = [morph.parse(word)[0].normal_form for word in words if word not in russian_stopwords]
    
    return ' '.join(words_lemmatized)