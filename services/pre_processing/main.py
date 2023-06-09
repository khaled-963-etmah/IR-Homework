import re
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.stem import PorterStemmer, WordNetLemmatizer, ISRIStemmer


def remove_special_chars(sentence):
    new_str = re.sub(r'[.|?|@|$|%|&]', ' ', sentence)
    return new_str.split()


stop_words = set(stopwords.words('english'))


def remove_stop_words(sentence):
    return [word.lower() for word in sentence if word.lower() not in stop_words]


stemmer = PorterStemmer()


def stemming(sentence):
    return [stemmer.stem(word) for word in sentence]


verbs_tags = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']


def stemming_tags(sentence):
    return [stemmer.stem(word) if pos_tag(word.split(' '))[0][1] in verbs_tags else word for word in sentence]


lemmatizer = WordNetLemmatizer()


def lemmatization(sentence):
    return [lemmatizer.lemmatize(word) for word in sentence]


def preprocess(sentence, rm_stop_words=True, rm_special_chars=True, stemmer=True, lemmatizer=True):
    sentence = str(sentence).split()
    if rm_special_chars:
        sentence = remove_special_chars(" ".join(sentence))

    if rm_stop_words:
        sentence = remove_stop_words(sentence)

    if lemmatizer:
        sentence = lemmatization(sentence)

    if stemmer:
        sentence = stemming(sentence)

    return " ".join(sentence)
