
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk


def generate_suggestions(query):
    # Preprocess the query
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    tokens = nltk.word_tokenize(query.lower())
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]

   
    
    # Get synonyms and antonyms for each word in the query
    suggestions = set()
    for token in tokens:
        for syn in wordnet.synsets(token):
            for lemma in syn.lemmas():
                suggestions.add(lemma.name())
                for antonym in lemma.antonyms():
                    suggestions.add(antonym.name())
    
    # Use TF-IDF to get the most relevant documents
    documents = [' '.join(tokens)]
    for suggestion in suggestions:
        documents.append(suggestion.replace('_', ' '))
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(documents)
    cosine_similarities = cosine_similarity(vectors[0], vectors[1:])
    most_similar_idx = cosine_similarities.argsort()[0][-10:]
    top_suggestions = [documents[i+1] for i in most_similar_idx]
    
    # Return the suggestions as a list
    return top_suggestions

print(generate_suggestions("sign"))


# nltk.download('punkt')


# dog cat programming happy sign