from collections import defaultdict
import math
import nltk


def create_inverted_index(data):
    inverted_index = defaultdict(list)
    for doc_id, doc_content in data.docs_iter()[0:10]:
        terms = nltk.word_tokenize(doc_content)
        for term in terms:
            inverted_index[term].append(doc_id)
    return dict(inverted_index)
