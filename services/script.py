import ir_datasets
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from services.matching_and_ranking.matching_ranking import matching
from services.pre_processing.preprocess_script import preprocess
from services.pre_processing.apply_preprocess_on_all_docs import apply_preprocess_on_all_docs
from services.re_presentation.vectorize_data import vectorize_data


def search_task(query, vectorize, clean_dataset, inverted_index):
    documents = list(clean_dataset.keys())
    tfidf_matrix_dataset = vectorize_data(clean_dataset, vectorize, False)
    query_vector = vectorize_data(preprocess(query), vectorize, True)
    matching_data, cosine = matching(tfidf_matrix_dataset, query_vector, documents)

    return [inverted_index[value][1] for value in matching_data]



# dataset = ir_datasets.load("car/v1.5/train/fold0")
# clean_dataset ,inverted_index = []
# vectorize = TfidfVectorizer()
# clean_dataset ,inverted_index = apply_preprocess_on_all_docs("car/v1.5/train/fold0")
# documents = list(clean_dataset.keys())
# tfidf_matrix_dataset = vectorize_data(clean_dataset, vectorize, False)
# query_vector = vectorize_data(preprocess("car is good and speed"), vectorize, True)
# matching_data, cosine = matching(tfidf_matrix_dataset, query_vector, documents)
# result = [inverted_index[value][1] for value in matching_data]