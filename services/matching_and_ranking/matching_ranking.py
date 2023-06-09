from sklearn.metrics.pairwise import cosine_similarity
# import sys


def matching(vectorized_query, vectorized_dataset, dataset):
    cosine_similarities = cosine_similarity(vectorized_query, vectorized_dataset).flatten()

    most_similar_indices = cosine_similarities.argsort()[::-1]

    num_similar_documents = 10
    most_similar_documents = [dataset[i] for i in most_similar_indices[:num_similar_documents]]

    dic = {}
    for index in range(0, len(most_similar_documents)):
        dic[most_similar_documents[index]] = cosine_similarities[index]
    return dic, cosine_similarities
