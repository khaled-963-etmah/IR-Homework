def vectorize_data(data, vectorized, is_query):

    if not is_query:
        documents = list(data.values())
        return vectorized.fit_transform(documents)
    else:
        return vectorized.transform([data])
