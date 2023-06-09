from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import ir_datasets
from pre_processing.apply_preprocess_on_all_docs import apply_preprocess_on_all_docs
from pre_processing.preprocess_script import preprocess

vectorizer = TfidfVectorizer()


def clustering_dataset():
    print("start clean data set")
    dataset = ir_datasets.load('antique/train')
    clean_dataset, inverted_index = apply_preprocess_on_all_docs(
        'antique/train')

    vectorizer = TfidfVectorizer()
    documents = list(clean_dataset.values())
    print("vectorize")
    tfidf = vectorizer.fit_transform(documents)

    # Perform K-means clustering
    k = 10  # Specify the number of clusters
    print("clust")
    kmeans = KMeans(n_clusters=k, n_init=10)
    kmeans.fit(tfidf)

    # Get the cluster labels for each document
    cluster_labels = kmeans.labels_

    # Store the cluster labels alongside the documents

    # Process a query
    query = "war in iraq"

    # Preprocess the query
    preprocessed_query = preprocess(query)

    # Vectorize the query
    query_vector = vectorizer.transform([preprocessed_query])

    # Compute similarity between the query and documents in each cluster
    cosine_similarities = cosine_similarity(query_vector, tfidf).flatten()

    most_similar_indices = cosine_similarities.argsort()[::-1]

    # Determine the cluster with the highest similarity score
    cluster_index = most_similar_indices.argmax()

    print(f"The query '{query}' belongs to cluster {cluster_index}")


clustering_dataset()