import operator

import ir_datasets


def calculate_precision_at_k(actual, predicted, k):
    predicted_docs = {k: predicted[k] for k in list(predicted)[:k]}
    # predicted_docs = predicted # Access the retrieved documents for the specific query ID
    relevant = set(actual)
    retrieved = set(predicted_docs)
    intersection = relevant.intersection(retrieved)
    precision = len(intersection) / k
    return precision


def calculate_recall(actual, predicted):
    relevant = set(actual)
    retrieved = set(predicted)
    intersection = relevant.intersection(retrieved)
    recall = len(intersection) / len(relevant)
    return recall


def calculate_average_precision(actual, predicted):
    relevant = set(actual)
    precision_sum = 0.0
    num_relevant = 0
    for i, doc in enumerate(predicted):
        if doc in relevant:
            num_relevant += 1
            precision = num_relevant * actual.get(doc) / (i + 1)
            precision_sum += precision
    average_precision = precision_sum / len(relevant)
    return average_precision


def calculate_mrr(actual, predicted):
    relevant = set(actual)
    precision_sum = 0.0
    num_relevant = 0
    maxx = max(actual.items(), key=operator.itemgetter(1))[0]
    for i, doc in enumerate(predicted):
        if doc in relevant:
            num_relevant += 1
            precision = actual.get(maxx) + 1 - actual.get(doc) / num_relevant
            precision_sum += precision
    mrr = precision_sum / len(relevant)
    return mrr


# Load the relevance judgments (qrels) from the dataset
def evaluation(queries):
    dataset = ir_datasets.load("antique/test")
    qrels = dataset.qrels_dict()

    # Retrieve the ranking results from your retrieval system
    predicted = queries
    total_average_precision = 0.0
    num_queries = 0
    # Evaluate the retrieval system using the qrels
    for query_id, predicted_docs in predicted.items():
        actual_docs = qrels.get(query_id, {})  # Get the relevant documents for the query from qrels
        precision_at_3 = calculate_precision_at_k(actual_docs, predicted_docs, k=10)
        recall = calculate_recall(actual_docs, predicted_docs)
        average_precision = calculate_average_precision(actual_docs, predicted_docs)
        mrr = calculate_mrr(actual_docs, predicted_docs)

        print(f"Query: {query_id}")
        print("Precision@10:", precision_at_3)
        print("Recall:", recall)
        print("Average Precision:", average_precision)
        print("MRR:", mrr)
        print()
        total_average_precision += average_precision
        num_queries += 1

    map_score = total_average_precision / num_queries
    print("Mean Average Precision (MAP):", map_score)
