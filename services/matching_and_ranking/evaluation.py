def evaluation(cosine_similarity, qerls):

    tp = sum(cosine_similarity)
    fp = sum(cosine_similarity) - tp
    fn = sum(qerls) - tp
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * (precision * recall) / (precision + recall)

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score
    }
