from services.pre_processing.preprocess_script import preprocess
import ir_datasets


def apply_preprocess_on_all_docs(dataset, limitation=1000):
    dataset = ir_datasets.load(dataset)
    clean_dataset = {}
    inverted_index = {}
    if limitation > 0:
        for doc in dataset.docs_iter()[:1000]:
            clean_dataset[doc[0]] = preprocess(doc[1])
            inverted_index[doc[0]] = (preprocess(doc[1]), doc[1])
    else:
        for doc in dataset.docs_iter():
            clean_dataset[doc[0]] = preprocess(doc[1])
            inverted_index[doc[0]] = (preprocess(doc[1]), doc[1])

    return clean_dataset, inverted_index
