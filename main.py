# import our services
from jedi import settings
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
from services.script import search_task
from services.evaluation.main import evaluation
from services.matching_and_ranking.matching_ranking import matching
from services.re_presentation.vectorize_data import vectorize_data
from services.pre_processing.preprocess_script import preprocess
from services.pre_processing.apply_preprocess_on_all_docs import apply_preprocess_on_all_docs
import json
# import external services

import pandas as pd
import ir_datasets
import httpx
import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


# initialize
app = FastAPI()
vectorize = TfidfVectorizer()
clean_dataset = {}
inverted_index = {}


# @app.get("/{name_dataset}")
# async def initialize_datset(name_dataset):
#     global clean_dataset, inverted_index
#     name_dataset = name_dataset.replace("-", "/")
#     clean_dataset, inverted_index = apply_preprocess_on_all_docs(name_dataset)

# get dataset name


@app.get("/datasets")
def dataset_set():
    data = {
        "data": [
            "antique",
            "antique-test",
            "antique-train",
            "aquaint",
        ], }
    return data


# get all dataset
@app.get("/get_all")
async def get_all_clean():
    return {"inverted_index": inverted_index, "clean_dataset": clean_dataset}


# pre process on dataset
@app.get("/pre_process_dataset/{name_dataset}")
async def pre_process_on_dataset_api(name_dataset: str):
    global clean_dataset, inverted_index
    name_dataset = name_dataset.replace("-", "/")
    clean_dataset, inverted_index = apply_preprocess_on_all_docs(name_dataset)
    return "processing done"


# pre process on query
@app.get("/pre_process_on_query/{query}")
async def pre_process_on_query_api(query: str):
    return preprocess(query)
#
#

# re presentation dataset after cleaning


@app.get("/re_presentation/dataset/")
async def re_presentation_data_api():
    tfidf_data = vectorize_data(clean_dataset, vectorize, False)
    csr_dict = {
        "__module__": "scipy.sparse._csr",
        "__doc__": tfidf_data.__doc__,
        "format": tfidf_data.format,
        "data": tfidf_data.data.tolist(),
        "indices": tfidf_data.indices.tolist(),
        "indptr": tfidf_data.indptr.tolist(),
        "shape": tfidf_data.shape,
        "dtype": str(tfidf_data.dtype),
    }
#
#     # Serialize the dictionary to JSON
#     # json_dataset_object = json.dumps(csr_dict)
    return csr_dict
#
#

# representation query after cleaning


@app.get("/re_presentation/query/{query}")
async def re_presentation_query_api(query: str):
    tfidf_query = vectorize_data(query, vectorize, True)
    csr_dict = {
        "__module__": "scipy.sparse._csr",
        "__doc__": tfidf_query.__doc__,
        "format": tfidf_query.format,
        "data": tfidf_query.data.tolist(),
        "indices": tfidf_query.indices.tolist(),
        "indptr": tfidf_query.indptr.tolist(),
        "shape": tfidf_query.shape,
        "dtype": str(tfidf_query.dtype),
    }
    return csr_dict
#
#


# matching
@app.get("/matching/{query}")
async def matching_api(query: str):
    json_object_dataset = await re_presentation_data_api()
    vectorize_dataset = json_to_csr_matrix(json_object_dataset)

    query_after_processing = await pre_process_on_query_api(query)
    json_query_object = await re_presentation_query_api(query_after_processing)
    vectorize_query = json_to_csr_matrix(json_query_object)

    documents = list(clean_dataset.keys())
    dic, cosine_similarity = matching(
        vectorize_query, vectorize_dataset, documents)

    return dic, cosine_similarity

# to make input evaluation function


@app.get("/pre_evaluate")
async def pre_evaluate_api():
    dataset = ir_datasets.load("antique/train")
    await pre_process_on_dataset_api("antique/train")
    json_object_dataset = await re_presentation_data_api()
    vectorize_dataset = json_to_csr_matrix(json_object_dataset)
    query_vector = []
    query1 = ''
    queries = {}
    documents = list(clean_dataset.keys())
    for query in dataset.queries_iter():
        query1 = query
        tokenize_query = preprocess(query[1])
        query_vector = vectorize_data(tokenize_query, vectorize, True)
        matching_data, cosine = matching(
            vectorize_dataset, query_vector, documents)
        queries[query1[0]] = matching_data

    return queries

# evaluate function


@app.get("/evaluation_searcg_engine")
async def evaluate_api():
    queries = pre_evaluate_api()
    evaluation(queries)


# main task
@app.get("/search_task/{query}")
async def search(query: str):
    x = search_task(query, vectorize, clean_dataset, inverted_index)
    data = {"results": x}
    return data


# Utils function because tfidf is object

def json_to_csr_matrix(json_object):
    data = json_object["data"]
    indices = json_object["indices"]
    indptr = json_object["indptr"]
    shape = tuple(json_object["shape"])
    # dtype = eval(json_object["dtype"])
    return sparse.csr_matrix((data, indices, indptr), shape=shape)
