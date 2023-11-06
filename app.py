import time
import json
import numpy as np

import streamlit as st
from pathlib import Path
from collections import defaultdict

import sys
path_root = Path("./")
sys.path.append(str(path_root))


st.set_page_config(page_title="PSC Runtime",
                   page_icon='🌸', layout="centered")



name = st.selectbox(
    "Choose a dataset",
    ["dl19", "dl20"],
    index=None,
    placeholder="Choose a dataset..."
)

model_name = st.selectbox(
    "Choose a model",
    ["gpt-3.5", "gpt-4"],
    index=None,
    placeholder="Choose a model..."
)


if name and model_name:
    import torch
    # fn = f"dl19-gpt-3.5.pt"
    fn = f"{name}-{model_name}.pt"
    object = torch.load(fn)

    outputs = object[2]
    query2outputs = {}
    for output in outputs:
        all_queries = {x['query'] for x in output}
        assert len(all_queries) == 1
        query = list(all_queries)[0]
        query2outputs[query] = [x['hits'] for x in output]

    search_query = st.selectbox(
        "Choose a query from the list",
        sorted(query2outputs),
        # index=None,
        # placeholder="Choose a query from the list..."
    )
    
    def preferences_from_hits(list_of_hits):
        docid2id = {}
        id2doc = {}
        preferences = []

        for result in list_of_hits:
            for doc in result:
                if doc["docid"] not in docid2id:
                    id = len(docid2id)
                    docid2id[doc["docid"]] = id
                    id2doc[id] = doc
            print([doc["docid"] for doc in result])
            print([docid2id[doc["docid"]] for doc in result])
            preferences.append([docid2id[doc["docid"]] for doc in result])
    
        #  = {v: k for k, v in docid2id.items()}
        return np.array(preferences), id2doc


    def load_qrels(name):
        import ir_datasets
        if name == "dl19":
            ds_name = "msmarco-passage/trec-dl-2019/judged"
        elif name == "dl20":
            ds_name = "msmarco-passage/trec-dl-2020/judged"
        else:
            raise ValueError(name)
    
        dataset = ir_datasets.load(ds_name)
        qrels = defaultdict(dict)
        for qrel in dataset.qrels_iter():
            qrels[qrel.query_id][qrel.doc_id] = qrel.relevance
        return qrels


    def aggregate(list_of_hits):
        import numpy as np
        from permsc import KemenyOptimalAggregator, sum_kendall_tau, ranks_from_preferences
        from permsc import BordaRankAggregator
    
        preferences, id2doc = preferences_from_hits(list_of_hits)
        y_optimal = KemenyOptimalAggregator().aggregate(preferences)
        # y_optimal = BordaRankAggregator().aggregate(preferences)
    
        return [id2doc[id] for id in y_optimal]


    def write_ranking(search_results, text):
        st.write(f'<p align=\"right\" style=\"color:grey;\"> {text} ms</p>', unsafe_allow_html=True)

        qid = {result["qid"] for result in search_results}
        assert len(qid) == 1
        qid = list(qid)[0]
    
        for i, result in enumerate(search_results):
            result_id = result["docid"]
            contents = result["content"]

            label = qrels[str(qid)].get(str(result_id), -1)
            label_text = "Unlabeled"
            if label == 3:
                style = "style=\"color:rgb(237, 125, 12);\""
                label_text = "High"
            elif label == 2:
                style = "style=\"color:rgb(244, 185, 66);\""
                label_text = "Medium"
            elif label == 1:
                style = "style=\"color:rgb(241, 177, 118);\""
                label_text = "Low"
            elif label == 0:
                style = "style=\"color:black;\""
                label_text = "Not Relevance"
            else:
                style = "style=\"color:grey;\""

            print(qid, result_id, label, style)
            # output = f'<div class="row"> <b>Rank</b>: {i+1} | <b>Document ID</b>: {result_id} | <b>Score</b>:{result_score:.2f}</div>'
            output_1 = f'<div class="row" {style}> <b>Rank</b>: {i+1} | <b>Document ID</b>: {result_id}</div>'
            output_2 = f'<div class="row" {style}> <b>True Relevance</b>: {label_text}</div>'
    
            try:
                st.write(output_1, unsafe_allow_html=True)
                st.write(output_2, unsafe_allow_html=True)
                st.write(
                    f'<div class="row" {style}>{contents}</div>', unsafe_allow_html=True)
    
            except:
                pass
            st.write('---')
    
    
    aggregated_ranking = aggregate(query2outputs[search_query])
    qrels = load_qrels(name)
    col1, col2 = st.columns([5, 5])
    
    if search_query:
        with col1:
            if search_query or button_clicked:
                write_ranking(search_results=query2outputs[search_query][0], text="w/o PSC")
        
        with col2:
            if search_query or button_clicked:
                write_ranking(search_results=aggregated_ranking, text="w/ PSC")
