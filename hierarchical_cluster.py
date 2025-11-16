import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import os
import json
import pandas as pd
import pickle

from proto_classification import prepare_encode_func

ZH_EMBEDDING_MODEL_PATH = "<your-Chinese-sentence-embedding-model-path>"
# r"local_models/hfl_chinese-roberta-wwm-ext"
EN_EMBEDDING_MODEL_PATH = "<your-English-sentence-embedding-model-path>"
# r"local_models/sentence-transformers_all-mpnet-base-v2"

ZH_DATA_PATH = "Anliji_data.json"
EN_DATA_PATH = "AIID_data.json"
EN_DATA_EMBEDDING_SAVE_PATH = r"AIID_text_encodings_all-mpnet-base-v2.npy"
ZH_DATA_EMBEDDING_SAVE_PATH = r"Anliji_text_encodings_hfl_chinese-roberta-wwm-ext.npy"


def UAMP_dr(input_data, rdims=8, rs=43, n_neighbors=25, min_dist=0.01):
    import umap
    ######## !pip install umap-learn
    # dmeasure: distance metric | dmeasure="euclidean"
    # r-dims: Reduced dimensionality
    # rs: Random seed
    print(f"UMAP dimensionality reduction to {rdims} dimensions.")
    
    # Create and apply a UMAP 'reducer'
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=rdims,
        random_state=rs)
    reduced_en_data_encodings = reducer.fit_transform(input_data)

    # Create a dictionary that is easily converted into a pandas df
    embedded_dict = {}
    for i in range(0, reduced_en_data_encodings.shape[1]):
        embedded_dict[f"Dim {i+1}"] = reduced_en_data_encodings[:,i] # D{dimension_num} (Dim 1...Dim n)
    DFE = pd.DataFrame(embedded_dict, )#index=df.index)
    del(embedded_dict)
    return DFE


def clustering(DFE, save=False, num_clusters=5):
    from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, centroid

    Z = linkage(
        DFE[[x for x in DFE.columns if x.startswith('Dim ')]], 
        method='ward', metric='euclidean')
    # Save the output if you want to explore further later
    if save:
        pickle.dump(Z, open(os.path.join('data','Z.pickle'), 'wb'))

    # Extract clustering based on Z object
    clusterings  = fcluster(
        Z, 
        num_clusters, 
        criterion='maxclust'
    )
    print(clusterings)
    print(clusterings.shape)

    # 统计每一类别的文档数量
    from collections import Counter
    print(dict(Counter(clusterings.tolist())))
    return clusterings


if __name__ == "__main__":
    #### Step 2-1: Document Representation
    encode_sentence_english = prepare_encode_func(EN_EMBEDDING_MODEL_PATH)
    encode_sentence_chinese = prepare_encode_func(ZH_EMBEDDING_MODEL_PATH)

    with open(EN_DATA_PATH, "r", encoding="utf-8") as f:
        AIID_dataset = json.load(f)
    with open(ZH_DATA_PATH, "r", encoding="utf-8") as f:
        Anliji_dataset = json.load(f)
        
    if os.path.exists(EN_DATA_EMBEDDING_SAVE_PATH):
        en_data_encodings = np.load(EN_DATA_EMBEDDING_SAVE_PATH)
    else:
        en_data_encodings = encode_sentence_english([datapoint["title"]+ "\n" + datapoint["content"] for datapoint in AIID_dataset])
        np.save(EN_DATA_EMBEDDING_SAVE_PATH, en_data_encodings)
    print(f"en_data_encodings.shape: {en_data_encodings.shape}")

    if os.path.exists(ZH_DATA_EMBEDDING_SAVE_PATH):
        zh_data_encodings = np.load(ZH_DATA_EMBEDDING_SAVE_PATH)
    else:
        zh_data_encodings = encode_sentence_chinese([datapoint["title"]+ "\n" + datapoint["content"] for datapoint in Anliji_dataset])
        np.save(ZH_DATA_EMBEDDING_SAVE_PATH, zh_data_encodings)
    print(f"zh_data_encodings.shape: {zh_data_encodings.shape}")
    
    
    #### Step 2-2: Dimensionality Reduction using Uniform Manifold Approximation and Projection (UMAP)
    DFE_en = UAMP_dr(input_data=en_data_encodings, rdims=8)
    DFE_zh = UAMP_dr(input_data=zh_data_encodings, rdims=8)

    #### Step 2-3: Hierarchical clustering
    en_clusters = clustering(DFE_en)
    zh_clusters = clustering(DFE_zh)


