import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import os
import json
import pandas as pd


ZH_EMBEDDING_MODEL_PATH = "<your-Chinese-sentence-embedding-model-path>"
# r"local_models/hfl_chinese-roberta-wwm-ext"
EN_EMBEDDING_MODEL_PATH = "<your-English-sentence-embedding-model-path>"
# r"local_models/sentence-transformers_all-mpnet-base-v2"

ZH_DATA_PATH = "<your-AI-risk-data-path-in-json-format>"
EN_DATA_PATH = "<your-AI-risk-data-path-in-json-format>"

PROTOTYPES_FILEPATH = r"prototypes.json"
EN_PROTOTYPES_EMBEDDING_SAVE_PATH = r"en_prototype_encodings_all-mpnet-base-v2.npy"
ZH_PROTOTYPES_EMBEDDING_SAVE_PATH = r"zh_prototype_encodings_hfl_chinese-roberta-wwm-ext.npy"
EN_DATA_EMBEDDING_SAVE_PATH = r"AIID_text_encodings_all-mpnet-base-v2.npy"
ZH_DATA_EMBEDDING_SAVE_PATH = r"Anliji_text_encodings_hfl_chinese-roberta-wwm-ext.npy"



# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def prepare_encode_func(embed_model_path):
    # Load model from HuggingFace Hub or from local
    tokenizer = AutoTokenizer.from_pretrained(embed_model_path)
    model = AutoModel.from_pretrained(embed_model_path)

    def encode(sentences):
        # Tokenize sentences
        encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

        # Compute token embeddings
        with torch.no_grad():
            model_output = model(**encoded_input)

        # Perform pooling
        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

        # Normalize embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

        print("Sentence embeddings:")
        # print(sentence_embeddings)
        # print(sentence_embeddings.shape)

        # print(sentence_embeddings.numpy())
        # print(sentence_embeddings.numpy().shape)
        return sentence_embeddings.numpy()
    return encode


def compute_distances(A, B):
    # 计算 A 和 B 每个向量之间的欧氏距离
    # 计算 (A - B) 的差
    diff = A[:, np.newaxis, :] - B[np.newaxis, :, :]
    # 计算差的平方
    squared_diff = np.square(diff)
    # 按照最后一维（向量的维度）求和
    summed_squared_diff = np.sum(squared_diff, axis=-1)
    # 开方得到欧氏距离
    distances = np.sqrt(summed_squared_diff)
    return distances


def nearse_neighbour(text_encodings, prototype_encodings):
    from collections import Counter
    
    distances = compute_distances(text_encodings, prototype_encodings)
    # print(distances.shape)
    min_indexes = np.argmin(distances, axis=-1)
    # print(min_indexes)
    # print(min_indexes.shape)
    min_indexes = min_indexes.tolist()
    CLASS_INDEX = ["数据采集", "数据存储", "数据计算", "数据管理", "数据应用"]
    classification_output = []
    for c in min_indexes:
        classification_output.append(CLASS_INDEX[c//5])
    print(f"分类结果统计：{Counter(classification_output)}")
    return classification_output


if __name__ == "__main__":
    ######## 测试 ########
    # 示例：测试句子表示向量计算
    demo_sentences = ['This is an example sentence', 'Each sentence is converted']
    encode_sentence_english = prepare_encode_func(EN_EMBEDDING_MODEL_PATH)
    encode_sentence_chinese = prepare_encode_func(ZH_EMBEDDING_MODEL_PATH)
    sentence_embeddings = encode_sentence_english(demo_sentences)
    print(f"句子嵌入得到向量的维度为：{sentence_embeddings.shape()}")
    
    
    # 示例：矩阵 A 和 B 之中每对向量两两之间的欧式距离计算
    A = np.array([[1, 2], [3, 4], [5, 6]])  # 3个向量，每个向量维度为2
    B = np.array([[7, 8], [9, 10]])  # 2个向量，每个向量维度为2
    # 计算 A 中每个向量和 B 中每个向量的欧氏距离
    distances = compute_distances(A, B)
    print(f"欧氏距离矩阵：{distances}")
    
    ######## 计算 ########
    #### 步骤 1-1：获取原型
    #### Step 1-1: Prototype Generation
    # Using generative LLM to generate prototypes under a prompt and checking manually
    # Already done and saved to `prototypes.json`, you can also generate your version

    #### 步骤 1-2：计算文本表征
    #### Step 1-2: Text Representation
    # calculating prototype embeddings
    with open(PROTOTYPES_FILEPATH, "r", encoding="utf-8") as f:
        prototype_list = json.load(f)
    
    if os.path.exists(EN_PROTOTYPES_EMBEDDING_SAVE_PATH):
        en_prototype_encodings = np.load(EN_PROTOTYPES_EMBEDDING_SAVE_PATH)
    else:
        en_prototype_encodings = encode_sentence_english([prototype["text-en"] for prototype in prototype_list])
        np.save(EN_PROTOTYPES_EMBEDDING_SAVE_PATH, en_prototype_encodings)
    print(f"en_prototype_encodings.shape: {en_prototype_encodings.shape}")
    
    if os.path.exists(ZH_PROTOTYPES_EMBEDDING_SAVE_PATH):
        zh_prototype_encodings = np.load(ZH_PROTOTYPES_EMBEDDING_SAVE_PATH)
    else:
        zh_prototype_encodings = encode_sentence_chinese([prototype["text-zh"] for prototype in prototype_list])
        np.save(ZH_PROTOTYPES_EMBEDDING_SAVE_PATH, zh_prototype_encodings)
    print(f"zh_prototype_encodings.shape: {zh_prototype_encodings.shape}")
    
    # calculating data embeddings
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
    
    #### 步骤 1-3：分类
    #### Step 1-3: Classification
    en_classification_output = nearse_neighbour(en_data_encodings, en_prototype_encodings)
    zh_classification_output = nearse_neighbour(zh_data_encodings, zh_prototype_encodings)
    print(f"en_classification_output: {en_classification_output}")
    print(f"zh_classification_output: {zh_classification_output}")
