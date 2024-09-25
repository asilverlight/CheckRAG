import sys
sys.path.append("..")
import os
import torch
import ujson
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import BertTokenizer, BertModel, AdamW, BertForSequenceClassification, AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
import numpy as np
import torch
from modelscope import snapshot_download
import re
from typing import List
from flashrag.config import Config
from flashrag.utils import get_retriever
from utils import load_model, get_dataset
from pipeline import CheckRAG

config_dict = {
    'data_dir': '/data00/yifei_chen/multi_llms_for_CoT/datasets/nq/test.jsonl',
    'index_path': '/data00/jiajie_jin/flashrag_indexes/wiki_dpr_100w/e5_flat_inner.index',#'/data00/yifei_chen/FlashRAG/examples/quick_start/indexes/e5_Flat.index',
    #,# ,'
    'corpus_path': '/data00/jiajie_jin/flashrag_indexes/wiki_dpr_100w/wiki_dump.jsonl',##'/data00/yifei_chen/FlashRAG/examples/quick_start/indexes/general_knowledge.jsonl',None,,
}

config_inference = {
    'batch_size': 4,
    'RAG_type': 'check',
    'use_refiner': False,
    'use_rethink': True,# 是否需要多轮思考
    'rethinking_rounds': 1,# 需要重新思考的轮数
    'result_path': '/data00/yifei_chen/multi_llms_for_CoT/results/nq/checkRAG.jsonl',
    'hlcn_type': ["factual incorrectness", "fabrication"],#"misinterpretation", "logical inconsistency", 
    'generator': 
        {
            'model_name': 'glm-4-9b-chat',#'llama3-8B-instruct',#'qwen2-7B-instruct',
            'model_path': '/data00/yifei_chen/multi_llms_for_CoT/models/ZhipuAI/glm-4-9b-chat',#'/data00/LLaMA-3-8b-Instruct/',#'/data00/yifei_chen/BERT_classification/models/qwen/Qwen2-7B-Instruct/',
            'max_input_len': 2048,
            'device': 'cuda:0',
            "framework": "hf",
            'type': torch.bfloat16,
            'generator_params':
                {
                    'do_sample': True,
                    'max_new_tokens': 2048,
                    'temperature': 1,
                    'top_p': 0.7,
                }
        },
    # 'checker':
    #     {
    #         'model_name': 'glm-4-9b-chat',
    #         'model_path': '/data00/yifei_chen/multi_llms_for_CoT/models/ZhipuAI/glm-4-9b-chat',
    #         'max_input_len': 2048,
    #         'device': 'cuda:1',
    #         'type': torch.bfloat16,
    #         "framework": "hf",
    #         'generator_params':
    #             {
    #                 'do_sample': True,
    #                 'max_new_tokens': 512,
    #                 'temperature': 1,
    #                 'top_p': 0.7,
    #             }
    #     },
    # 'modifier':
    #     {
    #         'model_name': 'glm-4-9b-chat',
    #         'model_path': '/data00/yifei_chen/multi_llms_for_CoT/models/ZhipuAI/glm-4-9b-chat',
    #         'max_input_len': 1024,
    #         'device': 'cuda:1',
    #         'type': torch.bfloat16,
    #         "framework": "hf",
    #         'generator_params':
    #             {
    #                 'do_sample': True,
    #                 'max_new_tokens': 512,
    #                 'temperature': 1,
    #                 'top_p': 0.7,
    #             }
    #     },
}

def run():
    config = Config("/data00/yifei_chen/multi_llms_for_CoT/flashrag/config/basic_config.yaml", config_dict=config_dict)
    test_data = get_dataset(config)
    retriever = get_retriever(config)
    models, tokenizers = load_model(config_inference, is_naiverag=False)
    golden_answers = get_dataset(config, value="golden_answers")
    # test_data = test_data[:3]
    # golden_answers = golden_answers[:3]
    pipeline = CheckRAG(config=config_inference, models=models, tokenizers=tokenizers, retriever=retriever)
    pipeline.run(test_data, golden_answers)
    
if __name__ == '__main__':
    run()