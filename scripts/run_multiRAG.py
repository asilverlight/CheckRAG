import sys
sys.path.append("..")
from flashrag.config import Config
from flashrag.utils import get_retriever, get_reranker, get_generator
from scripts.pipeline import SequentialPipeline, MultiRAG
from modelscope import snapshot_download
import ujson
import random
import os
import torch
import ujson
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import BertTokenizer, BertModel, AdamW, BertForSequenceClassification, AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
import datetime, time
import random
import numpy as np
import argparse
import torch
from scripts.utils import load_model, get_dataset

# model_dir = snapshot_download(model_id="ZhipuAI/glm-4-9b-chat",
#                               cache_dir="/data00/yifei_chen/multi_llms_for_CoT/models")
# print(model_dir)
config_dict = {
    'data_dir': '/data00/yifei_chen/multi_llms_for_CoT/datasets/nq/test.jsonl',
    'index_path': '/data00/jiajie_jin/flashrag_indexes/wiki_dpr_100w/e5_flat_inner.index',
    #,# ,'/data00/yifei_chen/FlashRAG/examples/quick_start/indexes/e5_Flat.index'
    'corpus_path': '/data00/jiajie_jin/flashrag_indexes/wiki_dpr_100w/wiki_dump.jsonl',#None,,'/data00/yifei_chen/FlashRAG/examples/quick_start/indexes/general_knowledge.jsonl'
    # ,# 
    #'retrieval_cache_path': '/data00/yifei_chen/multi_llms_for_CoT/datasets/wiki_dump/json/default-bfb1d23b390d4c13/0.0.0/7483f22a71512872c377524b97484f6d20c275799bb9e7cd8fb3198178d8220a/',
    'max_input_len': 1024,
    'device': 'cuda:2',
    "framework": "hf",
    'do_sample': True,
    'max_new_tokens': 1024,
    'temperature': 1,
    'top_p': 0.7,
    'framework': "hf",
    'generator_model': 'llama3-8B-instruct',# baichuan2-7B-chat
    'generator_model_path': "/data00/LLaMA-3-8b-Instruct/",
    'batch_size': 4,
    'result_path': '/data00/yifei_chen/multi_llms_for_CoT/datasets/nq/results/ensembleRAG.jsonl',
    'RAG_type': 'ensemble',
    'retrieval_batch_size': 4,
    'use_refiner': False,
    'use_retrieval_cache': False,
    'save_retrieval_cache': False,
    # 'device': 'cuda',
    # 'cache_dir': "/data00/yifei_chen/multi_llms_for_CoT/datasets",
}

config_inference = {
    'ensembler':
        {
            'model_name': 'llama3-8B-instruct',
            'model_path': '/data00/LLaMA-3-8b-Instruct/',
            'max_input_len': 1024,
            'device': 'cuda:1',
            "framework": "hf",
            'type': torch.bfloat16,
            'generator_params':
                {
                    'do_sample': True,
                    'max_new_tokens': 1024,
                    'temperature': 1,
                    'top_p': 0.7,
                },
            'refiner_topk': 5,
            'refiner_pooling_method': 'mean',
            'refiner_encode_max_length': 1024,
            'refiner_max_input_length': 1024,
            'refiner_max_output_length': 1024,
        },
    'refiner':
        {
            'model_name': 'llama3-8B-instruct',
            'model_path': '/data00/LLaMA-3-8b-Instruct/',
            'max_input_len': 1024,
            'device': 'cuda',
            "framework": "hf",
            'type': torch.bfloat16,
            'generator_params':
                {
                    'do_sample': True,
                    'max_new_tokens': 1024,
                    'temperature': 1,
                    'top_p': 0.7,
                },
            'refiner_topk': 5,
            'refiner_pooling_method': 'mean',
            'refiner_encode_max_length': 1024,
            'refiner_max_input_length': 1024,
            'refiner_max_output_length': 1024,
        },
    'generators':
        [
            # {
            #     'model_name': 'baichuan2-13b-chat',
            #     'model_path': '/data00/yifei_chen/multi_llms_for_CoT/models/baichuan-inc/Baichuan2-13B-Chat',
            #     'max_input_len': 1024,
            #     'device': 'cuda:2',
            #     'data_path': '/data00/yifei_chen/multi_llms_for_CoT/datasets/hotpotqa/sample_small_test.jsonl',
            #     "framework": "hf",
            #     'generator_params':
            #         {
            #             'do_sample': True,
            #             'max_new_tokens': 512,
            #             'temperature': 1,
            #             'top_p': 0.7,
            #         }
            # },
            {
                'model_name': 'llama3-8B-instruct',
                'model_path': '/data00/LLaMA-3-8b-Instruct/',
                'max_input_len': 1024,
                'device': 'cuda:2',
                'data_path': '/data00/yifei_chen/multi_llms_for_CoT/datasets/hotpotqa/sample_small_test.jsonl',
                "framework": "hf",
                'type': torch.bfloat16,
                'generator_params':
                    {
                        'do_sample': True,
                        'max_new_tokens': 512,
                        'temperature': 1,
                        'top_p': 0.7,
                    }
            },
            {
                'model_name': 'mistral-7B-instruct-v0.3',
                'model_path': '/data00/jiajie_jin/model/Mistral-7B-Instruct-v0.3',
                'max_input_len': 1024,
                'device': 'cuda:1',
                'data_path': '/data00/yifei_chen/multi_llms_for_CoT/datasets/hotpotqa/sample_small_test.jsonl',
                "framework": "hf",
                'type': torch.bfloat16,
                'generator_params':
                    {
                        'do_sample': True,
                        'max_new_tokens': 512,
                        'temperature': 1,
                        'top_p': 0.7,
                    }
            },
            {
                'model_name': 'qwen2-7B-instruct',
                'model_path': '/data00/yifei_chen/BERT_classification/models/qwen/Qwen2-7B-Instruct/',
                'max_input_len': 1024,
                
                'device': 'cuda:2',
                'data_path': '/data00/yifei_chen/multi_llms_for_CoT/datasets/hotpotqa/sample_small_test.jsonl',
                "framework": "hf",
                'type': torch.bfloat16,
                'generator_params':
                    {
                        'do_sample': True,
                        'max_new_tokens': 512,
                        'temperature': 1,
                        'top_p': 0.7,
                    }
            },
            # {
            #     'model_name': 'glm-4-9b-chat',
            #     'model_path': '/data00/yifei_chen/multi_llms_for_CoT/models/ZhipuAI/glm-4-9b-chat',
            #     'max_input_len': 1024,
            #     'device': 'cuda:0',
            #     'type': torch.bfloat16,
            #     'data_path': '/data00/yifei_chen/multi_llms_for_CoT/datasets/hotpotqa/sample_small_test.jsonl',
            #     "framework": "hf",
            #     'generator_params':
            #         {
            #             'do_sample': True,
            #             'max_new_tokens': 512,
            #             'temperature': 1,
            #             'top_p': 0.7,
            #         }
            # }
        ]
    
}

def run():
    # '/data00/jiajie_jin/flashrag_indexes/wiki_dpr_100w/e5_flat_inner.index'
    # '/data00/jiajie_jin/flashrag_indexes/wiki_dpr_100w/wiki_dump.jsonl'
    config = Config("/data00/yifei_chen/multi_llms_for_CoT/flashrag/config/basic_config.yaml", config_dict=config_dict)
    test_data = get_dataset(config)
    retriever = get_retriever(config)
    # retriever = None
    models, tokenizers = load_model(config_inference, is_naiverag=False)
    golden_answers = get_dataset(config, value="golden_answers")

    # models['judgers'].append(models['decomposer'])
    # tokenizers['judgers'].append(tokenizers['decomposer'])
    # models['judgers'] = [models['decomposer']] * 3
    # tokenizers['judgers'] = [tokenizers['decomposer']] * 3
    # models = None
    # tokenizers = None

    # config_inference['rewritter'] = config_inference['decomposer']
    # config_inference['generator'] = config_inference['decomposer']
    # config_inference['modifier'] = config_inference['decomposer']
    # config_inference['ensembler'] = config_inference['decomposer']
    
    pipeline = MultiRAG(config=config, config_inference=config_inference, retriever=retriever, models=models, tokenizers=tokenizers)

    # pipeline = SequentialPipeline(config=config, config_inference=config_inference, retriever=retriever, models=models, tokenizers=tokenizers)
    # pipeline.run_naive_RAG(test_data)
    pipeline.run(test_data, golden_answers)

    # pipeline = SequentialPipeline(config)

    # output_dataset = pipeline.run(test_data, do_eval=False)
    # print("---generation output---")
    # print(output_dataset.pred)
if __name__ == '__main__':
    run()