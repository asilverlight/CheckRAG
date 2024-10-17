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
# from pipeline import CheckRAG_v2, CheckRAG
from pipeline_api import CheckRAG
import argparse
import asyncio

config_dict = {
    'data_dir': '/data00/yifei_chen/multi_llms_for_CoT/datasets/nq/test.jsonl',
    'index_path': '/data00/jiajie_jin/flashrag_indexes/wiki_dpr_100w/e5_flat_inner.index',#'/data00/yifei_chen/FlashRAG/examples/quick_start/indexes/e5_Flat.index',
    #,# ,'
    'corpus_path': '/data00/jiajie_jin/flashrag_indexes/wiki_dpr_100w/wiki_dump.jsonl',##'/data00/yifei_chen/FlashRAG/examples/quick_start/indexes/general_knowledge.jsonl',None,,
    'retrieval_method': 'bge',#'contriever',#
    'retrieval_model_path': "/data00/yifei_chen/multi_llms_for_CoT/models/BAAI/bge-base-en-v1___5",#"/data00/yifei_chen/multi_llms_for_CoT/models/contriever",#
}

config_inference = {
    'batch_size': 5,
    'RAG_type': 'check',
    'use_refiner': False,
    'use_rethink': True,# 是否需要多轮思考
    'batch_process': True,
    'use_few_shot': True,
    'few_shot_path': "",
    'judgments_rounds': 3,# 需要重新思考的轮数
    'modify_rounds':3,
    'simplify_rounds': 3,
    'result_path': '/data00/yifei_chen/multi_llms_for_CoT/results/2wikimultihopqa/checkRAG.jsonl',
    'result_path_indent4': "",
    'hlcn_type': ["factual incorrectness", "fabrication"],#'logical error', 'question-answer inconsisitency', 
    # 新加了一个misinterpretation
    'classifier': 
        {
            'model_name': 'llama3-8B-instruct',#'llama2-7B-chat',#'qwen2-7B-instruct',
            'model_path': '/data00/LLaMA-3-8b-Instruct/',#'/data00/yifei_chen/FlashRAG/models/shakechen/Llama-2-7b-chat-hf',#'/data00/yifei_chen/BERT_classification/models/qwen/Qwen2-7B-Instruct/',
            'max_input_len': 8000,
            "framework": "hf",
            'type': torch.bfloat16,
            'stop_token_ids': [151329,
                        151336,
                        151338],
            'gpu_use':0.8,
            'generator_params':
                {
                    'max_tokens': 512,
                    'temperature': 1,
                    'top_p': 0.7,
                }
        },
    'simplifier': 
        {
            'model_name': 'llama3-8B-instruct',#'llama2-7B-chat',#'qwen2-7B-instruct',
            'model_path': '/data00/LLaMA-3-8b-Instruct/',#'/data00/yifei_chen/FlashRAG/models/shakechen/Llama-2-7b-chat-hf',#'/data00/yifei_chen/BERT_classification/models/qwen/Qwen2-7B-Instruct/',
            'max_input_len': 8000,
            "framework": "hf",
            'type': torch.bfloat16,
            'stop_token_ids': [151329,
                        151336,
                        151338],
            'gpu_use':0.8,
            'generator_params':
                {
                    'max_tokens': 512,
                    'temperature': 1,
                    'top_p': 0.7,
                }
        },
    'modifier': 
        {
            'model_name': 'llama3-8B-instruct',#'llama2-7B-chat',#'qwen2-7B-instruct',
            'model_path': '/data00/LLaMA-3-8b-Instruct/',#'/data00/yifei_chen/FlashRAG/models/shakechen/Llama-2-7b-chat-hf',#'/data00/yifei_chen/BERT_classification/models/qwen/Qwen2-7B-Instruct/',
            'max_input_len': 8000,
            "framework": "hf",
            'type': torch.bfloat16,
            'stop_token_ids': [151329,
                        151336,
                        151338],
            'gpu_use':0.8,
            'generator_params':
                {
                    'max_tokens': 512,
                    'temperature': 1,
                    'top_p': 0.7,
                }
        },
    'checker': 
        {
            'model_name': 'llama3-8B-instruct',#'llama2-7B-chat',#'qwen2-7B-instruct',
            'model_path': '/data00/LLaMA-3-8b-Instruct/',#'/data00/yifei_chen/FlashRAG/models/shakechen/Llama-2-7b-chat-hf',#'/data00/yifei_chen/BERT_classification/models/qwen/Qwen2-7B-Instruct/',
            'max_input_len': 8000,
            "framework": "hf",
            'type': torch.bfloat16,
            'stop_token_ids': [151329,
                        151336,
                        151338],
            'gpu_use':0.8,
            'generator_params':
                {
                    'max_tokens': 512,
                    'temperature': 1,
                    'top_p': 0.7,
                }
        },
    'ensembler':
        {
            'llama3-8B-instruct':{
                'model_name': 'llama3-8B-instruct',#'llama2-7B-chat',#'qwen2-7B-instruct',
                'model_path': '/data00/LLaMA-3-8b-Instruct/',#'/data00/yifei_chen/FlashRAG/models/shakechen/Llama-2-7b-chat-hf',#'/data00/yifei_chen/BERT_classification/models/qwen/Qwen2-7B-Instruct/',
                'max_input_len': 8000,
                "framework": "hf",
                'type': torch.bfloat16,
                'stop_token_ids': [151329,
                            151336,
                            151338],
                'gpu_use':0.9,
                'generator_params':
                    {
                        'max_tokens': 512,
                        'temperature': 0,
                        'top_p': 0.7,
                    },
                'port': '114514'
            },
            'qwen2.5-7b-instruct':{
                'model_name': 'qwen2.5-7b-instruct',#'llama2-7B-chat',#'qwen2-7B-instruct',
                'model_path': '/data00/yifei_chen/multi_llms_for_CoT/models/Qwen/Qwen2___5-7B-Instruct',#'/data00/yifei_chen/FlashRAG/models/shakechen/Llama-2-7b-chat-hf',#'/data00/yifei_chen/BERT_classification/models/qwen/Qwen2-7B-Instruct/',
                'max_input_len': 8000,
                "framework": "hf",
                'type': torch.bfloat16,
                'stop_token_ids': [151329,
                            151336,
                            151338],
                'gpu_use':0.9,
                'generator_params':
                    {
                        'max_tokens': 512,
                        'temperature': 0,
                        'top_p': 0.7,
                    },
                'port': '1919810'
            },
            'mistral-7B-instruct-v0.3':{
                'model_name': 'mistral-7B-instruct-v0.3',#'llama2-7B-chat',#'qwen2-7B-instruct',
                'model_path': "/data00/jiajie_jin/model/Mistral-7B-Instruct-v0.3",#'/data00/yifei_chen/FlashRAG/models/shakechen/Llama-2-7b-chat-hf',#'/data00/yifei_chen/BERT_classification/models/qwen/Qwen2-7B-Instruct/',
                'max_input_len': 8000,
                "framework": "hf",
                'type': torch.bfloat16,
                'stop_token_ids': [151329,
                            151336,
                            151338],
                'gpu_use':0.9,
                'generator_params':
                    {
                        'max_tokens': 512,
                        'temperature': 0,
                        'top_p': 0.7,
                    },
                'port': '1557'
            },
        },
    'generator':
        {
            'model_name': 'qwen2.5-7b-instruct',#'llama2-7B-chat',#'qwen2-7B-instruct',
            'model_path': "/data00/yifei_chen/multi_llms_for_CoT/models/Qwen/Qwen2___5-7B-Instruct",#'/data00/yifei_chen/FlashRAG/models/shakechen/Llama-2-7b-chat-hf',#'/data00/yifei_chen/BERT_classification/models/qwen/Qwen2-7B-Instruct/',
            'max_input_len': 8000,
            "framework": "hf",
            'type': torch.bfloat16,
            'stop_token_ids': [151329,
                        151336,
                        151338],
            'gpu_use':0.9,
            'generator_params':
                {
                    'max_tokens': 512,
                    'temperature': 0,
                    'top_p': 0.7,
                },
            'port': '4396'
        },   
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
    parser = argparse.ArgumentParser()
    # parser.add_argument("--result_path", type=str)
    # parser.add_argument("--result_path_indent4", type=str)
    parser.add_argument('--RAG_type', type=str)
    # parser.add_argument('--data_dir', type=str)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--use_few_shot', action='store_true')
    parser.add_argument('--few_shot_path', type=str, default="")
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--is_multiple_choice', action='store_true')
    parser.add_argument('--rethink_rounds', type=int, default=3)
    parser.add_argument('--judgments_rounds', type=int, default=3)
    parser.add_argument('--modify_rounds', type=int, default=3)
    parser.add_argument('--simplify_rounds', type=int, default=3)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top_p', type=float, default=0.9)
    parser.add_argument('--pipeline_type', type=str, default='multi_hlcn')
    args = parser.parse_args()
    config_inference['RAG_type'] = args.RAG_type
    config_inference['batch_size'] = args.batch_size
    config_inference['use_few_shot'] = args.use_few_shot
    config_inference['few_shot_path'] = args.few_shot_path
    config_inference['result_path'] = '/data00/yifei_chen/multi_llms_for_CoT/results/' + args.dataset_name + '/' + args.RAG_type + 'RAG.jsonl'
    config_inference['result_path_indent4'] = '/data00/yifei_chen/multi_llms_for_CoT/results/' + args.dataset_name + '/' + args.RAG_type + 'RAG_indent4.jsonl'
    config_dict['data_dir'] = '/data00/yifei_chen/multi_llms_for_CoT/datasets/' + args.dataset_name + '/test.jsonl'
    config_inference['rethink_rounds'] = args.rethink_rounds
    config_inference['judgments_rounds'] = args.judgments_rounds
    config_inference['modify_rounds'] = args.modify_rounds
    config_inference['simplify_rounds'] = args.simplify_rounds
    # config_inference['generator']['generator_params']['temperature'] = args.temperature
    # config_inference['generator']['generator_params']['top_p'] = args.top_p
    config_inference['pipeline_type'] = args.pipeline_type
    for model_config in config_inference['ensembler'].values():
        model_config['generator_params']['temperature'] = args.temperature
        model_config['generator_params']['top_p'] = args.top_p
    config = Config("/data00/yifei_chen/multi_llms_for_CoT/flashrag/config/basic_config.yaml", config_dict=config_dict)
    # print(config)
    # models, tokenizers, sample_params = load_model(config_inference, pipeline_type=config_inference['pipeline_type'])
    retriever = get_retriever(config)
    # return
    test_data = get_dataset(config)
    golden_answers = get_dataset(config, value="golden_answers")
    
    # test_data = test_data[:10]
    # golden_answers = golden_answers[:10]
    if config_inference['pipeline_type'] == 'multi_hlcn':
        pipeline = CheckRAG(config=config_inference, models=models, tokenizers=tokenizers, retriever=retriever, sample_params=sample_params)
    else:
        pipeline = CheckRAG(config=config_inference, retriever=retriever)
    if args.is_multiple_choice:
        choices = get_dataset(config, value="choices")
        # choices = choices[:20]
        pipeline.run_multiple_choice(test_data, choices, golden_answers)
    else:
        retrieval_results = retriever.batch_search(test_data)
        asyncio.run(pipeline.run(test_data, golden_answers, retrieval_results))
        # pipeline.run_v2(test_data, golden_answers)
    
if __name__ == '__main__':
    run()