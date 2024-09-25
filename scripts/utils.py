import sys
sys.path.append("..")
from flashrag.config import Config
from flashrag.utils import get_dataset, get_generator, get_retriever, get_reranker
from modelscope import snapshot_download
import ujson
import random
import os
import torch
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
import json
import re


def load_model(config, is_test=True, is_naiverag=False):
    # 加载要用的model和tokenizer
    # RAG，modify，ensemble：llama2
    # decompose，rewrite：llama3
    # judgment：baichuan，mistral，qwen，glm
    models = {}
    tokenizers = {}
    if not is_test:
        for key, value in config_inference.items():
            if key != 'judgers':
                models[key] = AutoModelForCausalLM.from_pretrained(
                    value['model_path'],
                    torch_dtype="auto",
                    device_map="auto",
                    trust_remote_code=True,
                ).to(torch.device(value['device']))
        model_j = [
            AutoModelForCausalLM.from_pretrained(
                    value['model_path'],
                    torch_dtype="auto",
                    device_map="auto",
                    trust_remote_code=True,
                ).to(torch.device(value['device']))
            for value in config_inference['judgers']
        ]
        models['judgers'] = model_j
        
        for key, value in config_inference.items():
            if key != 'judgers':
                tokenizers[key] = AutoTokenizer.from_pretrained(
                    value['model_path'],
                    trust_remote_code=True
                )# .to(torch.device(value['device']))
        tokenizer_j = [
            AutoTokenizer.from_pretrained(
                    value['model_path'],
                    trust_remote_code=True
                )# .to(torch.device(value['device']))
            for value in config_inference['judgers']
        ]
        tokenizers['judgers'] = tokenizer_j
    else:
        # models['decomposer'] = AutoModelForCausalLM.from_pretrained(
        #             config_inference['decomposer']['model_path'],
        #             torch_dtype=config_inference['decomposer']['type'],
        #             device_map=config_inference['decomposer']['device'],
        #             trust_remote_code=True,
        #         )# .to(torch.device(config_inference['decomposer']['device']))
        # # models['decomposer'] = None
        # models['rewritter'] = models['decomposer']
        # models['generator'] = models['decomposer']
        # models['modifier'] = models['decomposer']
        # models['ensembler'] = models['decomposer']
        
        # tokenizers['decomposer'] = AutoTokenizer.from_pretrained(
        #             config_inference['decomposer']['model_path'],
        #             trust_remote_code=True
        #         )# .to(torch.device(config_inference['decomposer']['device']))
        # # tokenizers['decomposer'] = None
        # tokenizers['rewritter'] = tokenizers['decomposer']
        # tokenizers['generator'] = tokenizers['decomposer']
        # tokenizers['modifier'] = tokenizers['decomposer']
        # tokenizers['ensembler'] = tokenizers['decomposer']
        
        # model_j = [
        #     AutoModelForCausalLM.from_pretrained(
        #             value['model_path'],
        #             device_map=value['device'],
        #             torch_dtype=value['type'],
        #             trust_remote_code=True,
        #         )# .to(torch.device(value['device']))
        #     for value in config_inference['generators']
        # ]
        
        # models['generators'] = model_j
        # tokenizer_j = [
        #     AutoTokenizer.from_pretrained(
        #             value['model_path'],
        #             trust_remote_code=True,
        #             model_max_length=value['max_input_len']
        #         )# .to(torch.device(value['device']))
        #     for value in config_inference['generators']
        # ]
        
        # tokenizers['generators'] = tokenizer_j
        # models['refiner'] = models['generators'][0]
        # tokenizers['refiner'] = tokenizers['generators'][0]
        # models['ensembler'] = models['generators'][0]
        # tokenizers['ensembler'] = tokenizers['generators'][0]
        models['generator'] = AutoModelForCausalLM.from_pretrained(
                    config['generator']['model_path'],
                    device_map=config['generator']['device'],
                    torch_dtype=config['generator']['type'],
                    trust_remote_code=True,
                )# .to(torch.device(value['device']))
        # models['modifier'] = AutoModelForCausalLM.from_pretrained(
        #             config['modifier']['model_path'],
        #             device_map=config['modifier']['device'],
        #             torch_dtype=config['modifier']['type'],
        #             trust_remote_code=True,
        #         )
        models['modifier'] = models['generator']
        models['checker'] = models['modifier']
        models['rethinker'] = models['checker']
        
        tokenizers['generator'] = AutoTokenizer.from_pretrained(
                    config['generator']['model_path'],
                    trust_remote_code=True,
                    model_max_length=config['generator']['max_input_len']
                )
        # tokenizers['modifier'] = AutoTokenizer.from_pretrained(
        #             config['modifier']['model_path'],
        #             trust_remote_code=True,
        #             model_max_length=config['modifier']['max_input_len']
        #         )
        tokenizers['modifier'] = tokenizers['generator']
        tokenizers['checker'] = tokenizers['modifier']
        tokenizers['rethinker'] = tokenizers['checker']
        
    return models, tokenizers

def get_dataset(config, data_dir='data_dir', value='question'):
    
    data_path = config[data_dir]
    questions = []
    with open(data_path, 'r', encoding='utf-8') as fr:
        for line in fr:
            data = json.loads(line)
            questions.append(data[value])
    return questions

# def get_result(config):
#     result_path = config['result_path']
#     results = []
#     with open(result_path, 'r', encoding='utf-8') as fr:
#         for line in fr:
#             data = ujson.loads(line)
#             results.append(data['answer'])
#     return results

# def get_golden_answer(config):
def pooling(pooler_output, last_hidden_state, attention_mask=None, pooling_method="mean"):
    if pooling_method == "mean":
        last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    elif pooling_method == "cls":
        return last_hidden_state[:, 0]
    elif pooling_method == "pooler":
        return pooler_output
    else:
        raise NotImplementedError("Pooling method not implemented!")
    
def remove_substring(s, substring):
    # 使用正则表达式匹配子串及其左右可能存在的空行
    pattern = re.compile(r'\n*\s*' + re.escape(substring) + r'\s*\n*', re.IGNORECASE)
    # 替换匹配到的子串及其左右空行为单个空行
    modified_s = re.sub(pattern, '\n', s)
    return modified_s.strip()  # 去除字符串首尾的空行
    