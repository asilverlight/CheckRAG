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
from vllm import LLM, SamplingParams
# torch.cuda.empty_cache()
# import gc
# gc.collect()
# torch.cuda.device_count()

# available_gpus = [i for i in range(torch.cuda.device_count())]
# print("Available GPUs:", available_gpus)
# import os

# os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'


def load_model(config, is_test=True, pipeline_type='multi_hlcn'):
    # 加载要用的model和tokenizer
    # RAG，modify，ensemble：llama2
    # decompose，rewrite：llama3
    # judgment：baichuan，mistral，qwen，glm
    models = {}
    tokenizers = {}
    sample_params = {}
    generation_params = {}
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
        
        # models['checker'] = LLM(
        #     config['checker']['model_path'],
        #     dtype=config['checker']['type'],
        #     enforce_eager=True,
        #     trust_remote_code=True,
        #     max_model_len=4096,
        #     gpu_memory_utilization=0.5,
        #     device=torch.device('cuda:0'),
        # )
        # tokenizers['checker'] = models['checker'].get_tokenizer()
        # generation_params['checker'] = dict()
        # generation_params['checker'].update(config['checker']['generator_params'])
        # # generation_params['checker']['max_input_len'] = config['checker']['max_input_len']
        # # generation_params['checker']["stop_token_ids"] = config['checker']['stop_token_ids']#[tokenizers['checker'].eos_token_id, 128001, 128009]#
        # sample_params['checker'] = SamplingParams(
        #     **generation_params['checker'],
        #     stop_token_ids=config['checker']['stop_token_ids'],
        #     )
        
        # os.environ["CUDA_VISIBLE_DEVICES"] = "2"
        models['generator'] = LLM(
            config['generator']['model_path'],
            dtype=config['generator']['type'],
            enforce_eager=True,
            trust_remote_code=True,
            max_model_len=config['generator']['max_input_len'],
            gpu_memory_utilization=config['generator']['gpu_use'],
            # tensor_parallel_size=2,
        )
        tokenizers['generator'] = models['generator'].get_tokenizer()
        generation_params['generator'] = dict()
        generation_params['generator'].update(config['generator']['generator_params'])
        if 'llama' in config['generator']['model_name'].lower():
            generation_params['generator']['stop_token_ids'] = [tokenizers['generator'].eos_token_id, tokenizers['generator'].convert_tokens_to_ids("<|eot_id|>")]
        # generation_params['checker']['max_input_len'] = config['checker']['max_input_len']
        # generation_params['checker']["stop_token_ids"] = config['checker']['stop_token_ids']#[tokenizers['checker'].eos_token_id, 128001, 128009]#
            sample_params['generator'] = SamplingParams(
                **generation_params['generator'],
                )
        else:
            sample_params['generator'] = SamplingParams(
                **generation_params['generator'],
                # stop_token_ids=[tokenizers['generator'].eos_token_id]
                )
            
        # models['checker'] = LLM(
        #     config['checker']['model_path'],
        #     dtype=config['checker']['type'],
        #     enforce_eager=True,
        #     trust_remote_code=True,
        #     max_model_len=4096,
        #     gpu_memory_utilization=config['checker']['gpu_use'],
        #     tensor_parallel_size=2,
        # )
        # tokenizers['checker'] = models['checker'].get_tokenizer()
        # generation_params['checker'] = dict()
        # generation_params['checker'].update(config['checker']['generator_params'])
        # if 'llama' in config['checker']['model_name'].lower():
        #     generation_params['checker']['stop_token_ids'] = [tokenizers['checker'].eos_token_id, tokenizers['checker'].convert_tokens_to_ids("<|eot_id|>")]
        # # generation_params['checker']['max_input_len'] = config['checker']['max_input_len']
        # # generation_params['checker']["stop_token_ids"] = config['checker']['stop_token_ids']#[tokenizers['checker'].eos_token_id, 128001, 128009]#
        #     sample_params['checker'] = SamplingParams(
        #         **generation_params['checker'],
        #         )
        # else:
        #     sample_params['checker'] = SamplingParams(
        #         **generation_params['checker'],
        #         stop_token_ids=config['checker']['stop_token_ids']
        #         )
        
        
        
        if pipeline_type == 'multi_hlcn':
            models['modifier'] = models['generator']
            models['checker'] = models['generator']
            models['rethinker'] = models['checker']
            tokenizers['modifier'] = tokenizers['generator']
            tokenizers['checker'] = tokenizers['generator']
            tokenizers['rethinker'] = tokenizers['checker']
            sample_params['modifier'] = sample_params['generator']
            sample_params['checker'] = sample_params['generator']
            sample_params['rethinker'] = sample_params['checker']
        elif pipeline_type == 'single_hlcn':
            models['classifier'] = models['generator']
            models['simplifier'] = models['generator']
            models['modifier'] = models['generator']
            tokenizers['classifier'] = tokenizers['generator']
            tokenizers['simplifier'] = tokenizers['generator']
            tokenizers['modifier'] = tokenizers['generator']
            sample_params['classifier'] = sample_params['generator']
            sample_params['simplifier'] = sample_params['generator']    
            sample_params['modifier'] = sample_params['generator']
        
    return models, tokenizers, sample_params

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
    
def retain_after_last_substring(s, substring):
    # 查找最后一个"user\n"子串的位置
    index = s.rfind(substring)
    
    # 如果找到了该子串，则截取该位置之后的所有字符
    if index != -1 and index + len(substring) < len(s):
        return s[index + len(substring):]
    else:
        # 如果没有找到该子串，则返回原字符串
        return s