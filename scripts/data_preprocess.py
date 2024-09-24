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

data_source_path = "/data00/yifei_chen/multi_llms_for_CoT/datasets/nq/nq_test.jsonl"
result_path = "/data00/yifei_chen/multi_llms_for_CoT/datasets/nq/test.jsonl"
datas = []
with open(data_source_path, 'r', encoding='utf-8') as fr:
    for line in fr:
        data = ujson.loads(line)
        temp_dict = dict()
        temp_dict['id'] = data['id']
        temp_dict['question'] = data['question']
        temp_dict['golden_answers'] = data['golden_answers']
        datas.append(temp_dict)
datas = random.sample(datas, 100)
with open(result_path, 'w', encoding='utf-8') as fw:
    for data in datas:
        fw.write(ujson.dumps(data) + '\n')