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
rng = np.random.default_rng(1557)

dataset_name = "2wikimultihopqa"
data_source_path = "/data00/yifei_chen/multi_llms_for_CoT/datasets/" + dataset_name + "/dev.jsonl"#"/data00/yifei_chen/multi_llms_for_CoT/datasets/2wikimultihopqa/dev.jsonl"#"/data00/yifei_chen/multi_llms_for_CoT/datasets/nq/nq_test.jsonl"
result_path = "/data00/yifei_chen/multi_llms_for_CoT/datasets/" + dataset_name + "/test.jsonl"#"/data00/yifei_chen/multi_llms_for_CoT/datasets/2wikimultihopqa/test.jsonl"
sample_path = "/data00/yifei_chen/multi_llms_for_CoT/datasets/" + dataset_name + "/sample_example.jsonl"#"/data00/yifei_chen/multi_llms_for_CoT/datasets/2wikimultihopqa/sample_example.jsonl"
sample_path_indent4 = "/data00/yifei_chen/multi_llms_for_CoT/datasets/" + dataset_name + "/sample_example_indent4.jsonl"#"/data00/yifei_chen/multi_llms_for_CoT/datasets/2wikimultihopqa/sample_example_indent4.jsonl"
new_result_path = "/data00/yifei_chen/multi_llms_for_CoT/datasets/" + dataset_name + "/test_flashrag.jsonl"#"/data00/yifei_chen/multi_llms_for_CoT/datasets/2wikimultihopqa/test.jsonl"
datas = []
with open(data_source_path, 'r', encoding='utf-8') as fr:
    for line in fr:
        data = ujson.loads(line)
        temp_dict = dict()
        temp_dict['id'] = data['id']
        temp_dict['question'] = data['question']
        temp_dict['golden_answers'] = data['golden_answers']
        # content = data['metadata']['context']['content']
        # temp_dict['documents'] = [content_[0] for content_ in content]
        datas.append(temp_dict)
datas_ = rng.choice(datas, 1000, replace=False)
with open(result_path, 'w', encoding='utf-8') as fw:
    for data in datas_:
        fw.write(ujson.dumps(data) + '\n')




# with open(result_path, 'r', encoding='utf-8') as fr:
#     for line in fr:
#         data = ujson.loads(line)
#         datas.append(data['id'])
        
# datas_ = []
# with open(data_source_path, 'r', encoding='utf-8') as fr:
#     for line in fr:
#         data = ujson.loads(line)
#         if data['id'] in datas:
#             datas_.append(data)
            
# with open(new_result_path, 'w', encoding='utf-8') as fw:
#     for data in datas_:
#         fw.write(ujson.dumps(data) + '\n')
        


# dev_msmarco_path = "/data00/yifei_chen/multi_llms_for_CoT/datasets/msmarcoqa/dev.jsonl"
# train_msmarco_path = "/data00/yifei_chen/multi_llms_for_CoT/datasets/msmarcoqa/train.jsonl"
# test_msmarco_path = "/data00/yifei_chen/multi_llms_for_CoT/datasets/msmarcoqa/test.jsonl"
# # 从dev中抽取500条作为test，剩下作为train，不重复
# datas = []
# with open(dev_msmarco_path, 'r', encoding='utf-8') as fr:
#     for line in fr:
#         data = ujson.loads(line)
#         temp_dict = dict()
#         temp_dict['id'] = data['id']
#         temp_dict['question'] = data['question']
#         temp_dict['golden_answers'] = data['golden_answers']
#         datas.append(temp_dict)

# rng.shuffle(datas)
# test_datas = datas[:500]
# train_datas = datas[500:]
# with open(test_msmarco_path, 'w', encoding='utf-8') as fw:
#     for data in test_datas:
#         fw.write(ujson.dumps(data) + '\n')
# with open(train_msmarco_path, 'w', encoding='utf-8') as fw:
#     for data in train_datas:
#         fw.write(ujson.dumps(data) + '\n')