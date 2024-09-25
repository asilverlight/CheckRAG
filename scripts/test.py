import sys
sys.path.append("..")
import random
import os
from tqdm import tqdm
import datetime, time
import random
import argparse
import json
import ujson
from flashrag.evaluator.utils import normalize_answer

naive_path = "/data00/yifei_chen/multi_llms_for_CoT/datasets/nq/results/naiveRAG.jsonl"
check_path = "/data00/yifei_chen/multi_llms_for_CoT/results/nq/checkRAG.jsonl"
golden_answer = []
differ_check = []
differ_naive = []
result = []

# with open(naive_path, 'r', encoding='utf-8') as fr:
#     for line in fr:
#         data = ujson.loads((line))
#         golden_answer.append(data['golden answer'])
#         differ_naive.append(data)
        
with open(check_path, 'r', encoding='utf-8') as fr:
    for line in fr:
        differ_check.append(ujson.loads((line)))
        
for data in differ_check:
    if 'No need change' in data['answer']:
        print(data['answer'])
        
# for i in range(len(differ_check)):
#     temp_check = normalize_answer(differ_check[i]['answer'])
#     temp_naive = normalize_answer((differ_naive[i]['answer']))
#     for golden in golden_answer[i]:
#         golden = normalize_answer((golden))
#         if golden == temp_naive and golden != temp_check:
#             result.append((differ_check[i]))
            
# res_path = "/data00/yifei_chen/multi_llms_for_CoT/results/nq/differ_naive_check.jsonl"
# with open(res_path, 'w', encoding='utf-8') as fw:
#     for data in result:
#         fw.write(ujson.dumps(data, indent=4) + '\n')

# # rng = np.random.default_rng(1557)



# data_path = "/data00/yifei_chen/multi_llms_for_CoT/datasets/nq/nq_test.jsonl"
# sample_data_path = "/data00/yifei_chen/multi_llms_for_CoT/datasets/nq/test.jsonl"
# data = []
# sample_data = []
# with open(data_path, 'r', encoding='utf-8') as fr:
#     for line in fr:
#         data.append(json.loads(line))

# with open(sample_data_path, 'r', encoding='utf-8') as fr:
#     for line in fr:
#         sample_data.append(json.loads(line))
        
# res = []
# mylist = [d["id"] for d in sample_data]
# while True:
#     random_data = random.sample(data, 3)
#     if all(metadata not in sample_data for metadata in random_data):
#         print(random_data)
#         break
    
# s = '\nApril 1st'
# print(s.lstrip('\n'))
import re

def remove_substring_and_surrounding_newlines(s, substring):
    # 使用正则表达式匹配子串及其左右可能存在的空行
    pattern = re.compile(r'\n*\s*' + re.escape(substring) + r'\s*\n*', re.IGNORECASE)
    # 替换匹配到的子串及其左右空行为单个空行
    modified_s = re.sub(pattern, '\n', s)
    return modified_s.strip()  # 去除字符串首尾的空行

# 示例使用
input_string = """
This is a test string.


This part should be removed.

The rest of the text.
"""

cleaned_string = remove_substring_and_surrounding_newlines(input_string, "No need change")
print(cleaned_string)
