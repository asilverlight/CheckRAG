import sys
sys.path.append("..")
import random
import os
from tqdm import tqdm
import datetime, time
import random
import argparse
import re
import ujson
from flashrag.evaluator.utils import normalize_answer

dataset_name = "hotpotqa"
naive_path = "/data00/yifei_chen/multi_llms_for_CoT/results/"+ dataset_name + "/naiveRAG.jsonl"
check_path = "/data00/yifei_chen/multi_llms_for_CoT/results/"+ dataset_name + "/checkRAG.jsonl"# 最终结果
check = []
naive = []
result = []
save_path = "/data00/yifei_chen/multi_llms_for_CoT/results/"+ dataset_name + "/differ_naive_check.jsonl"
source_data_path = "/data00/yifei_chen/multi_llms_for_CoT/datasets/truthfulqa/dev.jsonl"


# from modelscope import snapshot_download
# model_dir = snapshot_download('BAAI/bge-base-en-v1.5', cache_dir='/data00/yifei_chen/multi_llms_for_CoT/models/')



# with open(check_path, 'r', encoding='utf-8') as fr_c, open(naive_path, 'r', encoding='utf-8') as fr_n:
#     for line in fr_n:
#         data = ujson.loads(line)
#         differ_naive.append(data)
#     for line in fr_c:
#         data = ujson.loads(line)
#         differ_check.append(data)
    
def golden_answer_in_retrieval(golden_answer, retrieval_results):
    count = 0
    for retrieval_result in retrieval_results:
        for golden_answer_ in golden_answer:
            pattern = r'\b' + re.escape(golden_answer_) + r'\b'
            if re.search(pattern, retrieval_result['contents']):
                count += 1
                break
    return count
        
        
# for i in range(len(differ_check)):
#     golden_answers = differ_check[i]['golden answer']
#     if not any(golden_answer.lower() in differ_check[i]['answer'].lower() for golden_answer in golden_answers) and any(golden_answer.lower() in differ_naive[i]['answer'].lower() for golden_answer in golden_answers):#:# and not any(golden_answer.lower() in differ_check[i]['answer'].lower() for golden_answer in golden_answers):
#         result.append(differ_check[i])
#         # print(result)
# new_results = []
# for data in result:
#     if golden_answer_in_retrieval(data['golden answer'], data['retrieval result']):
#         new_results.append(data)
        
# print(len(new_results))
# with open(save_path, 'w', encoding='utf-8') as fw:
#     for data_ in new_results:
#         fw.write(ujson.dumps(data_, indent=4) + '\n')


with open(check_path, 'r', encoding='utf-8') as fr:
    for line in fr:
        data = ujson.loads(line)
        check.append(data)
        
# # 1、先找出irel docs和rel docs
# doc_situations = [[] for i in range(len(check[0]["retrieval result"]) + 1)]
# # print(check[0]["retrieval result"][0])
# for data in check:
#     count = golden_answer_in_retrieval(data['golden answer'], data['retrieval result'])
#     # print(count)
#     doc_situations[count].append(data)
    
def find_answer_check(datas, first_answer_type, second_answer_type):
    first_answers = []
    second_answers = []
    for data in datas:
        if first_answer_type == 'wrong':
            if all(golden_answer not in data['initial answer'] and 'no information' not in data['initial answer'].lower() for golden_answer in data['golden answer']):
                first_answers.append(data)
        elif first_answer_type == 'correct':
            if any(golden_answer in data['initial answer'] for golden_answer in data['golden answer']):
                first_answers.append(data)
        elif first_answer_type == 'no information':
            if 'no information' in data['initial answer'].lower():
                first_answers.append(data)
    for data in first_answers:
        if second_answer_type == 'wrong':
            if all(golden_answer not in data['answer'] and 'no information' not in data['answer'].lower() for golden_answer in data['golden answer']):
                second_answers.append(data)
        elif second_answer_type == 'correct':
            if any(golden_answer in data['answer'] for golden_answer in data['golden answer']):
                second_answers.append(data)
        elif second_answer_type == 'no information':
            if 'no information' in data['answer'].lower():
                second_answers.append(data)
    
    with open(save_path, 'w', encoding='utf-8') as fw:
        for data_ in second_answers:
            fw.write(ujson.dumps(data_, indent=4) + '\n')
    return round(len(second_answers) / len(first_answers) if len(first_answers) > 0 else 0, 4)
first_answer_type = ['wrong', 'correct', 'no information']
second_answer_type = ['wrong', 'correct', 'no information']
# _ = find_answer_check(check, 'no information', 'correct')
# for first_type in first_answer_type:
#     for second_type in second_answer_type:
#         print(f"The naiverag answer is {first_type}, The modified answer is {second_type}, P({second_type}|{first_type}): {find_answer(check, first_type, second_type)}")

def find_answer_naiverag(datas, retrieval_type, answer_type):
    retrieval_results = []
    rel_results = []
    for data in datas:
        for retrieval_result in data['retrieval result']:
            pattern = [r'\b' + re.escape(golden_answer) + r'\b' for golden_answer in data['golden answer']]
            if any(re.search(pattern[i], retrieval_result['contents']) for i in range(len(pattern))):
                rel_results.append(data)
                break
    irrel_results = [data for data in datas if data not in rel_results]
    if retrieval_type == 'irrel':
        retrieval_results = irrel_results
    else:
        retrieval_results = rel_results
    naiverag = []
    for data in retrieval_results:
        if answer_type == 'wrong':
            if all(golden_answer.lower() not in data['answer'].lower() and 'no information' not in data['answer'].lower() for golden_answer in data['golden answer']):
                naiverag.append(data)
        elif answer_type == 'correct':
            if any(golden_answer.lower() in data['answer'].lower() for golden_answer in data['golden answer']):
                naiverag.append(data)
        elif answer_type == 'no information':
            if 'no information' in data['answer'].lower():
                naiverag.append(data)
    with open(save_path, 'w', encoding='utf-8') as fw:
        for data_ in naiverag:
            fw.write(ujson.dumps(data_, indent=4) + '\n')
    return round(len(naiverag) / len(retrieval_results) if len(retrieval_results) > 0 else 0, 4)
retrieval_type = ['rel', 'irrel']
answer_type = ['wrong', 'correct', 'no information']
_ = find_answer_naiverag(check, 'irrel', 'correct')
# for retrieval_type_ in retrieval_type:
#     for answer_type_ in answer_type:
#         print(f"The retrieval documents are {retrieval_type_}, The answer type is {answer_type_}, P({answer_type_}|{retrieval_type_}): {find_answer_naiverag(check, retrieval_type_, answer_type_)}")