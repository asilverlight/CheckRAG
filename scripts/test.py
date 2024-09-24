import sys
sys.path.append("..")
import random
import os
from tqdm import tqdm
import datetime, time
import random
import argparse
import json
# # rng = np.random.default_rng(1557)



data_path = "/data00/yifei_chen/multi_llms_for_CoT/datasets/nq/nq_test.jsonl"
sample_data_path = "/data00/yifei_chen/multi_llms_for_CoT/datasets/nq/test.jsonl"
data = []
sample_data = []
with open(data_path, 'r', encoding='utf-8') as fr:
    for line in fr:
        data.append(json.loads(line))

with open(sample_data_path, 'r', encoding='utf-8') as fr:
    for line in fr:
        sample_data.append(json.loads(line))
        
res = []
mylist = [d["id"] for d in sample_data]
while True:
    random_data = random.sample(data, 3)
    if all(metadata not in sample_data for metadata in random_data):
        print(random_data)
        break
    
s = '\nApril 1st'
print(s.lstrip('\n'))