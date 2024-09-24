import sys
sys.path.append("..")
from flashrag.evaluator import Evaluator
from flashrag.dataset.utils import split_dataset, merge_dataset
from flashrag.utils import get_retriever, get_generator, get_refiner, get_judger
from flashrag.prompt import PromptTemplate
import datetime, time
import copy
from tqdm import tqdm
from flashrag.dataset.dataset import Dataset
from flashrag.evaluator import Evaluator
import ujson
import torch
import os
from utils import get_dataset
import datetime, time
from run_checkRAG import config_inference
# from multi_llms_for_CoT.scripts.run_multiRAG import config_inference
import json

config = {
    'metrics': ['em', 'f1', 'acc'],
    'result_path': '/data00/yifei_chen/multi_llms_for_CoT/datasets/nq/results/checkRAG.jsonl',
    'golden_answer_path': '/data00/yifei_chen/multi_llms_for_CoT/datasets/nq/results/naiveRAG.jsonl', 
    'save_dir': '/data00/yifei_chen/multi_llms_for_CoT/datasets/hotpotqa/LLM_ensemble_results.jsonl',
    'dataset_name': 'nq',
    'rag_naive': '/data00/yifei_chen/multi_llms_for_CoT/datasets/nq/results/naiveRAG.jsonl',
    'RAG_type': 'check generator',
    'refiner_name': None,#'llama3-8B-instruct',
}
config.update(config_inference)

results = [
    "The Pyrenees or the Sistema Central mountains.",
    "The Prestige",
    "\"A Thousand Years\"",
    "No director information available.",
    "John Steinbeck",
    "No information is available.",
    "There is no answer.",
    "None",
    "Corruption",
    "Wolf Creek",
    "Milium",
    "None",
    "Not enough information.",
    "No",
    "Nobel Prize in Literature",
    "Yes",
    "The answer cannot be determined based on the provided documents.",
    "No",
    "La Belle Assembl\u00e9e",
    "Up"
]

def evaluate(rag_type):
    results = get_dataset(config, 'result_path', 'answer')
    golden_answers = get_dataset(config, 'result_path', 'golden answer')

    # results_naiverag = get_dataset(config, 'rag_naive', 'answer')
    # results_ensemble = get_dataset(config, 'result_path', 'answer')
    results_ensemble = []
    # with open('/data00/yifei_chen/multi_llms_for_CoT/datasets/2wikimultihopqa/LLM_ensemble_results_refiner_decompose.jsonl', 'r', encoding='utf-8') as fr:
    #     for line in fr:
    #         results_ensemble.append(ujson.loads(line))

    evaluator = Evaluator(config)
    if rag_type == 'naive':
        results = get_dataset(config, 'rag_naive', 'answer')
    else:
        result = results_ensemble
    eval_results = evaluator.evaluate(results, golden_answers)
    
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    generators = config['generator']['model_name']
    if rag_type != 'naive':
        log_entry = f"[{current_time}] Result: {eval_results}, Generators: {generators}, Dataset: {config['dataset_name']}, Refiner: {config['refiner_name']}\n"
    else:
        log_entry = f"[{current_time}] Result: {eval_results}, RAG type: {rag_type}, Dataset: {config['dataset_name']}, Refiner: {config['refiner_name']}\n"
    log_file = open('/data00/yifei_chen/multi_llms_for_CoT/datasets/nq/result.log', 'a')
    log_file.write(log_entry)
    print(eval_results)
    
if __name__ == '__main__':
    evaluate('naive')
    