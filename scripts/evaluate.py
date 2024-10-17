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
import argparse
# from multi_llms_for_CoT.scripts.run_multiRAG import config_inference
import re
# from inference import Checker, Generator, Modifier, Rethinker
# from inference_choose import Generator, Classifier, Modifier, Simplifier

config = {
    'metrics': ['em', 'f1', 'acc', 'bleu'],
    'result_path': '/data00/yifei_chen/multi_llms_for_CoT/results/2wikimultihopqa/checkRAG.jsonl',
    'golden_answer_path': '/data00/yifei_chen/multi_llms_for_CoT/datasets/2wikimultihopqa/results/naiveRAG.jsonl', 
    'save_dir': '/data00/yifei_chen/multi_llms_for_CoT/datasets/hotpotqa/LLM_ensemble_results.jsonl',
    'dataset_name': 'nq',
    'rag_naive': '/data00/yifei_chen/multi_llms_for_CoT/results/2wikimultihopqa/naiveRAG.jsonl',
    'RAG_type': 'check',
    'refiner_name': None,#'llama3-8B-instruct',
    'metric_setting':
        {
            'retrieval_recall_topk': 5,
            'tokenizer_name': 'llama3',
        }
}
config.update(config_inference)

def extract_numbers(s):
    # 使用正则表达式匹配数字
    numbers = re.findall(r'\b\d+\b', s)
    if numbers:
        return numbers
    else:
        return [s]

def evaluate():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_type', type=str, default='naive')
    # parser.add_argument('--result_path', type=str)
    # parser.add_argument('--result_save_log', type=str)
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--is_multiple_choice', action='store_true')
    parser.add_argument('--pipeline_type', type=str, default='multi_hlcn')
    args = parser.parse_args()
    rag_type = args.eval_type
    result_log = '/data00/yifei_chen/multi_llms_for_CoT/results/' + args.dataset_name + '/result.log'
    config['RAG_type'] = args.eval_type
    config['result_path'] = '/data00/yifei_chen/multi_llms_for_CoT/results/' + args.dataset_name + '/' + config['RAG_type'] + 'RAG.jsonl'
    config['dataset_name'] = args.dataset_name
    results = get_dataset(config, data_dir='result_path', value='answer')
    golden_answers = get_dataset(config, data_dir='result_path', value='golden answer')

    # with open('/data00/yifei_chen/multi_llms_for_CoT/datasets/2wikimultihopqa/LLM_ensemble_results_refiner_decompose.jsonl', 'r', encoding='utf-8') as fr:
    #     for line in fr:
    #         results_ensemble.append(ujson.loads(line))

    evaluator = Evaluator(config)
    # if rag_type == 'naive':
    #     results = get_dataset(config, 'rag_naive', 'answer')
    #     golden_answers = get_dataset(config, 'rag_naive', 'golden answer'
    if args.is_multiple_choice:
        results = [extract_numbers(result)[0] for result in results]
        golden_answers = [str(golden_answer[0] + 1) for golden_answer in golden_answers]
    eval_results = evaluator.evaluate(results, golden_answers)
    
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    generators = config['generator']['model_name']
    if rag_type != 'naive':
        # log_entry = f"[{current_time}] Result: {eval_results}, Generators: {generators}, Dataset: {config['dataset_name']}, Refiner: {config['refiner_name']}, Hallucinations: {config['hlcn_type']}, Rethinking Rounds: {config['rethinking_rounds']}, Simplify Rounds: {config['simplify_rounds']}\n"
        log_entry = f"[{current_time}] Result: {eval_results}, Generators: {generators}, Dataset: {config['dataset_name']}, Judgment Rounds: {config['judgments_rounds']}, Modify Rounds: {config['modify_rounds']}, Simplify Rounds: {config['simplify_rounds']}\n"
    else:
        log_entry = f"[{current_time}] Result: {eval_results}, RAG type: {rag_type}, Dataset: {config['dataset_name']}, Refiner: {config['refiner_name']}\n"
    log_file = open(result_log, 'a')
    log_file.write(log_entry)
    if rag_type != 'naive':
        log_file_prompt = open('/data00/yifei_chen/multi_llms_for_CoT/results/prompt.log', 'a')
        if args.pipeline_type =='single_hlcn':
            from inference_choose import Generator, Classifier, Modifier, Simplifier
            generator_prompt = Generator().system_prompt
            classifier_prompt = Classifier().system_prompt
            classifier_prompt_rethink = Classifier().system_prompt_rethink
            modifier_prompt = Modifier().system_prompt
            modifier_prompt_rethink = Modifier().system_prompt_rethink
            simplifier_prompt = Simplifier().system_prompt
            log_entry = dict()
            log_entry['current_time'] = current_time
            log_entry['dataset_name'] = config['dataset_name']
            log_entry['generator_prompt'] = generator_prompt
            log_entry['classifier_prompt'] = classifier_prompt
            log_entry['classifier_prompt_rethink'] = classifier_prompt_rethink
            log_entry['modifier_prompt'] = modifier_prompt
            log_entry['modifier_prompt_rethink'] = modifier_prompt_rethink
            log_entry['simplifier_prompt'] = simplifier_prompt
            log_file_prompt.write(ujson.dumps(log_entry, indent=4) + '\n')
        elif args.pipeline_type =='multi_hlcn':
            from inference import Checker, Generator, Modifier, Rethinker
            generator_prompt = Generator().system_prompt
            checker_prompt = Checker().system_prompt
            checker_prompt_format = Checker().system_prompt_format_checker
            modifier_prompt = Modifier().system_prompt
            rethinker_prompt = Rethinker().system_prompt
            log_entry = dict()
            log_entry['current_time'] = current_time
            log_entry['dataset_name'] = config['dataset_name']
            log_entry['generator_prompt'] = generator_prompt
            log_entry['checker_prompt'] = checker_prompt
            log_entry['checker_prompt_format'] = checker_prompt_format
            log_entry['modifier_prompt'] = modifier_prompt
            log_entry['rethinker_prompt'] = rethinker_prompt
            log_file_prompt.write(ujson.dumps(log_entry, indent=4) + '\n')
    print(eval_results)
    
if __name__ == '__main__':
    evaluate()
    