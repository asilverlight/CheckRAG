# import sys
# sys.path.append("..")
from flashrag.evaluator import Evaluator
from flashrag.dataset.utils import split_dataset, merge_dataset
from flashrag.utils import get_retriever, get_generator, get_refiner, get_judger
from flashrag.prompt import PromptTemplate
import datetime, time
import copy
from tqdm import tqdm
from flashrag.dataset.dataset import Dataset
# from inference import AbstractiveRecompRefiner, Checker, Generator, Modifier, Rethinker, ExtractiveRefiner
import ujson
from utils import remove_substring, retain_after_last_substring
from multiprocessing import Pool, cpu_count
import copy
import string
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel
import gc
import torch
import asyncio
from openai import OpenAI
from multiprocessing import cpu_count
import numpy as np
import re
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class BasicPipeline:
    """Base object of all pipelines. A pipeline includes the overall process of RAG.
    If you want to implement a pipeline, you should inherit this class.
    """

    def __init__(self, config, prompt_template=None):
        self.config = config
        self.batch_size = config['batch_size']
        self.result_path = config['result_path']
        self.result_path_indent4 = config['result_path_indent4']
        self.rag_type = config['RAG_type']
        self.use_refiner = config['use_refiner']
        self.use_rethink = config['use_rethink']
        self.batch_process = config['batch_process']
        self.rethink_rounds = config['rethink_rounds']
        self.judgments_rounds = config['judgments_rounds']
        self.modify_rounds = config['modify_rounds']
        self.simplify_rounds = config['simplify_rounds']
        self.hlcn_type = config['hlcn_type']
        self.use_few_shot = config['use_few_shot']
        self.few_shot_path = config['few_shot_path']
        self.few_shot_examples = []
        if self.use_few_shot:
            with open(self.few_shot_path, 'r', encoding='utf-8') as fr:
                for line in fr:
                    data = ujson.loads(line)
                    self.few_shot_examples.append(data)

    def run(self, dataset):
        """The overall inference process of a RAG framework."""
        pass
    
    def split_line(self, *args):
        pass   
    
    def distributed(self):
        pass     


        
    
        
    
        
            
class CheckRAG(BasicPipeline):
    def __init__(self, config, retriever=None, models=None, tokenizers=None, sample_params=None):
        super().__init__(config)    
        from inference import AbstractiveRecompRefiner, Checker, Generator, Modifier, Rethinker, ExtractiveRefiner     
        self.retriever = retriever
        self.checker = Checker(config['checker'], models['checker'], tokenizers['checker'], sample_params['checker'])
        self.generator = Generator(config['generator'], models['generator'], tokenizers['generator'], sample_params['generator'])
        self.modifier = Modifier(config['generator'], models['modifier'], tokenizers['modifier'], sample_params['modifier'])
        self.refiner = None
        if self.use_rethink:
            self.rethinker = Rethinker(config['checker'], models['rethinker'], tokenizers['rethinker'], sample_params['rethinker'])
        if self.use_refiner:
            self.refiner = AbstractiveRecompRefiner(config['refiner'], models['refiner'], tokenizers['refiner'])

                    
    def run(self, questions, golden_answers=None):
        start_time = time.time()
        prefixes = ['answer: ', 'answer:']
        initial_results = []
        results = []
        modify_results = []
        retrieval_results = []
        simplified_results = []
        retrieval_results = self.retriever.batch_search(questions)
        retrieval_results = retrieval_results[0]
        # print(self.result_path)
        # return
        # print(retrieval_results[0])
        # return
        
        # 第一步：生成init result
        for i in tqdm(range(0, len(questions), self.batch_size), desc='Original Answer'):
            questions_ = questions[i:i + self.batch_size]
            retrieval_results_ = retrieval_results[i:i + self.batch_size]
            init_results = self.generator.inference(questions_, retrieval_results_, rag_type=self.rag_type)
            init_results = [init_result.lstrip('\n') for init_result in init_results]
            # print(init_results)
            if self.rag_type == 'naive':
                results.extend(init_results)
            else:
                initial_results.extend(init_results)
                
                
        if self.rag_type == 'naive':
            print(results)
            with open(self.result_path, 'w', encoding='utf-8') as fw:
                for question, retrieval_result, result, golden_answer in zip(questions, retrieval_results, results, golden_answers):
                    temp_dict = dict()
                    temp_dict['question'] = question
                    temp_dict['retrieval result'] = retrieval_result
                    temp_dict['answer'] = result
                    temp_dict['golden answer'] = golden_answer
                    fw.write(ujson.dumps(temp_dict) + '\n')
                    
            # save_path = '/data00/yifei_chen/multi_llms_for_CoT/results/nq/naiveRAG_indent4.jsonl'
            with open(self.result_path, 'r', encoding='utf-8') as fr, open(self.result_path_indent4, 'w', encoding='utf-8') as fw:
                for line in fr:
                    fw.write(ujson.dumps(ujson.loads(line), indent=4) + '\n')
            end_time = time.time()
            print(f'Program run time for {self.rag_type}: {(end_time - start_time)/60}min')
                    
            return
        # 第二步：开始修正判断
        initial_results = [retain_after_last_substring(retain_after_last_substring(initial_result, 'user\n'), 'Answer: ') for initial_result in initial_results]
        
        final_hlcn_judgments = [{} for _ in range(len(questions))]
        # 首先进行初步判断修正
        for hlcn in self.hlcn_type:
            original_judgment_results = []# 初步对某一幻觉推理的结果
            for i in tqdm(range(0, len(questions), self.batch_size), desc=f'First Judgment for {hlcn}'):
                questions_single = questions[i:min(i + self.batch_size, len(questions))]
                retrieval_result = retrieval_results[i:min(i + self.batch_size, len(questions))]
                init_results_single = initial_results[i:min(i + self.batch_size, len(questions))]
                original_judgment_results_single = self.checker.inference(
                    questions_single, 
                    retrieval_result, 
                    init_results_single, 
                    rag_type=self.rag_type,
                    hlcn_type=hlcn,
                    )
                original_judgment_results_single = [retain_after_last_substring(original_judgment_results_single_, 'user\n') for original_judgment_results_single_ in original_judgment_results_single]
                original_judgment_results.extend(original_judgment_results_single)
            for i in range(len(final_hlcn_judgments)):
                final_hlcn_judgments[i][hlcn] = [original_judgment_results[i]]
                
        # 接下来rethink之前的验证结果
        for hlcn in self.hlcn_type:
            ids = list(range(len(questions)))
            for i in range(self.judgments_rounds):
                rethink_questions = [questions[id] for id in ids]# 取出要用的question
                rethink_init_results = [initial_results[id] for id in ids] # 取出要用的initial results
                rethink_check_results = [final_hlcn_judgments[id][hlcn][-1] for id in ids] # 取出上一轮的check结果
                retrieval_results_single = [retrieval_results[id] for id in ids] # 取出需要的检索结果
                
                this_round_check_results = []# 这一轮即将得到的修正结果
                for j in tqdm(range(0, len(rethink_check_results), self.batch_size), desc=f'Round {i+1} for {hlcn}'):
                    rethink_questions_single = rethink_questions[j:min(j + self.batch_size, len(rethink_questions))]
                    rethink_retrieval_results = retrieval_results_single[j:min(j + self.batch_size, len(retrieval_results_single))]
                    rethink_init_results_single = rethink_init_results[j:min(j + self.batch_size, len(rethink_init_results))]
                    rethink_check_single = rethink_check_results[j:min(j + self.batch_size, len(rethink_check_results))]
                    # print(rethink_check_single)
                    # return
                    rethink_new_results = self.rethinker.inference(
                        rethink_questions_single, 
                        rethink_retrieval_results, 
                        rethink_init_results_single,
                        rethink_check_single,
                        hlcn_type=hlcn
                        )
                    rethink_new_results = [retain_after_last_substring(rethink_new_results_, 'user\n') for rethink_new_results_ in rethink_new_results]
                    this_round_check_results.extend(rethink_new_results)# 得到这一轮新的修正结果
                # 之后，找出接着需要rethink的结果
                temp_ids = []
                for j in range(len(this_round_check_results)):
                    if not ('The judgment is correct' in this_round_check_results[j] 
                            or 'the judgment is correct' in this_round_check_results[j]) and this_round_check_results[j] != final_hlcn_judgments[ids[j]][hlcn][-1]:# 仍然需要rethink
                        temp_ids.append(ids[j])
                        final_hlcn_judgments[ids[j]][hlcn].append(this_round_check_results[j])
                ids = copy.deepcopy(temp_ids)
                if len(ids) == 0:
                    break
        
                    
        # 处理final_hlcn_judgments的格式
        # 现在的格式：
        # [
        #     {
        #         'hlcn1': [
                    
        #         ],
        #         'hlcn2': [
                    
        #         ],
        #         'hlcn3': [
                    
        #         ]
        #     }
        #     ......
        # ]
        # 目标格式：
        # [
        #     [0-3个]
        #     [0-3个]
        #     ......
        # ]  
        temp_final_hlcn_judgments = [[] for _ in range(len(questions))]
        for i in range(len(final_hlcn_judgments)):
            for hlcn in self.hlcn_type:
                # if "The answer has " not in final_hlcn_judgment[hlcn][-1]:
                temp_final_hlcn_judgments[i].append(final_hlcn_judgments[i][hlcn][-1])
        hlcn_judgments_trajectory = copy.deepcopy(final_hlcn_judgments)
        final_hlcn_judgments = copy.deepcopy(temp_final_hlcn_judgments)
                
        # 第三步：开始集成
        result_to_be_simplify = copy.deepcopy(initial_results)# 集成过后等待简化的结果
        simplify_to_be_modify = [[i, s] for i, s in enumerate(final_hlcn_judgments) if len(s) != 0]
        for j in tqdm(range(0, len(simplify_to_be_modify), self.batch_size), desc='Modify Answer'):
            result_to_be_modify_batch = simplify_to_be_modify[j:min(j + self.batch_size, len(simplify_to_be_modify))]# 等待集成的结果，是一个元组
            questions_ = questions[j:min(j + self.batch_size, len(simplify_to_be_modify))]
            retrieval_results_ = retrieval_results[j:min(j + self.batch_size, len(simplify_to_be_modify))]
            init_results_ = initial_results[j:min(j + self.batch_size, len(simplify_to_be_modify))]
            result2bmodify_ = [result_to_be_modify_batch_single[1] for result_to_be_modify_batch_single in result_to_be_modify_batch]# 等待集成的结果，是一个list包list
            result_ = self.modifier.inference(questions_, retrieval_results_, init_results_, result2bmodify_, rag_type=self.rag_type)
            # 集成出的最终结果，是一个list
            for k in range(len(result_)):
                result_to_be_simplify[result_to_be_modify_batch[k][0]] = result_[k]
                
        # 第四步：开始简化
        result_to_be_simplify = [retain_after_last_substring(retain_after_last_substring(result_to_be_simplify_, 'user\n'), 'Answer: ') for result_to_be_simplify_ in result_to_be_simplify]
        final_results = [[res] for res in result_to_be_simplify]
        ids = list(range(len(final_results)))
        for j in range(self.simplify_rounds):
            result_to_be_simplify = [final_results[id][-1] for id in ids]# 等待简化的结果
            questions_ = [questions[id] for id in ids]
            temp_ids = []
            results_batch = []# 简化处理后的结果
            for k in tqdm(range(0, len(result_to_be_simplify), self.batch_size), desc=f'Round {j+1} for Simplify'):
                questions_single = questions_[k:min(k + self.batch_size, len(result_to_be_simplify))]
                result2bsimplify_single = result_to_be_simplify[k:min(k + self.batch_size, len(result_to_be_simplify))]
                results_ = self.checker.inference(questions_single, result2bsimplify_single, check_format=True)
                results_ = [retain_after_last_substring(retain_after_last_substring(result_, 'user\n'), 'Answer: ') for result_ in results_]
                results_batch.extend(results_)
            for m in range(len(results_batch)):
                if 'No need change' not in results_batch[m]:# 需要继续简化 and not (results_batch[m] == final_results[ids[m]][-1])
                    temp_ids.append(ids[m])
                    final_results[ids[m]].append(results_batch[m])
            ids = copy.deepcopy(temp_ids)# 新的待简化答案的id
            if len(ids) == 0:
                break
        results = [final_result[-1] for final_result in final_results]
        # print(results)
        results = [result[0] if isinstance(result, list) else result for result in results]
        # 第五步：后处理
        for result in results:
            result = remove_substring(result, "No need change")
            result = result.lstrip('\n')
            result = result.strip(string.punctuation)
            for prefix in prefixes:
                if result.lower().startswith(prefix):
                    result = result[len(prefix):]
        
        print(results)
        # print(self.result_path)
        
        with open(self.result_path, 'w', encoding='utf-8') as fw:
            for question, retrieval_result, result, golden_answer, check_result, init_result, sim_res in zip(questions, retrieval_results, results, golden_answers, hlcn_judgments_trajectory, initial_results, final_results):
                temp_dict = dict()
                temp_dict['question'] = question
                temp_dict['retrieval result'] = retrieval_result
                temp_dict['initial answer'] = init_result
                temp_dict['judgments'] = check_result
                temp_dict['simplify process'] = sim_res
                temp_dict['answer'] = result
                temp_dict['golden answer'] = golden_answer
                fw.write(ujson.dumps(temp_dict) + '\n')
                
        # save_path = '/data00/yifei_chen/multi_llms_for_CoT/results/nq/checkRAG_indent4.jsonl'
        with open(self.result_path, 'r', encoding='utf-8') as fr, open(self.result_path_indent4, 'w', encoding='utf-8') as fw:
            for line in fr:
                fw.write(ujson.dumps(ujson.loads(line), indent=4) + '\n')
        end_time = time.time()
        print(f'Program run time for {self.rag_type}: {(end_time - start_time)/60}min')
        
    def run_multiple_choice(self, questions, options=None, golden_answers=None):
        start_time = time.time()
        prefixes = ['answer: ', 'answer:']
        initial_results = []# 最开始第一步generate的结果
        results = []# 最后的结果
        retrieval_results = self.retriever.batch_search(questions)
        retrieval_results = retrieval_results[0]
        
        # 第一步：生成init result
        for i in tqdm(range(0, len(questions), self.batch_size), desc='Original Answer'):
            questions_ = questions[i:i + self.batch_size]
            retrieval_results_ = retrieval_results[i:i + self.batch_size]
            options_ = options[i:i + self.batch_size]
            init_results = self.generator.inference(questions_, retrieval_results_, options_, rag_type=self.rag_type, multiple_choice=True)
            init_results = [init_result.lstrip('\n') for init_result in init_results]
            # print(init_results)
            if self.rag_type == 'naive':
                results.extend(init_results)
            else:
                initial_results.extend(init_results)
                
        if self.rag_type == 'naive':
            print(results)
            with open(self.result_path, 'w', encoding='utf-8') as fw:
                for question, retrieval_result, result, golden_answer in zip(questions, retrieval_results, results, golden_answers):
                    temp_dict = dict()
                    temp_dict['question'] = question
                    temp_dict['retrieval result'] = retrieval_result
                    temp_dict['answer'] = result
                    temp_dict['golden answer'] = golden_answer
                    fw.write(ujson.dumps(temp_dict) + '\n')
                    
            with open(self.result_path, 'r', encoding='utf-8') as fr, open(self.result_path_indent4, 'w', encoding='utf-8') as fw:
                for line in fr:
                    fw.write(ujson.dumps(ujson.loads(line), indent=4) + '\n')
            end_time = time.time()
            print(f'Program run time for {self.rag_type}: {(end_time - start_time)/60}min')
                    
            return
        # 第二步：开始修正判断
        final_hlcn_judgments = [{} for _ in range(len(questions))]
        # 首先进行初步判断修正
        for hlcn in self.hlcn_type:
            original_judgment_results = []# 初步对某一幻觉推理的结果
            for i in tqdm(range(0, len(questions), self.batch_size), desc=f'First Judgment for {hlcn}'):
                questions_single = questions[i:min(i + self.batch_size, len(questions))]
                retrieval_result = retrieval_results[i:min(i + self.batch_size, len(questions))]
                init_results_single = initial_results[i:min(i + self.batch_size, len(questions))]
                options_single = options[i:min(i + self.batch_size, len(questions))]
                original_judgment_results_single = self.checker.inference(
                    questions_single, 
                    retrieval_result, 
                    options_single,
                    init_results_single, 
                    rag_type=self.rag_type,
                    hlcn_type=hlcn,
                    multiple_choice=True
                    )
                original_judgment_results.extend(original_judgment_results_single)
            for i in range(len(final_hlcn_judgments)):
                final_hlcn_judgments[i][hlcn] = [original_judgment_results[i]]
                
        # 接下来rethink之前的验证结果
        for hlcn in self.hlcn_type:
            ids = list(range(len(questions)))
            for i in range(self.rethinking_rounds):
                rethink_questions = [questions[id] for id in ids]# 取出要用的question
                rethink_init_results = [initial_results[id] for id in ids] # 取出要用的initial results
                rethink_check_results = [final_hlcn_judgments[id][hlcn][-1] for id in ids] # 取出上一轮的check结果
                retrieval_results_single = [retrieval_results[id] for id in ids] # 取出需要的检索结果
                options_single = [options[id] for id in ids] # 取出需要的选项
                
                this_round_check_results = []# 这一轮即将得到的修正结果
                for j in tqdm(range(0, len(rethink_check_results), self.batch_size), desc=f'Round {i+1} for {hlcn}'):
                    rethink_questions_single = rethink_questions[j:min(j + self.batch_size, len(rethink_questions))]
                    rethink_retrieval_results = retrieval_results_single[j:min(j + self.batch_size, len(retrieval_results_single))]
                    rethink_init_results_single = rethink_init_results[j:min(j + self.batch_size, len(rethink_init_results))]
                    rethink_check_single = rethink_check_results[j:min(j + self.batch_size, len(rethink_check_results))]
                    rethink_options_single = options_single[j:min(j + self.batch_size, len(options_single))]
                    rethink_new_results = self.rethinker.inference(
                        rethink_questions_single, 
                        rethink_retrieval_results, 
                        rethink_options_single,
                        rethink_init_results_single,
                        rethink_check_single,
                        hlcn_type=hlcn,
                        multiple_choice=True
                        )
                    this_round_check_results.extend(rethink_new_results)# 得到这一轮新的修正结果
                # 之后，找出接着需要rethink的结果
                temp_ids = []
                for j in range(len(this_round_check_results)):
                    if not ('No revise is required' in this_round_check_results[j] or 'no revise is required' in this_round_check_results[j]):# 仍然需要rethink
                        temp_ids.append(ids[j])
                        final_hlcn_judgments[ids[j]][hlcn].append(this_round_check_results[j])
                ids = copy.deepcopy(temp_ids)
                if len(ids) == 0:
                    break
        
                    
        # 处理final_hlcn_judgments的格式
        # 现在的格式：
        # [
        #     {
        #         'hlcn1': [
                    
        #         ],
        #         'hlcn2': [
                    
        #         ],
        #         'hlcn3': [
                    
        #         ]
        #     }
        #     ......
        # ]
        # 目标格式：
        # [
        #     [0-3个]
        #     [0-3个]
        #     ......
        # ]  
        temp_final_hlcn_judgments = [[] for _ in range(len(questions))]
        for i in range(len(final_hlcn_judgments)):
            final_hlcn_judgment = final_hlcn_judgments[i]
            for hlcn in self.hlcn_type:
                # if "The answer has " not in final_hlcn_judgment[hlcn][-1]:
                temp_final_hlcn_judgments[i].append(final_hlcn_judgments[i][hlcn][-1])
        hlcn_judgments_trajectory = copy.deepcopy(final_hlcn_judgments)
        final_hlcn_judgments = copy.deepcopy(temp_final_hlcn_judgments)
                
            
        # 第三步：开始集成
        result_to_be_simplify = copy.deepcopy(initial_results)# 集成过后等待简化的结果
        simplify_to_be_modify = [(i, s) for i, s in enumerate(final_hlcn_judgments) if len(s) != 0]
        for j in tqdm(range(0, len(simplify_to_be_modify), self.batch_size), desc='Modify Answer'):
            result_to_be_modify_batch = simplify_to_be_modify[j:min(j + self.batch_size, len(simplify_to_be_modify))]# 等待集成的结果，是一个元组
            questions_ = questions[j:min(j + self.batch_size, len(simplify_to_be_modify))]
            retrieval_results_ = retrieval_results[j:min(j + self.batch_size, len(simplify_to_be_modify))]
            init_results_ = initial_results[j:min(j + self.batch_size, len(simplify_to_be_modify))]
            options_ = options[j:min(j + self.batch_size, len(simplify_to_be_modify))]
            result2bmodify_ = [result_to_be_modify_batch_single[1] for result_to_be_modify_batch_single in result_to_be_modify_batch]# 等待集成的结果，是一个list包list
            result_ = self.modifier.inference(questions_, retrieval_results_, options_, init_results_, result2bmodify_, rag_type=self.rag_type, multiple_choice=True)
            # 集成出的最终结果，是一个list
            for k in range(len(result_)):
                result_to_be_simplify[result_to_be_modify_batch[k][0]] = result_[k]
        results = result_to_be_simplify
        print(results)
        # print(self.result_path)
        
        with open(self.result_path, 'w', encoding='utf-8') as fw:
            for question, retrieval_result, result, golden_answer, check_result, init_result in zip(questions, retrieval_results, results, golden_answers, hlcn_judgments_trajectory, initial_results):
                temp_dict = dict()
                temp_dict['question'] = question
                temp_dict['retrieval result'] = retrieval_result
                temp_dict['initial answer'] = init_result
                temp_dict['judgments'] = check_result
                temp_dict['answer'] = result
                temp_dict['golden answer'] = golden_answer
                fw.write(ujson.dumps(temp_dict) + '\n')
                
        with open(self.result_path, 'r', encoding='utf-8') as fr, open(self.result_path_indent4, 'w', encoding='utf-8') as fw:
            for line in fr:
                fw.write(ujson.dumps(ujson.loads(line), indent=4) + '\n')
                
        end_time = time.time()
        print(f'Program run time for {self.rag_type}: {(end_time - start_time)/60}min')
from inference_api import Generator, Rethinker, Simplifier, Classifier, Modifier    
class CheckRAG_v2(BasicPipeline): 
    def __init__(self, config, retriever):
        super().__init__(config)
        self.retriever = retriever
        # self.generator = Generator(models['generator'], tokenizers['generator'], sample_params['generator'])
        # self.classifier = Classifier(models['classifier'], tokenizers['classifier'], sample_params['classifier'])
        # self.simplifier = Simplifier(models['simplifier'], tokenizers['simplifier'], sample_params['simplifier'])
        # self.modifier = Modifier(models['modifier'], tokenizers['modifier'], sample_params['modifier'])
        self.model_names = list(self.config['ensembler'].keys())
        self.model_paths = [model['model_path'] for model in self.config['ensembler'].values()]
        self.model_load = {model_name: {} for model_name in self.model_names}
        self.base_urls = [model['port'] for model in self.config['ensembler'].values()]
        self.base_urls = [f'http://localhost:{base_url}/v1' for base_url in self.base_urls]
        for idx, key in enumerate(self.model_load.keys()):
            self.model_load[key]['model_path'] = self.model_paths[idx]
            self.model_load[key]['api_key'] = "aaa"
            self.model_load[key]['base_url'] = self.base_urls[idx]
        # self.generator_model = {
        #     'model_path': self.config['generator']['model_path'],
        #     'api_key': "aaa",
        #     'base_url': f'http://localhost:{self.config["generator"]["port"]}/v1',
        # }
        
    
        
    def load_model(self, config):
        model = LLM(
            config['model_path'],
            dtype=config['type'],
            enforce_eager=True,
            trust_remote_code=True,
            max_model_len=config['max_input_len'],
            gpu_memory_utilization=config['gpu_use'],
            # tensor_parallel_size=num_devices,
        )
        tokenizer = model.get_tokenizer()
        generation_params = dict()
        generation_params.update(config['generator_params'])
        if 'llama' in config['model_name'].lower():
            generation_params['stop_token_ids'] = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
        sample_params = SamplingParams(
            **generation_params,
        )
        return model, tokenizer, sample_params
    
    def del_model(self, model, tokenizer):
        del tokenizer
        destroy_model_parallel()
        del model.llm_engine.model_executor.driver_worker
        del model
        gc.collect()
        torch.cuda.empty_cache()
     
 
    def run_v2(self, questions, golden_answers):
        start_time = time.time()
        prefixes = ['answer: ', 'answer:']
        initial_results = []
        results = []
        retrieval_results = self.retriever.batch_search(questions)
        retrieval_results = retrieval_results[0]
        modified_results_all_rounds = [[[] for _ in range(len(questions))] for _ in range(self.judgments_rounds)]# 保存所有轮次的修正结果
        simplified_results_all_rounds = [[] for _ in range(len(questions))] # 保存所有轮次的简化结果
        
    # 第一步：生成init results
        config_generator = self.config['ensembler']['qwen2.5-7b-instruct']
        # print(config_generator)
        model, tokenizer, sample_params = self.load_model(config_generator)
        generator = Generator(model, tokenizer, sample_params)
        
        for i in tqdm(range(0, len(questions), self.batch_size), desc='Original Answer'):
            questions_ = questions[i:min(i + self.batch_size, len(questions))]
            retrieval_results_ = retrieval_results[i:min(i + self.batch_size, len(questions))]
            init_results = generator.inference(questions_, retrieval_results_, rag_type=self.rag_type)
            init_results = [init_result.lstrip('\n') for init_result in init_results]
            init_results = [retain_after_last_substring(retain_after_last_substring(init_result, 'user\n'), 'Answer: ') for init_result in init_results]
            if self.rag_type == 'naive':
                results.extend(init_results)
            else:
                initial_results.extend(init_results)
        if self.rag_type == 'naive':
            # results = [retain_after_last_substring(retain_after_last_substring(result, 'user\n'), 'Answer: ') for result in results]
            print(results)
            with open(self.result_path, 'w', encoding='utf-8') as fw:
                for question, retrieval_result, result, golden_answer in zip(questions, retrieval_results, results, golden_answers):
                    temp_dict = dict()
                    temp_dict['question'] = question
                    temp_dict['retrieval result'] = retrieval_result
                    temp_dict['answer'] = result
                    temp_dict['golden answer'] = golden_answer
                    fw.write(ujson.dumps(temp_dict) + '\n')
                    
            # save_path = '/data00/yifei_chen/multi_llms_for_CoT/results/nq/naiveRAG_indent4.jsonl'
            with open(self.result_path, 'r', encoding='utf-8') as fr, open(self.result_path_indent4, 'w', encoding='utf-8') as fw:
                for line in fr:
                    fw.write(ujson.dumps(ujson.loads(line), indent=4) + '\n')
            end_time = time.time()
            print(f'Program run time for {self.rag_type}: {(end_time - start_time)/60}min')
                    
            return
        
        self.del_model(model, tokenizer)
        # 第二步：开始判断hlcn种类   
        final_results = [[]]
        final_results[0] = initial_results
        hlcn_judgments = [
            [
                {
                    model_name: [] 
                    for model_name in self.model_names
                } for _ in range(len(questions))
            ] for i in range(self.judgments_rounds)
        ]
        # 第一层：判断轮数
        # 第二层：data数量
        # 第三层：模型名称
        # 第四层：模型判断结果
        for i in range(self.judgments_rounds):
            for model_name in self.model_names:
                model, tokenizer, sample_params = self.load_model(self.config['ensembler'][model_name])
                classifier = Classifier(model, tokenizer, sample_params)
                for j in tqdm(range(0, len(questions), self.batch_size), desc=f'Round {i+1} for Judgment for {model_name}'):
                    questions_batch = questions[j:min(j + self.batch_size, len(questions))]
                    retrieval_results_batch = retrieval_results[j:min(j + self.batch_size, len(questions))]
                    initial_results_batch = [final_results[-1][k] for k in range(j, min(j + self.batch_size, len(questions)))]
                    hlcn_judgments_batch = classifier.inference(
                        questions_batch, retrieval_results_batch, initial_results_batch
                    )
                    batch_indices = range(j, min(j + self.batch_size, len(questions)))
                    for index, hlcn_judgment in zip(batch_indices, hlcn_judgments_batch):
                        hlcn_judgments[i][index][model_name].append(hlcn_judgment)
                self.del_model(model, tokenizer)
            # 之后互相交换答案
            
            # 首先判断，是否所有模型都认为没有幻觉
            ids = list(range(len(questions)))
            for j in range(self.rethink_rounds):
                original_judgments = []
                temp_ids = []
                for id in ids:
                    temp_judgments_batch = [hlcn_judgments[i][id][model_name][-1] for model_name in self.model_names]
                    if any('does not have hallucination' not in judgment for judgment in temp_judgments_batch):# 存在幻觉，则加进来
                        temp_ids.append(id)
                        original_judgments.append(temp_judgments_batch)
                ids = copy.deepcopy(temp_ids)
                # print(ids)
                if len(ids) == 0:
                    break
                questions_judgments = [questions[id] for id in ids]
                retrieval_results_judgments = [retrieval_results[id] for id in ids]
                initial_results_judgments = [final_results[-1][id] for id in ids]
                # original_judgments = [[hlcn_judgments[i][id][model_name][-1] for model_name in self.model_names] for id in ids]
                other_judgments = [
                    [
                        [
                            hlcn_judgments[i][id][model_name_][-1]
                            for model_name_ in self.model_names 
                            if model_name_ != model_name
                        ] for model_name in self.model_names
                    ] for id in ids
                ]
                this_round_judgments = [[] for _ in range(len(self.model_names))]
                for model_name in self.model_names:
                    model, tokenizer, sample_params = self.load_model(self.config['ensembler'][model_name])
                    classifier = Classifier(model, tokenizer, sample_params)
                    for k in tqdm(range(0, len(questions_judgments), self.batch_size), desc=f'Rethink for {j+1} in {i+1} round for {model_name}'):
                        questions_batch = questions_judgments[k:min(k + self.batch_size, len(questions_judgments))]
                        retrieval_results_batch = retrieval_results_judgments[k:min(k + self.batch_size, len(retrieval_results_judgments))]
                        initial_results_batch = initial_results_judgments[k:min(k + self.batch_size, len(initial_results_judgments))]
                        original_judgments_batch = [original_judgments[m] for m in range(len(questions_batch))]
                        other_judgments_batch = [other_judgments[m] for m in range(len(questions_batch))]
                        # 现在得到的original_judgments_batch是长度为batch的list
                        # 每个元素有三个值
                        
                        # 现在得到的other_judgments_batch是长度为batch的list
                        # 每个元素是一个list，每个list有三个元素
                        # 每个元素也是一个长为2的list
                        # 即尺寸为batch*3*2
                        # 之后需要取每一个二维list的最后一个元素
                        rethink_new_judgments = classifier.inference(
                            questions_batch, retrieval_results_batch, initial_results_batch,
                            original_judgments_batch, other_judgments_batch, first_turn=False
                        )
                        
                        this_round_judgments[self.model_names.index(model_name)].extend(rethink_new_judgments)
                    self.del_model(model, tokenizer)
                this_round_judgments = [[row[m] for row in this_round_judgments] for m in range(len(this_round_judgments[0]))]
                for m in range(len(ids)):
                    for model_name, rethink_new_judgment in zip(self.model_names, this_round_judgments[m]):
                        # print([rethink_new_judgment])
                        # print('\n')
                        hlcn_judgments[i][ids[m]][model_name].append(rethink_new_judgment)
            
            # for j in range(self.rethink_rounds):
            #     for model_name in self.model_names:
            #         model, tokenizer, sample_params = self.load_model(self.config['ensembler'][model_name])
            #         classifier = Classifier(model, tokenizer, sample_params)
            #         for k in tqdm(range(0, len(questions), self.batch_size), desc=f'Round {i+1} for Rethink for {model_name} in Round {j+1}'):
            #             questions_batch = questions[k:min(k + self.batch_size, len(questions))]
            #             retrieval_results_batch = retrieval_results[k:min(k + self.batch_size, len(questions))]
            #             initial_results_batch = [final_results[-1][l] for l in range(k, min(k + self.batch_size, len(questions)))]
            #             original_judgments_batch = [hlcn_judgments[i][l][model_name][-1] for l in range(k, min(k + self.batch_size, len(questions)))]
            #             other_judgments_batch = [
            #                 [
            #                     hlcn_judgments[i][m][other_model_name][-1]
            #                     for other_model_name in self.model_names if other_model_name != model_name
            #                 ] for m in range(k, min(k + self.batch_size, len(questions)))
            #                 ]
            #             rethink_new_judgments = classifier.inference(
            #                 questions_batch, retrieval_results_batch, initial_results_batch,
            #                 original_judgments_batch, other_judgments_batch, first_turn=False
            #             )
            #             batch_indices = range(k, min(k + self.batch_size, len(questions)))
            #             for index, rethink_new_judgment in zip(batch_indices, rethink_new_judgments):
            #                 hlcn_judgments[i][index][model_name].append(rethink_new_judgment)
            #         self.del_model(model, tokenizer)
            
            # 进行答案集成
            # for modified_result in modified_results_all_rounds[i]:
            #     for result in final_results[i]:
            #         modified_result.append(result)
            for modified_result, result in zip(modified_results_all_rounds[i], final_results[i]):
                modified_result.append(result)
            model, tokenizer, sample_params = self.load_model(self.config['ensembler']['qwen2.5-7b-instruct'])
            modifier = Modifier(model, tokenizer, sample_params)
            ids = []
            for j in range(len(questions)):
                if not all('does not have hallucination' in hlcn_judgments[i][j][model_name][-1] for model_name in self.model_names):
                    ids.append(j)
            if len(ids) == 0:# 都不需要集成，则直接退出
                self.del_model(model, tokenizer)
                continue
            for j in tqdm(range(0, len(ids), self.batch_size), desc=f'Initial Modify'):
                batch_indices = range(j, min(j + self.batch_size, len(ids)))
                questions_batch = [questions[id] for id in batch_indices]
                retrieval_results_batch = [retrieval_results[id] for id in batch_indices]
                initial_results_batch = [final_results[-1][id] for id in batch_indices]
                hlcn_judgments_batch = [[hlcn_judgments[i][m][model_name][-1] for model_name in self.model_names] for m in range(j, min(j + self.batch_size, len(questions)))]
                modifier_results_batch = modifier.inference(
                    questions_batch, retrieval_results_batch, initial_results_batch, hlcn_judgments_batch
                )
                for index, modifier_result in zip(batch_indices, modifier_results_batch):
                    modified_results_all_rounds[i][ids[index]].append(modifier_result)
            
            # for j in tqdm(range(0, len(questions), self.batch_size), desc=f'Initial Modify'):
            #     questions_batch = questions[j:min(j + self.batch_size, len(questions))]
            #     retrieval_results_batch = retrieval_results[j:min(j + self.batch_size, len(questions))]
            #     initial_results_batch = [final_results[-1][k] for k in range(j, min(j + self.batch_size, len(questions)))]
            #     hlcn_judgments_batch = [[hlcn_judgments[i][m][model_name][-1] for model_name in self.model_names] for m in range(j, min(j + self.batch_size, len(questions)))]
            #     modifier_results_batch = modifier.inference(
            #         questions_batch, retrieval_results_batch, initial_results_batch, hlcn_judgments_batch
            #     )
            #     batch_indices = range(j, min(j + self.batch_size, len(questions)))
            #     for index, modifier_result in zip(batch_indices, modifier_results_batch):
            #         modified_results_all_rounds[i][index].append(modifier_result)
            # print(modified_results_all_rounds)
            if self.modify_rounds > 0:
                ids = list(range(len(questions)))
                for j in range(self.modify_rounds):
                    rethink_questions = [questions[id] for id in ids]
                    rethink_documents = [retrieval_results[id] for id in ids]
                    # rethink_init_results = [final_results[-1][id] for id in ids]
                    rethink_hlcn_judgments = [[hlcn_judgments[i][id][model_name][-1] for model_name in self.model_names] for id in ids]
                    rethink_modifier_results = [modified_results_all_rounds[i][id][-1] for id in ids]
                    this_round_modifier_results = []
                    for k in tqdm(range(0, len(rethink_questions), self.batch_size), desc=f'Round {i+1} for Modify for {j+1}'):
                        rethink_questions_batch = rethink_questions[k:min(k + self.batch_size, len(rethink_questions))]
                        rethink_documents_batch = rethink_documents[k:min(k + self.batch_size, len(rethink_documents))]
                        # rethink_init_results_batch = rethink_init_results[k:min(k + self.batch_size, len(rethink_init_results))]
                        rethink_hlcn_judgments_batch = rethink_hlcn_judgments[k:min(k + self.batch_size, len(rethink_hlcn_judgments))]
                        rethink_modifier_results_batch = rethink_modifier_results[k:min(k + self.batch_size, len(rethink_modifier_results))]
                        rethink_new_modifier_results = modifier.inference(
                            rethink_questions_batch, rethink_documents_batch, 
                            rethink_hlcn_judgments_batch, rethink_modifier_results_batch, first_turn=False
                        )
                        this_round_modifier_results.extend(rethink_new_modifier_results)
                    temp_ids = []
                    for k in range(len(this_round_modifier_results)):
                        if not ('The answer is correct' in this_round_modifier_results[k]):# 仍然需要再次修正
                            temp_ids.append(ids[k])
                            modified_results_all_rounds[i][ids[k]].append(this_round_modifier_results[k])
                    ids = copy.deepcopy(temp_ids)
                    if len(ids) == 0:
                        break
            # print(modified_results_all_rounds[i])
            final_results.append([modifier_result[-1] for modifier_result in modified_results_all_rounds[i]])
            self.del_model(model, tokenizer)
            
            # 进行答案简化
        model, tokenizer, sample_params = self.load_model(self.config['ensembler']['qwen2.5-7b-instruct'])
        simplifier = Simplifier(model, tokenizer, sample_params)
        simplified_results_all_rounds = [[retain_after_last_substring(retain_after_last_substring(modified_result[-1], 'user\n'), 'Answer: ')] for modified_result in modified_results_all_rounds[-1]]
        ids = list(range(len(questions)))
        for j in range(self.simplify_rounds):
            rethink_questions = [questions[id] for id in ids]
            rethink_simplified_answers = [simplified_results_all_rounds[id][-1] for id in ids]
            this_round_simplified_answers = []
            for k in tqdm(range(0, len(rethink_questions), self.batch_size), desc=f'Simplify for {j+1}'):
                rethink_questions_batch = rethink_questions[k:min(k + self.batch_size, len(rethink_questions))]
                rethink_simplified_answers_batch = rethink_simplified_answers[k:min(k + self.batch_size, len(rethink_simplified_answers))]
                rethink_new_simplified_answers = simplifier.inference(rethink_questions_batch, rethink_simplified_answers_batch)
                rethink_new_simplified_answers = [retain_after_last_substring(retain_after_last_substring(rethink_new_simplified_answer, 'user\n'), 'Answer: ') for rethink_new_simplified_answer in rethink_new_simplified_answers]
                this_round_simplified_answers.extend(rethink_new_simplified_answers)
            temp_ids = []
            for k in range(len(this_round_simplified_answers)):
                if not ('No need change' in this_round_simplified_answers[k]) and this_round_simplified_answers[k] != simplified_results_all_rounds[ids[k]][-1]:# 仍然需要再次简化
                    temp_ids.append(ids[k])
                    simplified_results_all_rounds[ids[k]].append(this_round_simplified_answers[k])
            ids = copy.deepcopy(temp_ids)
            if len(ids) == 0:
                break
            
        self.del_model(model, tokenizer)
        
        final_results.append([simplified_result[-1] for simplified_result in simplified_results_all_rounds])
        results = final_results[-1]
        print(results)
        final_results = [[final_result[i] for final_result in final_results] for i in range(len(questions))]
        modified_results_all_rounds = [[modified_result[i] for modified_result in modified_results_all_rounds] for i in range(len(questions))]
        hlcn_judgments = [[hlcn_judgment[i] for hlcn_judgment in hlcn_judgments] for i in range(len(questions))]
        with open(self.result_path, 'w', encoding='utf-8') as fw:
            for question, retrieval_result, init_result, hlcn_judgment, modified_result, final_result, simplified_answer, golden_answer, result in zip(
                questions, retrieval_results, initial_results, hlcn_judgments, modified_results_all_rounds, final_results, simplified_results_all_rounds, golden_answers, results
                ):
                temp_dict = dict()
                temp_dict['question'] = question
                temp_dict['retrieval result'] = retrieval_result
                temp_dict['initial answer'] = init_result
                temp_dict2 = {f"round {i+1}": {'judgment': hlcn_judgment[i], 'modified answer': modified_result[i]} for i in range(self.judgments_rounds)}
                temp_dict['ensemble'] = temp_dict2
                # temp_dict['judgments'] = hlcn_judgment
                # temp_dict['modified answer'] = modified_result
                temp_dict['final result'] = final_result
                temp_dict['simplified answer'] = simplified_answer
                temp_dict['answer'] = result
                temp_dict['golden answer'] = golden_answer
                fw.write(ujson.dumps(temp_dict) + '\n')
                
        with open(self.result_path, 'r', encoding='utf-8') as fr, open(self.result_path_indent4, 'w', encoding='utf-8') as fw:
            for line in fr:
                fw.write(ujson.dumps(ujson.loads(line), indent=4) + '\n')
        end_time = time.time()
        print(f'Program run time for {self.rag_type}: {(end_time - start_time)/60}min')
