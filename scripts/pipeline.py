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
from inference import AbstractiveRecompRefiner, Checker, Generator, Modifier, Rethinker
import ujson
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, ALL_COMPLETED
from itertools import repeat
import torch
import os

class BasicPipeline:
    """Base object of all pipelines. A pipeline includes the overall process of RAG.
    If you want to implement a pipeline, you should inherit this class.
    """

    def __init__(self, config, prompt_template=None):
        self.config = config
        self.batch_size = config['batch_size']
        self.result_path = config['result_path']
        self.rag_type = config['RAG_type']
        self.use_refiner = config['use_refiner']
        self.use_rethink = config['use_rethink']
        self.rounds = config['rethinking_rounds']
        self.hlcn_type = config['hlcn_type']

    def run(self, dataset):
        """The overall inference process of a RAG framework."""
        pass
    
    def split_line(self, *args):
        pass   
    
    def distributed(self):
        pass     


            
class MultiRAG(BasicPipeline):
    def __init__(self, config, config_inference, prompt_template=None, retriever=None, models=None, tokenizers=None):
        super().__init__(config, prompt_template)
        self.retriever = retriever
        self.model_names = [model['model_name'] for model in config_inference['generators']]
        self.RAGers = [Generate(config, model, tokenizer) for config, model, tokenizer in zip(config_inference['generators'], models['generators'], tokenizers['generators'])]
        self.ensembler = Ensemble(config_inference['ensembler'], models['ensembler'], tokenizers['ensembler'])
        if self.use_refiner:
            self.refiner = AbstractiveRecompRefiner(config_inference['refiner'], models['refiner'], tokenizers['refiner'])
        
    def run_naive_RAG(self, questions):
        results = []
        assert len(self.RAGers) == 1
        for question in tqdm(questions, desc="RAG only"):
            retrieval_results = self.retriever.search(question)
            result = self.RAGers[0].inference(question, retrieval_results, rag_type=self.rag_type)
            results.append(result)
        # if not os.path.exists(self.result_path):
        with open(self.result_path, 'w', encoding='utf-8') as fw:
            for question, result in zip(questions, results):
                temp = dict()
                temp['question'] = question
                temp['answer'] = result
                fw.write(ujson.dumps(temp) + '\n')
    
    def split_line(self, question):
        s = question.rstrip('\n')
        index = len(s)
        while index > 0 and s[index - 1] != '\n':
            index -= 1
        if index > 0:
            return s[:index].rstrip('\n'), s[index:].lstrip('\n')
        else:
            return s, ""
        
    def distributed(self, generator, question, retrieval_result):
        return generator.inference(question, retrieval_result)
                    
    def run(self, questions, golden_answers=None):
        
        # questions = questions[:8]
        # golden_answers = golden_answers[:8]
        
        results = []
        ensemble_results = []
        retrieval_results = []
        retrieval_results = self.retriever.batch_search(questions)
        retrieval_results = retrieval_results[0]
        retrieval_results_refine = retrieval_results
        
        # for i in tqdm(range(0, len(questions), self.batch_size), desc='Questions Process'):
        #     question = questions[i:i + self.batch_size]
        #     retrieval_result = retrieval_results[i:i + self.batch_size]
        #     result = [RAGer.inference(question, retrieval_result, batch_size=self.batch_size, rag_type='naive') for RAGer in self.RAGers]
        #     results.extend(result[0])
        #     # print(result[0])
        # print(results)
        
        # retrieval_results_refine = self.refiner.batch_run(questions, retrieval_results, batch_size=self.batch_size)
        # with open('/data00/yifei_chen/multi_llms_for_CoT/datasets/2wikimultihopqa/refiner_results.jsonl', 'w', encoding='utf-8') as fw:
        #     for refine_document in retrieval_results_refine:
        #         fw.write(ujson.dumps(refine_document) + '\n')
        # return
        # with open('/data00/yifei_chen/multi_llms_for_CoT/datasets/2wikimultihopqa/refiner_results.jsonl', 'r', encoding='utf-8') as fr:
        #     retrieval_results_refine = []
        #     for line in fr:
        #         retrieval_results_refine.append(ujson.loads(line))
        # return
        
        # model_names = [
        #     'mistral-7B-instruct-v0.3',
        #     'qwen2-7B-instruct',
        #     'glm-4-9b-chat',
        # ]
        # temp_ensemble_results = [[] for _ in range(len(questions))]
        # for name in model_names:
        #     with open(f'/data00/yifei_chen/multi_llms_for_CoT/datasets/2wikimultihopqa/generator_{name}.jsonl', 'r', encoding='utf-8') as fr:
        #         for num, line in enumerate(fr, start=0):
        #             data = ujson.loads(line)
        #             temp_ensemble_results[num].append((next(iter(data.values()))[0], next(iter(data.values()))[1]))
        
        # 正式多模型推理过程
        for i in tqdm(range(0, len(questions), self.batch_size), desc='Questions Process'):
            question = questions[i:i + self.batch_size]
            retrieval_result = retrieval_results_refine[i:i + self.batch_size]
            # ensemble_result = temp_ensemble_results[i:i + self.batch_size]
            ensemble_result = [RAGer.inference(question, retrieval_result, batch_size=self.batch_size, use_refiner=self.use_refiner) for RAGer in self.RAGers]
            ensemble_result = list(map(list, zip(*ensemble_result)))
            # ensemble_result = [list(map(self.split_line, ensemble_per_result)) for ensemble_per_result in ensemble_result]
            # ensemble_result = list(map(lambda x: list(map(self.split_line, x)), ensemble_result))
            ensemble_result = [list(map(self.split_line, ensemble_per_result)) for ensemble_per_result in ensemble_result]
            ensemble_results.extend([dict(zip(self.model_names, ensemble_per_result)) for ensemble_per_result in ensemble_result])
            result = self.ensembler.inference(question, retrieval_result, ensemble_result)
            results.extend(result)
        # print(results)

        
        # results = self.refiner.batch_run(questions, retrieval_results, batch_size=self.batch_size)    
        # with open('/data00/yifei_chen/multi_llms_for_CoT/datasets/2wikimultihopqa/refiner_results.jsonl', 'w', encoding='utf-8') as fw:
        #     for result in results:
        #         fw.write(ujson.dumps(result) + '\n')
        # return
        # with open(f'/data00/yifei_chen/multi_llms_for_CoT/datasets/2wikimultihopqa/generator_{self.model_names[0]}.jsonl', 'w', encoding='utf-8') as fw:
        #     for ensemble_result in ensemble_results:
        #         fw.write(ujson.dumps(ensemble_result) + '\n')
        # return
        # with open('/data00/yifei_chen/multi_llms_for_CoT/datasets/2wikimultihopqa/LLM_ensemble_results_refiner_decompose.jsonl', 'w', encoding='utf-8') as fw:
        #     for result in results:
        #         fw.write(ujson.dumps(result) + '\n')
        # return
            
        with open(self.result_path, 'w', encoding='utf-8') as fw:
            for question, retrieval_result, result, golden_answer, ensemble_result in zip(questions, retrieval_results, results, golden_answers, ensemble_results):
                temp_dict = dict()
                temp_dict['question'] = question
                temp_dict['retrieval result'] = retrieval_result
                temp_dict['ensemble result'] = ensemble_result
                temp_dict['answer'] = result
                temp_dict['golden answer'] = golden_answer
                fw.write(ujson.dumps(temp_dict) + '\n')
        save_path = '/data00/yifei_chen/multi_llms_for_CoT/datasets/nq/results/ensembleRAG_indent4.jsonl'
        with open(self.result_path, 'r', encoding='utf-8') as fr, open(save_path, 'w', encoding='utf-8') as fw:
            for line in fr:
                fw.write(ujson.dumps(ujson.loads(line), indent=4) + '\n')

        # with open(self.result_path, 'w', encoding='utf-8') as fw:
        #     for question, retrieval_result, ensemble_result, result, golden_answer, refine in zip(questions, retrieval_results, ensemble_results, results, golden_answers, retrieval_results_refine):
        #         temp_dict = dict()
        #         temp_dict['question'] = question
        #         temp_dict['retrieval result'] = retrieval_result
        #         temp_dict['refine result'] = refine
        #         temp_dict['ensemble result'] = ensemble_result
        #         temp_dict['answer'] = result
        #         temp_dict['golden answer'] = golden_answer
        #         fw.write(ujson.dumps(temp_dict) + '\n')
        # save_path = '/data00/yifei_chen/multi_llms_for_CoT/datasets/nq/results/naiveRAG_results_indent4.jsonl'
        # with open(self.result_path, 'r', encoding='utf-8') as fr, open(save_path, 'w', encoding='utf-8') as fw:
        #     for line in fr:
        #         fw.write(ujson.dumps(ujson.loads(line), indent=4) + '\n')
        
            
            
class CheckRAG(BasicPipeline):
    def __init__(self, config, retriever=None, models=None, tokenizers=None):
        super().__init__(config)         
        self.retriever = retriever
        self.checker = Checker(config['generator'], models['checker'], tokenizers['checker'])
        self.generator = Generator(config['generator'], models['generator'], tokenizers['generator'])
        self.modifier = Modifier(config['generator'], models['modifier'], tokenizers['modifier'])
        if self.use_rethink:
            self.rethinker = Rethinker(config['generator'], models['rethinker'], tokenizers['rethinker'])
        if self.use_refiner:
            self.refiner = AbstractiveRecompRefiner(config['refiner'], models['refiner'], tokenizers['refiner'])
            
    def run(self, questions, golden_answers=None):
        initial_results = []
        results = []
        modify_results = []
        retrieval_results = []
        retrieval_results = self.retriever.batch_search(questions)
        retrieval_results = retrieval_results[0]
        
        for i in tqdm(range(len(questions)), desc='Questions Process'):
            init_result = self.generator.inference(questions[i], retrieval_results[i], rag_type=self.rag_type, use_refiner=False)
            init_result = init_result.lstrip('\n')
            if self.rag_type == 'naive':
                results.append(init_result)
                continue
            initial_results.append(init_result)
            
            # 下面开始修正
            temp_modify = dict()
            for hlcn in self.hlcn_type:
                temp_hlcn = []
                check_result = self.checker.inference(questions[i], retrieval_results[i], init_result, rag_type=self.rag_type, use_refiner=False, hlcn_type=hlcn)
                temp_hlcn.append(check_result)
                if self.use_rethink:
                    assert self.rounds > 0
                    for j in range(self.rounds):
                        check_result_ = self.rethinker.inference(questions[i], retrieval_results[i], init_result, check_result, use_refiner=False, hlcn_type=hlcn)
                        if not ('No revise is required' in check_result or 'no revise is required' in check_result):# 有修改
                            check_result = check_result_
                            break
                        temp_hlcn.append(check_result)
                temp_modify[hlcn] = temp_hlcn
            result = self.modifier.inference(questions[i], retrieval_results[i], init_result, list(temp_modify.values()), use_refiner=False)
            result = result.lstrip('\n')
            # result = result.rstrip('\n')
            modify_results.append(temp_modify)
            results.append(result)
        print(results)
        
        if self.rag_type != 'naive':
            with open(self.result_path, 'w', encoding='utf-8') as fw:
                for question, retrieval_result, result, golden_answer, check_result, init_result in zip(questions, retrieval_results, results, golden_answers, modify_results, initial_results):
                    temp_dict = dict()
                    temp_dict['question'] = question
                    temp_dict['retrieval result'] = retrieval_result
                    temp_dict['initial answer'] = init_result
                    temp_dict['judgments'] = check_result
                    temp_dict['answer'] = result
                    temp_dict['golden answer'] = golden_answer
                    fw.write(ujson.dumps(temp_dict) + '\n')
                    
            save_path = '/data00/yifei_chen/multi_llms_for_CoT/datasets/nq/results/checkRAG_indent4.jsonl'
            with open(self.result_path, 'r', encoding='utf-8') as fr, open(save_path, 'w', encoding='utf-8') as fw:
                for line in fr:
                    fw.write(ujson.dumps(ujson.loads(line), indent=4) + '\n')
        else:
            with open(self.result_path, 'w', encoding='utf-8') as fw:
                for question, retrieval_result, result, golden_answer in zip(questions, retrieval_results, results, golden_answers):
                    temp_dict = dict()
                    temp_dict['question'] = question
                    temp_dict['retrieval result'] = retrieval_result
                    temp_dict['answer'] = result
                    temp_dict['golden answer'] = golden_answer
                    fw.write(ujson.dumps(temp_dict) + '\n')
                    
            save_path = '/data00/yifei_chen/multi_llms_for_CoT/datasets/nq/results/naiveRAG_indent4.jsonl'
            with open(self.result_path, 'r', encoding='utf-8') as fr, open(save_path, 'w', encoding='utf-8') as fw:
                for line in fr:
                    fw.write(ujson.dumps(ujson.loads(line), indent=4) + '\n')
            
            
            
