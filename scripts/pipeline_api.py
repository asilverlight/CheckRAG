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
    
from inference_api2 import Generator, Rethinker, Simplifier, Classifier, Modifier    
class CheckRAG(BasicPipeline): 
    def __init__(self, config, retriever=None):
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
            
    async def async_task(self, *args, processor, **kwargs):
        result = await asyncio.to_thread(processor, *args, **kwargs)
        return result
    
    async def async_format(self, s, substring):
        result = await asyncio.to_thread(retain_after_last_substring, s, substring)
        return result
    
    async def run(self, questions, golden_answers, retrieval_results):
        start_time = time.time()
        model_num = len(self.model_load)
        initial_results = []
        results = []
        # retrieval_results = self.retriever.batch_searcch(questions)
        retrieval_results = retrieval_results[0]
        modified_results_all_rounds = [[[] for _ in range(len(questions))] for _ in range(self.judgments_rounds)]# 保存所有轮次的修正结果
        simplified_results_all_rounds = [[] for _ in range(len(questions))] # 保存所有轮次的简化结果
        
        key_agent_model = self.model_load['qwen2.5-7b-instruct']
        key_agent_model_name = 'qwen2.5-7b-instruct'
        # 第一步：生成init results
        # client_generator = OpenAI(api_key=key_agent_model['api_key'], base_url=key_agent_model['base_url'])
        generator = Generator()
        classifier = Classifier()
        # rethinker = Rethinker()
        modifier = Modifier()
        simplifier = Simplifier()
        # kwargs = {
        #     'rag_type': self.rag_type,
        #     'api_key': key_agent_model['api_key'],
        #     'base_url': key_agent_model['base_url'],
        #     'model_path': key_agent_model['model_path'],
        #     'first_run': False,
        #     'multiple_choice': False,
        # }
        if self.rag_type == 'naive':
            for i in tqdm(range(0, len(questions), self.batch_size * model_num), desc='Initial Generation'):
                questions_batch = questions[i:min(i + self.batch_size * model_num, len(questions))]
                retrieval_results_batch = retrieval_results[i:min(i + self.batch_size * model_num, len(questions))]
                # 创建task
                tasks = []
                for j in range(len(questions_batch)):
                    tasks.append(
                        self.async_task(
                            questions_batch[j],
                            retrieval_results_batch[j],
                            processor=generator.inference,
                            rag_type=self.rag_type,
                            api_key=key_agent_model['api_key'],#self.generator_model['api_key'],# 
                            base_url=key_agent_model['base_url'],#self.generator_model['base_url'],# 
                            model_path=key_agent_model['model_path'],#self.generator_model['model_path'],# 
                        )
                    )
                init_results = await asyncio.gather(*tasks)
                results.extend(init_results)
                
        
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
        else:
            for i in tqdm(range(0, len(questions), self.batch_size * model_num), desc='Initial Generation'):
                questions_batch = questions[i:min(i + self.batch_size * model_num, len(questions))]
                retrieval_results_batch = retrieval_results[i:min(i + self.batch_size * model_num, len(questions))]
                # 创建task
                tasks = []
                for j in range(len(questions_batch)):
                    tasks.append(
                        self.async_task(
                            questions_batch[j],
                            retrieval_results_batch[j],
                            processor=generator.inference,
                            rag_type='naive',
                            api_key=key_agent_model['api_key'],#self.generator_model['api_key'],# 
                            base_url=key_agent_model['base_url'],#self.generator_model['base_url'],# 
                            model_path=key_agent_model['model_path'],#self.generator_model['model_path'],# 
                        )
                    )
                init_results = await asyncio.gather(*tasks)
                initial_results.extend(init_results)
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
        
        # hlcn_judgments:
        # [
        #     [
        #         {llama: [], qwen: [], mistral: []},{llama: [], qwen: [], mistral: []},{llama: [], qwen: [], mistral: []},{},......
        #     ],
        #     [
        #         {llama: [], qwen: [], mistral: []},{},{},{},......
        #     ],
        #     [
        #         {llama: [], qwen: [], mistral: []},{},{},{},......
        #     ]
        # ]
        
        # 接下来尝试根据正确答案生成wrong answer
        # for i in tqdm(range(0, len(questions), self.batch_size * model_num), desc='Wrong Answer Generation'):
        #     questions_batch = questions[i:min(i + self.batch_size * model_num, len(questions))]
        #     retrieval_results_batch = retrieval_results[i:min(i + self.batch_size * model_num, len(questions))]
        #     initial_results_batch = initial_results[i:min(i + self.batch_size * model_num, len(initial_results))]
        #     tasks = []
        #     for j in range(len(questions_batch)):
        #         tasks.append(
        #             self.async_task(
        #                 questions_batch[j],
        #                 retrieval_results_batch[j],
        #                 initial_results_batch[j],
        #                 processor=generator.inference,
        #                 rag_type=self.rag_type,
        #                 api_key=key_agent_model['api_key'],
        #                 base_url=key_agent_model['base_url'],
        #                 model_path=key_agent_model['model_path'],
        #                 wrong_answer=True,
        #             )
        #         )
        #     wrong_answer_batch = await asyncio.gather(*tasks)
        #     batch_indices = range(i, min(i + self.batch_size * model_num, len(questions)))
        #     for index, wrong_answer in zip(batch_indices, wrong_answer_batch):
        #         final_results[-1][index] = wrong_answer
        
        for i in range(self.judgments_rounds):
            for j in tqdm(range(0, len(questions), self.batch_size), desc=f'Judgment for {i+1}'):
                questions_batch = questions[j:min(j + self.batch_size, len(questions))]
                retrieval_results_batch = retrieval_results[j:min(j + self.batch_size, len(questions))]
                initial_results_batch = [final_results[-1][k] for k in range(j, min(j + self.batch_size, len(questions)))]
                tasks = []# 创建task
                for k in range(len(questions_batch)):
                    for model_load in self.model_load.values():
                        tasks.append(
                            self.async_task(
                                questions_batch[k],
                                retrieval_results_batch[k],
                                initial_results_batch[k],
                                processor=classifier.inference,
                                rag_type=self.rag_type,
                                api_key=model_load['api_key'],
                                base_url=model_load['base_url'],
                                model_path=model_load['model_path'],
                            )
                        )
                # 该任务集合长度为batch_size*3，每3个为一组，对应一个data的三种模型处理结果
                hlcn_judgments_batch = await asyncio.gather(*tasks)
                hlcn_judgments_batch = [hlcn_judgments_batch[k:k + model_num] for k in range(0, len(hlcn_judgments_batch), model_num)]
                batch_indices = range(j, min(j + self.batch_size, len(questions)))
                for index, hlcn_judgment_batch in zip(batch_indices, hlcn_judgments_batch):
                    for model_name, hlcn_judgment in zip(self.model_names, hlcn_judgment_batch):
                        hlcn_judgments[i][index][model_name].append(hlcn_judgment)
                        
            # 接下来，先让模型思考一下自己的结果
            # for j in tqdm(range(0, len(questions), self.batch_size), desc=f'First Rethink for {i+1}'):
            #     questions_batch = questions[j:min(j + self.batch_size, len(questions))]
            #     retrieval_results_batch = retrieval_results[j:min(j + self.batch_size, len(questions))]
            #     initial_results_batch = [final_results[-1][k] for k in range(j, min(j + self.batch_size, len(questions)))]
            #     hlcn_judgments_batch = [[hlcn_judgments[i][k][model_name][-1] for model_name in self.model_names] for k in range(j, min(j + self.batch_size, len(questions)))]
            #     tasks = []
            #     for k in range(len(questions_batch)):
            #         for m, model_name in enumerate(self.model_names):
            #             tasks.append(
            #                 self.async_task(
            #                     questions_batch[k],
            #                     retrieval_results_batch[k],
            #                     initial_results_batch[k],
            #                     hlcn_judgments_batch[k][m],
            #                     processor=rethinker.inference,
            #                     rag_type=self.rag_type,
            #                     api_key=self.model_load[model_name]['api_key'],
            #                     base_url=self.model_load[model_name]['base_url'],
            #                     model_path=self.model_load[model_name]['model_path'],
            #                 )
            #             )
            #     rethink_judgments = await asyncio.gather(*tasks)
            #     # rethink长度为batch_size*3，每3个为一组，对应一个data的三种模型处理结果
            #     rethink_judgments = [rethink_judgments[k:k + model_num] for k in range(0, len(rethink_judgments), model_num)]
            #     batch_indices = range(j, min(j + self.batch_size, len(questions)))
            #     for index, rethink_judgment in zip(batch_indices, rethink_judgments):
            #         for model_name, rethink_judgment_ in zip(self.model_names, rethink_judgment):
            #             hlcn_judgments[i][index][model_name].append(rethink_judgment_)
                        
            # 之后相互交换结果
            # 首先判断，是否所有模型都认为没有幻觉
            # print([len(hlcn_judgments[i][m]['llama3-8B-instruct']) for m in range(len(questions))])
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
                this_round_judgments = []
                for k in tqdm(range(0, len(questions_judgments), self.batch_size), desc=f'Rethink for {j+1} in {i+1} round'):
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
                    tasks = []
                    for m in range(len(questions_batch)):
                        for n, model_name in enumerate(self.model_names):
                            tasks.append(
                                self.async_task(
                                    questions_batch[m],
                                    retrieval_results_batch[m],
                                    initial_results_batch[m],
                                    original_judgments_batch[m][n],
                                    other_judgments_batch[m][n],
                                    processor=classifier.inference,
                                    rag_type=self.rag_type,
                                    api_key=self.model_load[model_name]['api_key'],
                                    base_url=self.model_load[model_name]['base_url'],
                                    model_path=self.model_load[model_name]['model_path'],
                                    first_turn=False,
                                )
                            )
                    rethink_new_judgments = await asyncio.gather(*tasks)
                    # rethink_new_judgments的长度为batch*3
                    rethink_new_judgments = [rethink_new_judgments[n:n + model_num] for n in range(0, len(rethink_new_judgments), model_num)]
                    # rethink_new_judgments_array = np.array(rethink_new_judgments)
                    # print(rethink_new_judgments_array.shape)
                    this_round_judgments.extend(rethink_new_judgments)
                for m in range(len(ids)):
                    for model_name, rethink_new_judgment in zip(self.model_names, this_round_judgments[m]):
                        # print([rethink_new_judgment])
                        # print('\n')
                        hlcn_judgments[i][ids[m]][model_name].append(rethink_new_judgment)
                # print([len(hlcn_judgments[i][m]['llama3-8B-instruct']) for m in range(len(questions))])
                            
            # 进行答案集成
            for modified_result, result in zip(modified_results_all_rounds[i], final_results[i]):
                modified_result.append(result)
            ids = []
            for j in range(len(questions)):
                temp_judgments = [hlcn_judgments[i][j][model_name][-1] for model_name in self.model_names]
                if not all('does not have hallucination' in judgment for judgment in temp_judgments):
                    ids.append(j)
            for j in tqdm(range(0, len(ids), self.batch_size * model_num), desc=f'Initial Ensemble for {i+1} round'):
                questions_batch = [questions[id] for id in ids[j:min(j + self.batch_size * model_num, len(ids))]]
                retrieval_results_batch = [retrieval_results[id] for id in ids[j:min(j + self.batch_size * model_num, len(ids))]]
                initial_results_batch = [final_results[-1][id] for id in ids[j:min(j + self.batch_size * model_num, len(ids))]]
                hlcn_judgments_batch = [[hlcn_judgments[i][id][model_name][-1] for model_name in self.model_names] for id in ids[j:min(j + self.batch_size * model_num, len(ids))]]
                # initial_results_batch = [final_results[-1][k] for k in range(j, min(j + self.batch_size * model_num, len(questions)))]
                # hlcn_judgments_batch = [[hlcn_judgments[i][k][model_name][-1] for model_name in self.model_names] for k in range(j, min(j + self.batch_size * model_num, len(questions)))]
                tasks = []
                for k in range(len(questions_batch)):
                    tasks.append(
                        self.async_task(
                            questions_batch[k],
                            retrieval_results_batch[k],
                            initial_results_batch[k],
                            hlcn_judgments_batch[k],
                            processor=modifier.inference,
                            rag_type=self.rag_type,
                            api_key=key_agent_model['api_key'],#self.generator_model['api_key'],# 
                            base_url=key_agent_model['base_url'],#self.generator_model['base_url'],# 
                            model_path=key_agent_model['model_path'],#self.generator_model['model_path'],# 
                        )
                    )
                modifier_results_batch = await asyncio.gather(*tasks)
                # modifier_results_batch = [modifier_results_batch[l:l + model_num] for l in range(0, len(modifier_results_batch), model_num)]
                batch_indices = range(j, min(j + self.batch_size * model_num, len(ids)))
                for index, modifier_result_batch in zip(batch_indices, modifier_results_batch):
                    modified_results_all_rounds[i][ids[index]].append(modifier_result_batch)
                    
            if self.modify_rounds > 0:
                ids = list(range(len(questions)))
                for j in range(self.modify_rounds):
                    rethink_questions = [questions[id] for id in ids]
                    rethink_documents = [retrieval_results[id] for id in ids]
                    rethink_init_results = [final_results[-1][id] for id in ids]
                    rethink_hlcn_judgments = [[hlcn_judgments[i][id][model_name][-1] for model_name in self.model_names] for id in ids]
                    rethink_modified_results = [modified_results_all_rounds[i][id][-1] for id in ids]
                    this_round_modified_results = []
                    for k in tqdm(range(0, len(rethink_questions), self.batch_size * model_num), desc=f'Modify for {j+1} in {i+1} round'):
                        rethink_questions_batch = rethink_questions[k:min(k + self.batch_size * model_num, len(rethink_questions))]
                        rethink_documents_batch = rethink_documents[k:min(k + self.batch_size * model_num, len(rethink_documents))]
                        rethink_init_results_batch = rethink_init_results[k:min(k + self.batch_size * model_num, len(rethink_init_results))]
                        rethink_hlcn_judgments_batch = rethink_hlcn_judgments[k:min(k + self.batch_size * model_num, len(rethink_hlcn_judgments))]
                        rethink_modified_results_batch = rethink_modified_results[k:min(k + self.batch_size * model_num, len(rethink_modified_results))]
                        tasks = []
                        for m in range(len(rethink_questions_batch)):
                            tasks.append(
                                self.async_task(
                                    rethink_questions_batch[m],
                                    rethink_documents_batch[m],
                                    rethink_init_results_batch[m],
                                    rethink_hlcn_judgments_batch[m],
                                    rethink_modified_results_batch[m],
                                    processor=modifier.inference,
                                    rag_type=self.rag_type,
                                    api_key=key_agent_model['api_key'],#self.generator_model['api_key'],# 
                                    base_url=key_agent_model['base_url'],#self.generator_model['base_url'],# 
                                    model_path=key_agent_model['model_path'],#self.generator_model['model_path'],# 
                                    first_turn=False,
                                )
                            )
                        rethink_new_modified_results = await asyncio.gather(*tasks)
                        this_round_modified_results.extend(rethink_new_modified_results)
                    temp_ids = []
                    for k in range(len(ids)):
                        if not ('The answer is correct' in this_round_modified_results[k]):# 仍然需要再次修正
                            temp_ids.append(ids[k])
                            modified_results_all_rounds[i][ids[k]].append(this_round_modified_results[k])
                    ids = copy.deepcopy(temp_ids)
                    if len(ids) == 0:
                        break
            # 进行字符串格式化
            for j in range(0, len(questions), self.batch_size * model_num):
                modified_results_batch = [modified_results_all_rounds[i][k][-1] 
                                          for k in range(j, min(j + self.batch_size * model_num, len(questions)))]
                tasks = []
                for k in range(len(modified_results_batch)):
                    tasks.append(
                        self.async_format(
                            modified_results_batch[k],
                            'user\n'
                        )
                    )
                formatted_results_batch = await asyncio.gather(*tasks)
                tasks = []
                for k in range(len(formatted_results_batch)):
                    tasks.append(
                        self.async_format(
                            formatted_results_batch[k],
                            'Answer: '
                        )
                    )
                formatted_results_batch = await asyncio.gather(*tasks)
                indices = range(j, min(j + self.batch_size, len(questions)))
                for index, formatted_result in zip(indices, formatted_results_batch):
                    modified_results_all_rounds[i][index][-1] = formatted_result
            final_results.append([modifier_result[-1] for modifier_result in modified_results_all_rounds[i]])
        # 开始答案简化
        simplified_results_all_rounds = [[final_result] for final_result in final_results[-1]]
        ids = list(range(len(questions)))
        for i in range(self.simplify_rounds):
            rethink_questions = [questions[id] for id in ids]
            rethink_simplifer_results = [simplified_result_all_rounds[-1]
                                         for simplified_result_all_rounds in simplified_results_all_rounds]
            this_round_simplified_answers = []
            for j in tqdm(range(0, len(rethink_questions), self.batch_size * model_num), desc=f'Simplify for {i+1} round'):
                rethink_questions_batch = rethink_questions[j:min(j + self.batch_size * model_num, len(rethink_questions))]
                rethink_simplifer_results_batch = rethink_simplifer_results[j:min(j + self.batch_size * model_num, len(rethink_simplifer_results))]
                tasks = []
                for k in range(len(rethink_questions_batch)):
                    tasks.append(
                        self.async_task(
                            rethink_questions_batch[k],
                            rethink_simplifer_results_batch[k],
                            processor=simplifier.inference,
                            rag_type=self.rag_type,
                            api_key=key_agent_model['api_key'],
                            base_url=key_agent_model['base_url'],
                            model_path=key_agent_model['model_path'],
                        )
                    )
                rethink_new_simplified_results = await asyncio.gather(*tasks)
                this_round_simplified_answers.extend(rethink_new_simplified_results)
            temp_ids = []
            for j in range(len(ids)):
                if not ('No need change' in this_round_simplified_answers[j]) and this_round_simplified_answers[j] != simplified_results_all_rounds[ids[j]][-1]:# 仍然需要再次简化
                    temp_ids.append(ids[j])
                    simplified_results_all_rounds[ids[j]].append(this_round_simplified_answers[j])
            ids = copy.deepcopy(temp_ids)
            if len(ids) == 0:
                break
        results = [simplified_result_all_rounds[-1] 
                   for simplified_result_all_rounds in simplified_results_all_rounds]
        results = [
            re.sub(r'^\s*\n', '', re.sub(r'\n\s*$', '', s)) for s in results
        ]
        results = [
            "".join(filter(bool, text.splitlines())) for text in results
        ]
        print(results)
        final_results = [[final_result[i] for final_result in final_results] for i in range(len(questions))]
        modified_results_all_rounds = [[modified_result[i] for modified_result in modified_results_all_rounds] for i in range(len(questions))]
        
        # 测试
        # for i in range(self.judgments_rounds):
        #     for j in range(len(questions)):
        #         temp = [len(hlcn_judgments[i][j][model_name]) for model_name in self.model_names]
        #         print(temp)
        #     print('\n')
                
        
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