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
from modelscope import snapshot_download

class Decompose:
    def __init__(self, config):
        self.function = config['function']
        self.model_name = config['model_name']
        self.model_path = config["model_path"]
        self.max_input_len = config["max_input_len"]
        self.batch_size = config["batch_size"]
        self.device = config["device"]
        self.generator_params = config['generator_params']
        self.res_save_path = "datasets/hotpotqa/" + config['task_type'] + '.jsonl'
        self.system_prompt = {
            'question decompose': (
                "Here is a question for you, and please decompose this question into a small number of subproblems that cover all of the contents and goals of the original question, but do not break down the subproblems too finely. Only give me decomposed subquestions, and do not output any other words."
            ),
            "question rewrite": (
                "Here is a question-answer pair and another question for you, and please rewrite the second provided question according to the contents of the given question-answer pair, so that the rewritten question contains the contents of the above given question-answer pair. Only give me rewritten question, and do not output any other words."
            )
        }
        self.user_prompt = {
            'question decompose': (
                "Question: {question}\n"
            ),
            'question rewrite': (
                "Question and Answer pair: \nQuestion: {question}\nAnswer: {answer}\n\nQuestion: {another_question}\n"
            )
        }
        self.datas = []
        self.data_path = config['data_path']
        self.read_data()
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        if "qwen" not in self.model_name:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        if "llama" in self.model_name.lower():
            extra_eos_tokens = [
                self.tokenizer.eos_token_id,
                self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
            ]
            if "eos_token_id" in self.generator_params:
                self.generator_params["eos_token_id"].extend(extra_eos_tokens)
            else:
                self.generator_params["eos_token_id"] = extra_eos_tokens
        
    def read_data(self):
        with open(self.data_path, 'r', encoding='utf-8') as fr:
            for line in fr:
                data = ujson.loads(line)
                self.datas.append(data)
    def Decompose(self):
        if self.function in self.system_prompt:
            if self.function in self.system_prompt:
                batch_prompts = []
                try:
                    if self.function == 'question decompose':
                        batch_prompts = [self.user_prompt[self.function].format(question=data['question']) for data in self.datas]
                except KeyError as e:
                    print(f"Warning: Unmatched placeholder {e} found.")
            else:
                raise ValueError(f"{self.function} is not in existing methods!")
            if "llama" in self.model_name.lower():
                terminators = [
                    self.tokenizer.eos_token_id,
                    self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
                ]
                if "eos_token_id" in self.generator_params:
                    self.generator_params["eos_token_id"].extend(terminators)
                else:
                    self.generator_params["eos_token_id"] = terminators
            result = []
            for i in tqdm(range(0, len(self.datas), self.batch_size), desc=self.function):
                prompts = batch_prompts[i:i + self.batch_size]
                messages = [
                    [
                    {"role": "system", "content": self.system_prompt[self.function]},
                    {"role": "user", "content": prompt},
                ] for prompt in prompts
                ]
                input_ids = [
                    self.tokenizer.apply_chat_template(
                    message, 
                    tokenize=False,
                    add_generation_prompt=True
                    ) for message in messages
                ]
                input_ids = self.tokenizer(input_ids, return_tensors="pt", padding=True, truncation=True).to(self.device)
                outputs = self.model.generate(
                    **input_ids,
                    return_dict_in_generate=True,
                    **self.generator_params,
                )
            # for i in tqdm(range(0, len(self.datas)), desc=self.function):
            #     prompt = batch_prompts[i]
            #     message = [
            #         {"role": "system", "content": self.system_prompt[self.function]},
            #         {"role": "user", "content": prompt},
            #     ]
            #     input_ids = self.tokenizer.apply_chat_template(
            #         message, 
            #         return_tensors="pt",
            #         padding=True,
            #         truncation=True,
            #         max_length=self.max_input_len
            #         ).to(self.device)
            #     outputs = self.model.generate(
            #         input_ids,
            #         return_dict_in_generate=True,
            #         **self.generator_params,
            #     )
                # generated_ids = outputs[0][input_ids.shape[-1]:]
                # reshape_outputs = outputs.view(-1, len(self.datas), outputs.shape[-1])# outputsoutputs[0]
                # generated_ids = outputs.sequences
                # outputs = outputs.view(-1, len(self.datas), outputs.shape[-1])
                responses = self.tokenizer.batch_decode(outputs[0], skip_special_tokens=True)
                
                indexes = [response.find('assistant') + len('assistant') for response in responses]
                results = [response[index:].lstrip('\n') for response, index in zip(responses, indexes)]
                result.extend(results)
            with open(self.res_save_path, 'w', encoding='utf-8') as fw:
                for data, subresult in zip(self.datas, result):
                    question = data['question']
                    subquestion = subresult.split('\n')
                    subquestion = [s for s in subquestion if s]
                    temp_dict = dict()
                    temp_dict['question'] = question
                    temp_dict['sub-question'] = subquestion
                    fw.write(ujson.dumps(temp_dict) + '\n')
            
        else:
            raise ValueError(f"unknown function: {self.function}")
    
config = {
    'function': 'question decompose',
    'model_name': 'llama3-8B-instruct',
    'model_path': "/data00/LLaMA-3-8b-Instruct/",
    'max_input_len': 512,
    'batch_size': 5,
    'device': 'cuda',
    'data_path': 'datasets/hotpotqa/sample_test.jsonl',
    'task_type': 'question decompose',
    'generator_params':
        {
            'do_sample': True,
            'max_new_tokens': 512,
            'temperature': 1.0,
            'top_p': 0.2,
        }
}
Decompose_example = Decompose(config=config)
Decompose_example.Decompose()