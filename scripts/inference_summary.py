import os
import torch
import ujson
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import BertTokenizer, BertModel, AdamW, BertForSequenceClassification, AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
import random
import numpy as np
import torch
from modelscope import snapshot_download
import re
from typing import List

def get_prompt(message: str, chat_history: list[tuple[str, str]],
               system_prompt: str) -> str:
        texts = [f'<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n']#
        # The first user input is _not_ stripped
        do_strip = False
        for user_input, response in chat_history:
            user_input = user_input.strip() if do_strip else user_input
            do_strip = True
            texts.append(f'{user_input} [/INST] {response.strip()} </s><s>[INST] ')
        message = message.strip() if do_strip else message
        # print(message)
        texts.append(f'{message}')# [/INST]
        return ''.join(texts)
        
        # text_dicts = {
        #     'system': system_prompt,
        #     'user': message,
        #     'assistant': "",
        # }
        # input_template = '''
        # <|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
        # '''.format(system=text_dicts['system'], user=text_dicts['user'])
        # return input_template

data = {
    'system': "Please answer the question according to the contents of the document. Only return the correct answer, and do not return anything else.",
  'instruction': "Document:\n{DOCS} \n\nQuestion:\n{QUERY}",
}
docs = "\"I Know What You Did Last Summer\"\nI Know What You Did Last Summer I Know What You Did Last Summer is a 1997 American slasher film directed by Jim Gillespie, written by Kevin Williamson, and starring Jennifer Love Hewitt, Sarah Michelle Gellar, Ryan Phillippe, and Freddie Prinze Jr., with Anne Heche, Bridgette Wilson, and Johnny Galecki appearing in supporting roles. Loosely based on the 1973 novel of the same name by Lois Duncan, the film centers on four young friends who are stalked by a hook-wielding killer one year after covering up a car accident in which they were involved. The film also draws inspiration from"
query = "who was the killer in the movie i know what you did last summer"

# print(text)
# 以下是glm4，能跑通，能输出正确格式
model = AutoModelForCausalLM.from_pretrained(
                    '/data00/LLaMA-3-8b-Instruct/',#'/data00/yifei_chen/multi_llms_for_CoT/models/ZhipuAI/glm-4-9b-chat',
                    torch_dtype=torch.bfloat16,
                    device_map="cuda:2",
                    trust_remote_code=True,
                )#.to(torch.device('cuda:2'))
tokenizer = AutoTokenizer.from_pretrained(
                    '/data00/LLaMA-3-8b-Instruct/',#'/data00/yifei_chen/multi_llms_for_CoT/models/ZhipuAI/glm-4-9b-chat',
                    trust_remote_code=True
                )


# # 以下是glm
# # text = data['system'] + '\n\n' + data['instruction'].format(DOCS=docs, QUERY=query)
# input = [
#     {'role': 'system', 'content': data['system']},
#     {'role': 'user', 'content': data['instruction'].format(DOCS=docs, QUERY=query)}
# ]
# text = tokenizer.apply_chat_template(
#     input,
#     tokenize=False,
#     add_generation_prompt=True
# )
# inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False, return_token_type_ids=False)
# inputs = inputs.to(model.device)
# outputs = model.generate(
#     **inputs, 
#     do_sample=True, 
#     temperature=0.8, 
#     top_p=0.9, 
#     max_length=512 + inputs['input_ids'].size(-1), 
#     pad_token_id=tokenizer.eos_token_id,
#     #eos_token_id=extra_eos_tokens,#, 128001[128009]
#     )
# response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
# print(response)


# 以下是llama3，能跑通，能输出正确格式
# # text = get_prompt(data['instruction'].format(DOCS=docs, QUERY=query), [], data['system'])
# input = [
#     {'role': 'system', 'content': data['system']},
#     {'role': 'user', 'content': data['instruction'].format(DOCS=docs, QUERY=query)}
# ]
# inputs = tokenizer.apply_chat_template(
#     input,
#     return_tensors='pt',
#     add_generation_prompt=True
# )
# extra_eos_tokens = [
#     tokenizer.eos_token_id,
#     tokenizer.convert_tokens_to_ids("<|eot_id|>"),#<[/INST]>
# ]
# inputs = inputs.to(model.device)
# outputs = model.generate(
#     inputs, 
#     do_sample=True, 
#     temperature=0.8, 
#     top_p=0.9, 
#     max_length=512, 
#     pad_token_id=tokenizer.eos_token_id,
#     eos_token_id=extra_eos_tokens,#, 128001[128009]
#     )
# response = tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)
# print(response)


# 以下是qwen2
