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
from .utils import pooling
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'


class BaseInference:
    def __init__(self, config, model, tokenizer):
        self.config = config
        self.model_name = config['model_name']
        self.max_input_len = config["max_input_len"]
        self.device = config["device"]
        self.model = model
        self.tokenizer = tokenizer
        self.use_refiner = config['use_refiner']
        self.system_prompt = ()
        self.system_prompt_refiner = ()
        self.user_prompt = ()
        self.system_prompt_naiverag = ()
    
    def make_prompts(self, *args, **kwargs):
        pass
    
    def make_first_prompt(self, *args):
        pass
    
    def make_judgment_prompt(self, *args):
        pass
    
    def inference(self, *args, rag_type='LLM ensemble', batch_size=4, use_refiner=True):
        # print(self.__class__.__name__)
        if use_refiner:
            system_prompts = [self.system_prompt_refiner] * batch_size
        else:
            system_prompts = [self.system_prompt] * batch_size
        # if isinstance(self, Ensemble):
        #     system_prompts = [self.system_prompt] * batch_size
        user_prompts = self.make_prompts(*args, use_refiner=use_refiner)
        # if isinstance(self, Ensemble):
        #     print(system_prompts)
        #     print(user_prompts)
        # print(user_prompts)
        if rag_type == 'naive':
            system_prompts = [self.system_prompt_naiverag] * batch_size
       
        if self.is_chat:
            inputs = [
                [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
                for system_prompt, user_prompt in zip(system_prompts, user_prompts)
                ]
            if self.is_openai:
                for input in inputs:
                    for item in input:
                        if item["role"] == "system":
                            item["role"] == "assistant"
            # else:
            #     input = self.tokenizer.apply_chat_template(input, tokenize=False, add_generation_prompt=True)
        else:
            inputs = ["\n\n".join([prompt for prompt in [system_prompt, user_prompt] if prompt != ""]) for system_prompt, user_prompt in zip(system_prompts, user_prompts)]
        # if "llama" in self.model_name.lower():
        #     terminators = [
        #         self.tokenizer.eos_token_id,
        #         self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        #     ]
        #     if "eos_token_id" in self.generator_params:
        #         self.generator_params["eos_token_id"].extend(terminators)
        #     else:
        #         self.generator_params["eos_token_id"] = terminators
        # if isinstance(self, Judgment):
        #     print(input)
        if "baichuan" in self.model_name.lower():
            responses_split = [self.model.chat(self.tokenizer, input) for input in inputs]
        else:
            input_texts = [
                        self.tokenizer.apply_chat_template(
                        input, 
                        tokenize=False,
                        add_generation_prompt=True,
                    ) for input in inputs
                ]
            # .to(self.device)
            # self.model.to(self.device)
            # print(input_ids.device)
            # print(self.model.device)
            # input_ids.to(self.model.device)
            # print(input_texts[0])
            input_ids = self.tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True).to(self.model.device)
            # input_ids = input_ids.to(self.device)
            # input_ids = [input_id.cuda(int(self.device[-1])) for input_id in input_ids]
            outputs = self.model.generate(
                **input_ids,
                streamer=None,
                **self.generator_params,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            # generate_ids = [
            #     output_ids[len(input_id):] for input_id, output_ids in zip(input_ids, outputs)
            # ]
            generate_ids = outputs
            responses = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True)
            # print(input_texts)
            # print(responses)
            responses_split = []
            # if self.is_chat:
            for i in range(len(responses)):
                response = responses[i]
                assistant = ""
                if "assistant\n\n" in response:
                    index = response.rfind("assistant\n\n")
                    assistant = "assistant\n\n"
                elif "assistant\n" in response:
                    index = response.rfind("assistant\n")
                    assistant = "assistant\n"
                else:
                    response_split = response[len(system_prompts[i] + "\n\n" + user_prompts[i]):]
                    responses_split.append(response_split)
                    continue
                if index != -1:
                    response_split = response[index + len(assistant):]
                else:
                    response_split = response
                responses_split.append(response_split)
                
                    
            # responses_split = [response.split("assistant\n", 1)[-1] for response in responses]
            # print(responses)
            # print('\n')
        print(responses)
        print(responses_split)
        return responses_split
    
    def format_reference(self, retrieval_result):
        format_reference = ""
        for idx, doc_item in enumerate(retrieval_result):
            # if isinstance(doc_item, list):
            #     print(retrieval_result)
            #     print(doc_item)
            content = doc_item["contents"]
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            # if format_reference is not None:
            #     format_reference += format_reference.format(idx=idx, title=title, text=text)
            # else:
            #     format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"
            # if format_reference is None:
            #     format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"
            text = re.sub(r'[\{\}]', '', text)
            format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"
            format_reference.format(idx=idx, title=title, text=text)

        return format_reference
        
        
# class Ensemble(BaseInference):
#     def __init__(self, config, model, tokenizer):
#         super().__init__(config, model, tokenizer)
#         self.count = 0
#         self.system_prompt = (
#             "Here is a question and some referenced documents. "
#             "Here are also answers and reasons for questions that other LLMs generate based on the referenced documents. "
#             "You are now asked to output a final answer to the given question based on the referenced documents and other LLMs' answers and reasons, "
#             "only give me the final answer and do not output any other words."
#         )
#         self.user_prompt = (
#             "Question: {question}\n"
#             "Referenced Documents:\n {documents}\n\n"
#             "Other LLMs' answers and reasons:\n{answers}"
#         )
        
#     def make_prompts(self, questions, documents, answers, use_refiner=True):
#         if use_refiner:
#             format_documents = documents
#         else:
#             format_documents = [self.format_reference(document) for document in documents]
#         format_answers = [self.format_reasons_answers(answer) for answer in answers]
        
#         input_params = [{"question": question, "documents": format_document, "answers": format_answer} for question, format_document, format_answer in zip(questions, format_documents, format_answers)]
#         return [self.user_prompt.format(**input_param) for input_param in input_params]

#     def format_reasons_answers(self, answers):
#         # 传入list, list每个元素是一个元组
#         format_reference = ""
#         for idx, doc_item in enumerate(answers):
#             format_reference += f"Reason and Answer {idx+1}: \nAnswer: {doc_item[1]}\nReason: {doc_item[0]}\n\n"
#         return format_reference
        
# class Generate(BaseInference):
#     def __init__(self, config, model, tokenizer):
#         super().__init__(config, model, tokenizer)
#         self.system_prompt = (
#             "Here is a question and some referenced documents. "
#             "You are now asked to answer the question based only on the contents of the given documents."
#             "Give me the reason and thought process for generating your answer on the first line, then only output your answer to the question on the next line,"
#             "Please follow exactly the output format I've given you, and do not output any other words."
#         )
#         self.system_prompt_refiner = (
#             "Here is a question and a referenced passage. "
#             "You are now asked to answer the question based only on the contents of the given passage."
#             "Give me the reason and thought process for generating your answer on the first line, then only output your answer to the question on the next line,"
#             "Please follow exactly the output format I've given you, and do not output any other words."
#         )
#         self.system_prompt_naiverag = (
#             "Here is a question and some referenced documents. "
#             "You are now asked to answer the question based only on the contents of the given documents."
#             "Only give me the answer to the question, and do not output any other words."
#         )
#         #   , 
#         self.user_prompt = (
#             "Question: {question}\n"
#             "Referenced Documents: \n{reference}"
#         )
        
#     def make_prompts(self, questions, retrieval_results, use_refiner=True):
#         if use_refiner:
#             formatted_references = retrieval_results
#         else:
#             formatted_references = [self.format_reference(retrieval_result) for retrieval_result in retrieval_results]
#         # print(formatted_reference)
#         input_params = [{"question": question, "reference": formatted_reference} for question, formatted_reference in zip(questions, formatted_references)]
#         return [self.user_prompt.format(**input_param) for input_param in input_params]
    
class ExtractiveRefiner(BaseInference):
    def __init__(self, config, model, tokenizer):
        super().__init__(config, model, tokenizer)
        self.topk = config["refiner_topk"]
        self.pooling_method = config["refiner_pooling_method"]
        self.encode_max_length = config["refiner_encode_max_length"]
        self.model, self.tokenizer = model, tokenizer
        
    def encode(self, query_list: List[str], is_query=True):
        if is_query:# 判断处理query还是处理documents
            query_list = [f"query: {query}" for query in query_list]
        else:
            query_list = [f"passage: {query}" for query in query_list]

        inputs = self.tokenizer(
            query_list, max_length=self.encode_max_length, padding=True, truncation=True, return_tensors="pt"
        )
        inputs = {k: v.cuda() for k, v in inputs.items()}

        if "T5" in type(self.model).__name__:
            # T5-based retrieval model
            decoder_input_ids = torch.zeros((inputs["input_ids"].shape[0], 1), dtype=torch.long).to(
                inputs["input_ids"].device
            )
            output = self.model(**inputs, decoder_input_ids=decoder_input_ids, return_dict=True)
            query_emb = output.last_hidden_state[:, 0, :]

        else:
            output = self.model(**inputs, return_dict=True)
            query_emb = pooling(
                output.pooler_output, output.last_hidden_state, inputs["attention_mask"], self.pooling_method
            )

        query_emb = query_emb.detach().cpu().numpy()
        query_emb = query_emb.astype(np.float32)
        return query_emb

    def batch_run(self, questions, retrieval_results, batch_size=16):
        # only use text
        retrieval_results = [
            ["\n".join(doc_item["contents"].split("\n")[1:]) for doc_item in item_result]
            for item_result in retrieval_results
        ]

        # split into sentences: [[sent1, sent2,...], [...]]
        # print(retrieval_results)
        # sent_lists = [
        #     [i.strip() for i in re.split(r"(?<=[.!?])\s+", res) if len(i.strip()) > 5] for res in retrieval_results
        # ]
        sent_lists = retrieval_results
        score_lists = []  # matching scores, size == sent_lists
        for idx in tqdm(range(0, len(questions), batch_size), desc="Refining process: "):
            batch_questions = questions[idx : idx + batch_size]
            batch_sents = sent_lists[idx : idx + batch_size]

            question_embs = self.encode(batch_questions, is_query=True)
            sent_embs = self.encode(sum(batch_sents, []), is_query=False)  # n*d
            scores = question_embs @ sent_embs.T
            start_idx = 0
            for row_score, single_list in zip(scores, sent_lists):
                row_score = row_score.tolist()
                score_lists.append(row_score[start_idx : start_idx + len(single_list)])
                start_idx += len(single_list)

        # select topk sents
        retain_lists = []
        for sent_scores, sent_list in zip(score_lists, sent_lists):
            if len(sent_scores) < self.topk:
                retain_lists.append(sent_list)
                continue
            topk_idxs = torch.topk(torch.Tensor(sent_scores), self.topk).indices.tolist()
            retain_lists.append([sent_list[idx] for idx in sorted(topk_idxs)])

        return [" ".join(sents) for sents in retain_lists]
    
class AbstractiveRecompRefiner(BaseInference):
    def __init__(self, config, model, tokenizer):
        super().__init__(config, model, tokenizer)
        self.max_input_length = config["refiner_max_input_length"]
        self.max_output_length = config["refiner_max_output_length"]
        self.tokenizer = tokenizer
        self.model = model
        
    def batch_run(self, questions, retrieval_results, batch_size=2):
        # only use text
        retrieval_results = [
            ["\n".join(doc_item["contents"].split("\n")[1:]) for doc_item in item_result]
            for item_result in retrieval_results
        ]

        # input processing in recomp training format
        format_inputs = [
            "Question: {question}\n Document: {document}\n Summary: ".format(
                question=question, document="\n".join(docs)
            )
            for question, docs in zip(questions, retrieval_results)
        ]

        results = []
        for idx in tqdm(range(0, len(format_inputs), batch_size), desc="Refining process: "):
            batch_texts = format_inputs[idx : idx + batch_size]
            batch_inputs = self.tokenizer(
                batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_input_length
            ).to(self.model.device)

            batch_outputs = self.model.generate(**batch_inputs, max_length=self.max_output_length, pad_token_id=self.tokenizer.eos_token_id)

            batch_outputs = self.tokenizer.batch_decode(
                batch_outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            batch_outputs = [
                batch_output[len(format_input):] for batch_output, format_input in zip(batch_outputs, batch_texts)
            ]

            results.extend(batch_outputs)

        return results
    
class Checker(BaseInference):
    def __init__(self, config, model, tokenizer):
        super.__init__(config, model, tokenizer)
        self.hlcn_type = ""
        self.system_prompt = (
            "Here is a question and some referenced documents, "
            "and here is also an answer and reasons for the question according to the contents of referenced documents generated by a LLM. "
            "You are now asked to determine whether the answer and reasons given have \"{hlcn_type}\" hallucination. "
            "If so, output \"The answer and reasons have \"{hlcn_type}\" hallucination.\" in the first line, "
            "and output your reason for judgment in the second line. "
            "If not, only output \"The answer and reasons do not have \"{hlcn_type}\" hallucination.\". "
            "Please follow the output format strictly, and do not output anything else."
        )
        self.user_prompt = (
            "Question: {question}"
            "Referenced Documents: {documents}"
            "Reasons and Answer: {answer}"
        )
        
    def make_prompts(self, questions, retrieval_results, answers):
        if self.use_refiner:
            format_references = retrieval_results
        else:
            format_references = [self.format_reference(retrieval_result) for retrieval_result in retrieval_results]
            