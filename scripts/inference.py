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
from utils import pooling
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'


class BaseInference:
    def __init__(self, config, model, tokenizer):
        self.config = config
        self.model_params = config['generator_params']
        self.model_name = config['model_name']
        self.max_input_len = config["max_input_len"]
        self.device = config["device"]
        self.model = model
        self.tokenizer = tokenizer
        self.system_prompt = ()
        self.system_prompt_refiner = ()
        self.user_prompt = ()
        self.system_prompt_naiverag = ()
        self.hlcn_examples = {
            'factual incorrectness': "LLMs provide incorrect information based on referenced documents, but without inventing new, non-existent details. For example, an LLM might incorrectly state a patient's blood sugar level as 150 mg/dL when the correct value is 120 mg/dL.",
            'misinterpretation': "LLMs misclassify the intent or context, resulting in a response that does not accurately reflect the intended meaning, or misinterpret the question's query intent. For example, the question \"What is the meaning of lead?\" might be wrongly interpreted as a query about the chemical element instead of leadership, depending on the context.",
            'logical inconsistency': "LLM is logically inconsistent in answering questions, in particular, the statements of the LLM's output are contradictory, and there is no fixed truth value.",
            'fabrication': "LLM create entirely false statements that have no basis in referenced documents, and fabrications are pure inventions by the model. For example, an LLM might invent a quote from a historical figure that never existed.",
        }
    
    def make_prompts(self, *args, **kwargs):
        pass
    
    def make_first_prompt(self, *args):
        pass
        
    
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

        return format_reference
    
    def format_judgments(self, *args):
        pass
    
    def inference(self, *args, rag_type='LLM ensemble', use_refiner=True, hlcn_type=""):
        # print(self.__class__.__name__)
        if use_refiner:
            system_prompt = self.system_prompt_refiner
        else:
            system_prompt = self.system_prompt
        # if isinstance(self, Ensemble):
        #     system_prompts = [self.system_prompt] * batch_size
        if isinstance(self, Rethinker) or isinstance(self, Checker):
            system_prompt = system_prompt.format(hlcn_type=hlcn_type, example=self.hlcn_examples[hlcn_type])
        user_prompt = self.make_prompts(*args, use_refiner=use_refiner)
        # if isinstance(self, Ensemble):
        #     print(system_prompts)
        #     print(user_prompts)
        # print(user_prompts)
        self.model.eval()
        if rag_type == 'naive':
            system_prompt = self.system_prompt_naiverag
        if 'llama' in self.model_name.lower():
            input = [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt}
            ]
            inputs = self.tokenizer.apply_chat_template(
                input,
                return_tensors='pt',
                add_generation_prompt=True
            )
            extra_eos_tokens = [
                self.tokenizer.eos_token_id,
                self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),#<[/INST]>
            ]
            inputs = inputs.to(self.model.device)
            outputs = self.model.generate(
                inputs, 
                do_sample=True,
                temperature=0.8,
                top_p=0.8,
                max_new_tokens=2048,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=extra_eos_tokens,#, 128001[128009]
                )
            response = self.tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)
        elif 'glm' in self.model_name.lower():
            input = [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt}
            ]
            text = self.tokenizer.apply_chat_template(
                input,
                tokenize=False,
                add_generation_prompt=True
            )
            inputs = self.tokenizer(text, return_tensors="pt", add_special_tokens=False, return_token_type_ids=False)
            inputs = inputs.to(self.model.device)
            outputs = self.model.generate(
                    **inputs, 
                    do_sample=True, 
                    temperature=0.8, 
                    top_p=0.9, 
                    max_new_tokens=2048,
                    pad_token_id=self.tokenizer.eos_token_id,
                    #eos_token_id=extra_eos_tokens,#, 128001[128009]
            )
            response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        elif 'qwen' in self.model_name.lower():
            input = [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt}
            ]
            text = self.tokenizer.apply_chat_template(
                input,
                tokenize=False,
                add_generation_prompt=True
            )
            inputs = self.tokenizer(text, return_tensors="pt", add_special_tokens=False, return_token_type_ids=False)
            inputs = inputs.to(self.model.device)
            outputs = self.model.generate(
                    **inputs, 
                    do_sample=True, 
                    temperature=0.8, 
                    top_p=0.9, 
                    max_new_tokens=2048,
                    pad_token_id=self.tokenizer.eos_token_id,
                    #eos_token_id=extra_eos_tokens,#, 128001[128009]
            )
            response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return response
        
        
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
        super().__init__(config, model, tokenizer)
        self.hlcn_type = ""
        self.system_prompt = (
            "Here is a question and some referenced documents, "
            "and here is also an answer and reasons for the question according to the contents of referenced documents generated by a LLM. "
            "You are now asked to determine whether the answer and reasons given have \"{hlcn_type}\" problem"
            "(that is, {example}). "
            "If so, output \"The answer and reasons have \"{hlcn_type}\" problem.\" in the first line, "
            "and output your reason for judgment in the second line. "
            "If not, only output \"The answer and reasons do not have \"{hlcn_type}\" problem.\", and do not output anything else. "
            "Please follow the output format strictly, and do not output anything else."
        )
        self.user_prompt = (
            "Question: {question}\n\n"
            "Referenced Documents: \n{documents}\n\n"
            "Reasons and Answer: {answer}"
        )
        
    def make_prompts(self, question, retrieval_results, answer, use_refiner=False):
        if use_refiner:
            format_references = retrieval_results
        else:
            format_references = self.format_reference(retrieval_results)
        input_params = {'question': question, 'documents': format_references, 'answer': answer}
        return self.user_prompt.format(**input_params)
    
class Generator(BaseInference):
    def __init__(self, config, model, tokenizer):
        super().__init__(config, model, tokenizer)
        self.system_prompt = (
            "Here is a question and some referenced documents. "
            "You are now asked to answer the question based only on the contents of the given documents."
            "Give me the reason and thought process for generating your answer, and follow them with your answer. "
            "Do not output any other words."
        )
        self.user_prompt = (
            "Question: {question}\n\n"
            "Referenced Documents: \n{documents}"
        )
        self.system_prompt_naiverag = (
            "Here is a question and some referenced documents. "
            "You are now asked to answer the question based only on the contents of the given documents. "
            "Only output a final answer for the question and do not output any redundant words."
        )
        
    def make_prompts(self, question, retrieval_results, use_refiner=False):
        if use_refiner:
            format_references = retrieval_results
        else:
            format_references = self.format_reference(retrieval_results)
        input_params = {'question': question, 'documents': format_references}
        return self.user_prompt.format(**input_params)
    
class Modifier(BaseInference):
    def __init__(self, config, model, tokenizer):
        super().__init__(config, model, tokenizer)
        self.system_prompt = (
            "Here is a question and some referenced documents, and here is also an answer to the question based on the referenced documents. "
            "There are several LLMs' judgments to determine whether this answer has problems. "
            "You are now asked to synthesize the given information to output a final answer."
            #"Do not repeat given question, do not output your reasons or any thought process, and some words like \"The answer is...\"\"because of the reason that...\". "
            "Only output a final answer for the question and do not output any redundant words.\n"
            "Here is an example for answer's format instruction:\n"
            "Question: which material is the heaviest in term of density?\n"
            "Incorrect Format Answer:\n####\n"
            "I think that Osmium is the answer.\n"
            "Osmium is the heaviest material in term of dentisy.\n"
            "Answer: Osmium.\n####\n"
            "Correct Format Answer:\nOsmium"
        )
        self.user_prompt = (
            "Question: {question}\n\n"
            "Referenced Documents: \n{documents}\n\n"
            "Answer: {answer}\n\n"
            "Judgments: \n{judgments}"
        )
        # self.system_prompt_simplify = (
        #     "Here is a question and an answer for this question generated by a LLM. "
        #     "This answer may be appropriate, or it may be a little redundant"
        # )
        
    def format_judgments(self, judgments=List):
        format_judgments = ""
        for idx, judgment in enumerate(judgments):
            format_judgments += f"Judgment {idx+1}: {judgment}\n"
        return format_judgments    
        
    def make_prompts(self, question, retrieval_results, answer, judgments, use_refiner=False):
        if use_refiner:
            format_references = retrieval_results
        else:
            format_references = self.format_reference(retrieval_results)
        format_judgments = self.format_judgments(judgments)
        input_params = {'question': question, 'documents': format_references, 'answer': answer, 'judgments': format_judgments}
        return self.user_prompt.format(**input_params)
    
class Rethinker(BaseInference):
    def __init__(self, config, model, tokenizer):
        super().__init__(config, model, tokenizer)
        self.system_prompt = (
            "Here is a question and some referenced documents, and here is also an answer to the question based on the referenced documents. "
            "Here is also the judgment you just generated to determine whether this answer has \"{hlcn_type}\" problem"
            "(that is, {example}). "
            "You can optimize your judgment appropriately to make it more accurate and reasonable based on the quality of that judgment, or make the opposite judgment, or leave your original judgment unchanged. "
            "If no revise is required, only output a \"No revise is required.\", else output the revised judgment, and do not output anything else."
        )
        self.user_prompt = (
            "Question: {question}\n\n"
            "Referenced Documents: \n{documents}\n\n"
            "Answer: {answer}\n\n"
            "Judgment: \n{judgments}"
        )
        
    def make_prompts(self, question, retrieval_results, answer, judgment, use_refiner=False):
        if use_refiner:
            format_references = retrieval_results
        else:
            format_references = self.format_reference(retrieval_results)
        input_params = {'question': question, 'documents': format_references, 'answer': answer, 'judgments': judgment}
        return self.user_prompt.format(**input_params)
        
            