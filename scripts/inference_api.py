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
from typing import List, Dict
from utils import pooling
from openai import OpenAI
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import numpy as np


class BaseInference:
    def __init__(self, model, tokenizer, sample_param):#
        self.model = model
        self.tokenizer = tokenizer
        self.sample_param = sample_param
        self.system_prompt = ()
        self.system_prompt_refiner = ()
        self.system_prompt_multiple_choice = ()
        self.user_prompt_multiple_choice = ()
        self.user_prompt = ()
        self.system_prompt_naiverag = ()
        self.system_prompt_naiverag_multiple_choice = ()
        self.system_prompt_rethink = ()
        self.user_prompt_rethink = ()
        self.hlcn_examples = {
            'Distortion of Information': (
                "This hallucination involves situations where the answer is either unverifiable or contradicts the reference information, "
                "include that The answer is unverifiable within the given context, "
                "or the answer directly conflicts with the information in the reference.\nHere is an example:\n"
                # "The question asks about a specific event in a novel, but the answer mentions an event that is factually true but not mentioned in the referenced context."
                "Question: What is the name of the person who wrote the novel 'Harry Potter'?\n"
                "Referenced document: Harry Potter is a series of seven fantasy novels written by "
                "British author J. K. Rowling. The novels chronicle the lives of a young wizard, Harry Potter...\n"
                "LLM's answer: Hemingway\n"
                "Ground truth: J. K. Rowling\n"
                "Explanation: The answer is that the name of the author of Harry Potter is Hemingway, "
                "but the reference document clearly states that Harry Potter was written by J. K. Rowling, "
                "at which time the model's answer contradicts the correct information in the reference document, and 'Distortion of Information' hallucination occurs."
                ),
            'Entity/Concept Errors': (
                "This hallucination involves misuse or misrepresentation of entities and concepts, "
                "where entities or concepts in the answer are swapped, combined, or replaced inappropriately compared to the reference. "
                "It also inlcudes that entities or concepts in the answer are swapped compared to the reference, "
                "or a term or concept is replaced by an incorrect or related concept.\nHere is an example:\n"
                "Question: How many days is the silkworm in the pupa stage?\n"
                "Referenced document: A typical silkworm can live for just over a month, during which the period "
                "from hatching to cocooning varies roughly from 25 to 32 days depending on the season, "
                "followed by 15 to 18 days as a pupa, and finally 1 to 3 days as a moth.\n"
                "LLM's answer: 25 to 32 days\n"
                "Ground truth: 15 to 18 days\n"
                "Explanation: The answer is that the silkworm has 25 to 32 days in the pupa stage, but according to the reference document, "
                "this is the time from hatching to cocooning, and the document says 15 to 18 days as a pupa. So the 'Entity/Concept Errors' hallucination occurs."
            ),
            'Logical Confusion': (
                "This hallucination involves errors in causality or conditional logic, that is, errors in logical relationships, "
                "such as incorrect causal links, overgeneralizations, or misinterpreted conditions. "
                "It also includes that a specific detail in the reference is applied too broadly in the answer, or"
                "the cause and effect are reversed or a non-causal link is mistakenly created. \nHere is an example:\n"
                "Question: What is the relationship between the information technology and big data?\n"
                "Referenced document: With the rapid development of information technology, "
                "the application of big data across various industries is becoming increasingly widespread.\n"
                "LLM's answer: Big data has promoted the rapid development of information technology.\n"
                "Ground truth: Information technology has promoted the rapid development of big data.\n"
                "Explanation: The answer is that big data can promote the development of information technology, "
                "but in the reference document, the rapid development of information technology makes big data widespread across various industries. "
                "The answer reverses the relationship between cause and effect, so the 'Logical Confusion' hallucination occurs."
            )
        }
        self.system_prompt_format_checker = ()
        self.user_prompt_format_checker = ()
    
    def make_prompts(self, *args, **kwargs):
        pass
    
    def make_prompt_check_format(self, *args):
        pass
        
    def make_prompts_multiple_choice(self, *args):
        pass
    
    def make_prompts_rethink(self, *args):
        pass
    
    def make_prompts_rethink_multiple_choice(self, *args):
        pass
    
    def make_prompts_naiverag(self, *args):
        pass
    
    def make_prompts_wrong(self, *args):
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
    
    def format_options(self, options):
        format_options = ""
        for idx, option in enumerate(options):
            format_options += f"[{idx+1}] {option}\n"
        return format_options
    
    def format_judgments(self, judgments):
        format_judgments = ""
        for idx, judgment in enumerate(judgments):
            format_judgments += f"Judgment {idx+1}: {judgment}\n"
        return format_judgments 
    def format_hallucinations(self):
        format_hallucinations = ""
        for idx, (hallucination_type, explanation) in enumerate(self.hlcn_examples.items()):
            format_hallucinations += f"[{idx+1}] {hallucination_type}: {explanation}\n"
        return format_hallucinations
    
    # def format_other_judgments(self, other_judgments):
    #     format_other_judgments = ""
    #     for idx, judgment in enumerate(other_judgments):
    #         format_other_judgments += f"Judgment {idx+1}: {judgment}\n"
    #     return format_other_judgments
    
    def inference(self, *args, rag_type='check', api_key='my_key', base_url=None, model_path=None, first_turn=True, multiple_choice=False, wrong_answer=False):
        if not wrong_answer:
            if not multiple_choice:
                if first_turn:
                    system_prompt, user_prompt = self.make_prompts(*args)
                else:
                    system_prompt, user_prompt = self.make_prompts_rethink(*args)
            else:
                if first_turn:
                    system_prompt, user_prompt = self.make_prompts_multiple_choice(*args)
                else:
                    system_prompt, user_prompt = self.make_prompts_rethink_multiple_choice(*args)
                if rag_type == 'naive':
                    system_prompt = [self.system_prompt_naiverag_multiple_choice] * len(user_prompt)
            if rag_type == 'naive':
                system_prompt, user_prompt = self.make_prompts_naiverag(*args)
        
        if isinstance(self, Generator) and wrong_answer:
            system_prompt, user_prompt = self.make_prompts_wrong(*args)
        # system_prompt = [system_prompt] * len(user_prompt)
        text = [[
                    {'role': 'system', 'content': system_prompt_},
                    {'role': 'user', 'content': user_prompt_}
                ] for system_prompt_, user_prompt_ in zip(system_prompt, user_prompt)]
        text = [
            self.tokenizer.apply_chat_template(
                text_, tokenize=False,
            add_generation_prompt=True
            ) for text_ in text
        ]
        params = {
            'temperature':0,
            'top_p': 0.8,
            'stop': "<|eot_id|>",
            'max_tokens': 128,
            'frequency_penalty':1,
        }
        output = self.model.generate(
            text,
            self.sample_param,
            use_tqdm=False,
        )
        response = [output_.outputs[0].text for output_ in output]
        # print(response)
        
        
        
        # client = OpenAI(api_key=api_key, base_url=base_url)
        # chat_outputs = client.chat.completions.create(
        #     model=model_path,
        #     messages=text,
        #     **params
        # )
        
        return response
        # return chat_outputs.choices[0].message.content
    
class Generator(BaseInference):
    def __init__(self, model=None, tokenizer=None, sample_param=None):
        super().__init__(model, tokenizer, sample_param)
        self.system_prompt = (
            # 只有错误答案
            "Given referenced documents, generate an incorrect answer (e.g., a wrong noun or concept) for the question that contradicts "
            "the correct one but is grounded in the information from the documents. Follow these steps:\n1. Generate a factually "
            "incorrect answer using a different, incorrect noun or concept, without negating the correct answer.\n2. "
            "Do not provide any reasons or explanations for the answer.\n3. Do not imply the answer is incorrect, "
            "and do not mention the correct answer.\n4. Do not use negations (e.g., 'X is not Y'); instead, "
            "generate an alternative incorrect conclusion.\n\nKeep in mind: \n1. Ensure the answer is consistent with the information "
            "in the documents, but factually wrong.\n2. Only output the incorrect answer and nothing else.\n3. Do not indicate in any way "
            "that the answer or explanation is wrong, and do not mention any other answers.\n\n"
            "The following are given documents:\n\n{documents}\n\n"
            # "Here is an example:\n"
            # "question: Which animal is the fastest on land?\n"
            # "Answer: The turtle is the fastest animal on land. "
            
            # 错误答案和原因
            # "Given a question and referenced documents, generate an incorrect answer (e.g., a wrong noun or concept) "
            # "that contradicts the correct one, but is still grounded in the documents information. Follow these steps:\n"
            # "1. Generate a factually incorrect answer using a different, incorrect noun or concept, without negating the correct answer.\n"
            # "2. Justify the incorrect answer with a brief explanation based solely on the documents, making the reasoning seem plausible.\n"
            # "3. Do not imply the answer is incorrect, avoid mentioning the correct answer.\n"
            # "4. Do not use negations (e.g., 'X is not Y'); instead, generate an alternative incorrect conclusion.\n\n"
            # "Keep in mind: \n1. Only provide the incorrect answer and justification.\n"
            # "2. Ensure the reasoning is consistent with the documents but supports the incorrect conclusion.\n"
            # "3. Do not indicate in any way that the answer or explanation is wrong, and do not mention any other answers.\n\n"
            # "Example:\n"
            # "Question: Which animal is the fastest on land?\n"
            # "Answer and Reasons: The turtle is the fastest animal on land. Document A describes how turtles can be agile in certain situations, "
            # "such as escaping predators, which could be interpreted as a form of speed."
            
            # 正确答案
            # "Given a question and referenced documents, generate a correct answer that is grounded in the information from the documents for the given question. "
            
            # 正确答案和原因
            # "Given a question and referenced documents, generate a correct answer that is grounded in the information from the documents for the given question. "
            # "And Follow the answer with a brief explanation based solely on the documents. "
        )
        self.system_prompt_multiple_choice = (
            "Given a multiple-choice question, some referenced documents and some available options. "
            "Your task is to select an incorrect option that contradicts the question's intended meaning, but is still justifiable based on the documents. "
            "When generating your option, follow these steps:\n"
            "1. Identify relevant information from the documents.\n"
            "2. Select a factually incorrect option that opposes the correct interpretation.\n"
            "3. Briefly justify the incorrect choice using evidence from the documents, making it seem plausible.\n\n"
            "Do not provide the correct option or state that the option is wrong. Ensure the explanation aligns with the document while supporting the wrong choice.\n\n"
            "Here is an example:\n"
            "Question: Which animal is the fastest on land?\n"
            "Available Options: \n[1] Butterfly\n[2] Turtle\n[3] Snake\n[4] Lizard\n\n"
            "Option and Reasons: [2]. Document A notes that turtles are agile in specific situations, such as evading predators. "
            "Document B suggests their slow movement aids survival, which could be interpreted as metaphorical speed."
        )
        self.system_prompt_wrong = (
            "Generate an incorrect answer for the given question based on the referenced documents and the correct answer. "
            "Note that:\n"
            "1. Generate a factually incorrect answer using a different, incorrect noun or concept, without negating the correct answer.\n"
            "2. Do not provide any reasons or explanations for the answer.\n"
            "3. The incorrect answer you generate should be different from the correct one.\n4. Do not use negations (e.g., 'X is not Y'); instead, "
            "generate an alternative incorrect conclusion.\n\nKeep in mind: \n1. Ensure the answer is consistent with the information "
            "in the documents, but factually wrong.\n2. Only output the incorrect answer and nothing else.\n3. Do not indicate in any way "
            "that the answer or explanation is wrong, and do not mention any other answers.\n\n"
            "The following are given documents:\n\n{documents}\n\n"
        )
        self.user_prompt = (
            "Question: {question}"
        )
        self.user_prompt_multiple_choice = (
            "Question: {question}\n\n"
            "Referenced Documents: \n{documents}\n\n"
            "Available Options: \n{options}"
        )
        self.user_prompt_wrong = (
            "Question: {question}\n"
            "Correct Answer: {answer}\n"
        )
        self.system_prompt_naiverag = (
            "Answer the question based on the given document. "
            "Only give me the answer and do not output any other words.\n"
            "The following are given documents:\n\n"
            "{documents}\n\n"
            
            # "Answer the question based on the given document."
            # "Only give me the answer and do not output any other words."
        )
        self.system_prompt_naiverag_multiple_choice = (
            "Given a multiple-choice question, some referenced documents and some available options. "
            "You are now asked to read the referenced documents carefully, and then select a correct option to answer the question based only on the contents of the given documents, "
            "and do not base it on your own internal knowledge but based only on the contents of the given documents. "
            "Only give me the index of the correct option and do not output any other words.\n\n"
            "Here is an example:\n"
            "Question: Which animal is the fastest on land?\n"
            "Available Options: \n[1] Butterfly\n[2] Turtle\n[3] Snake\n[4] Cheetah\n\n"
            "Your Option: [4]"
        )
        
    def make_prompts(self, question, retrieval_results):
        format_references = [self.format_reference(retrieval_result) for retrieval_result in retrieval_results]
        input_params_system = [{"documents": format_reference} for format_reference in format_references]
        input_params_user = [{"question": question_} for question_ in question]
        return [self.system_prompt.format(**input_param_system) for input_param_system in input_params_system], [self.user_prompt.format(**input_param_user) for input_param_user in input_params_user]
        # format_reference = self.format_reference(retrieval_results)
        # input_params_system = {"documents": format_reference}
        # input_params_user = {"question": question}
        # return self.system_prompt.format(**input_params_system), self.user_prompt.format(**input_params_user)
    
    def make_prompts_naiverag(self, question, retrieval_results):
        format_references = [self.format_reference(retrieval_result) for retrieval_result in retrieval_results]
        input_params_system = [{"documents": format_reference} for format_reference in format_references]
        input_params_user = [{"question": question_} for question_ in question]
        return [self.system_prompt_naiverag.format(**input_param_system) for input_param_system in input_params_system], [self.user_prompt.format(**input_param_user) for input_param_user in input_params_user]
        format_reference = self.format_reference(retrieval_results)
        input_params_system = {"documents": format_reference}
        input_params_user = {"question": question}
        return self.system_prompt_naiverag.format(**input_params_system), self.user_prompt.format(**input_params_user)
    
    def make_prompts_multiple_choice(self, question, retrieval_results, options, use_refiner=False):
        if use_refiner:
            format_references = retrieval_results
        else:
            format_references = [self.format_reference(retrieval_result) for retrieval_result in retrieval_results]
        
        format_options = [self.format_options(options_) for options_ in options]
        input_params = [{'question': question_, 'documents': format_reference, 'options': format_options_} for question_, format_reference, format_options_ in zip(question, format_references, format_options)]
        return [self.user_prompt_multiple_choice.format(**input_param) for input_param in input_params]
    
    def make_prompts_wrong(self, question, retrieval_results, answer):
        format_reference = self.format_reference(retrieval_results)
        input_params_system = {"documents": format_reference}
        input_params_user = {"question": question, "answer": answer}
        return self.system_prompt_wrong.format(**input_params_system), self.user_prompt_wrong.format(**input_params_user)
           
class Classifier(BaseInference):
    def __init__(self, model=None, tokenizer=None, sample_param=None):
        super().__init__(model, tokenizer, sample_param)
        self.system_prompt = (
            # "Given a question, some referenced documents, an answer for the question and its reasons, and some hallucination problems and their explanations, "
            # "follow these guidelines:\n"
            # "1. Review the question and the documents carefully. Analyze each document carefully and try to extract key information from each document.\n"
            # "2. If the answer is correct, only output a 'The answer and reasons do not have hallucination', and do not output any other words. \n"
            # "3. If the answer is incorrect, output the type of hallucination from the given hallucinations in the first line, "
            # "and output the explanation and the correct answer in the next line. \n\n"
            # "Note that:\n1. the answer may have hallucination, or it may be correct.\n2. Each answer-reason pair will have at most one type of hallucination.\n"
            # "3. If the correct answer to the question is different in different documents, choose the one that appears the most often as the correct answer."
            "Act as a critic. Given a question, referenced documents, and some hallucination error types with explanations, "
            "judge the correctness of the answer. Follow these steps:\n1. Analyze each document for exact relevant information regarding the question. \n"
            "2. Judge whether the answer strictly aligns with the provided information in the documents. \n"#Do not assume correctness for terms not explicitly stated, even if they seem semantically close.\n"
            "3. Output your judgment for the given answer first.\n"
            "4. Then, if hallucination is found, clearly identify the issue in the original answer and justify your judgment. "
            "If no hallucination is found, output only a 'The answer does not have hallucination' and nothing else.\n\n"
            "Note that:\n1. Each answer will have at most one type of hallucination.\n"
            "2. Do not output any answers, only hallucination judgments and reasons.\n\n"
            "The question, documents and hallucination errors and"
            " their explanations are given as follows:\nQuestion: {question}\n\nReferenced Documents: \n{documents}\n\nHallucination "
            "Errors and Explanations: \n{hallucinations}"
        )
        self.system_prompt_rethink = (# 尝试多模型rethink
            # "Given a question, some referenced documents, an answer for the question and its reasons, some types of hallucination problems and their explanations, "
            # "and a hallucination judgment generated by a LLM, "
            # "follow these steps:\n"
            # "1. Review the documents thoroughly to determine whether the hallucination judgment for the given answer is correct. \n"
            # "2. If the judgment is correct, only output a 'The judgment is correct', and nothing else. \n"
            # "3. If the judgment is incorrect:\n"
            # "(1) First, output: 'The judgment is incorrect'.\n"
            # "(2) Then, reassess the answer and reasons.\n"
            # "(3) If the answer and reasons do have other types of hallucinations, provide the new hallucination type in the first line. "
            # "In the second line, give a concise explanation based on the documents to justify the new hallucination judgment.\n"
            # "(4) If no hallucination is found, output only: 'The answer and reasons do not have hallucination.', and nothing else.\n\n"
            # "Key considerations:\n1. The answer may or may not have hallucination.\n2. There is at most one type of hallucination in each answer-reason pair.\n"
            # "3. Provide the revised judgment only if necessary, ensuring it reflects the correct hallucination type and the essential reasons for the revision.\n"
            # "4. If no revision is needed, ensure the output is just a 'The judgment is correct'."
            
            
            # "Given a question, some referenced documents, an answer for the question and its reasons, hallucination types and their explanations, "
             
            # "Key considerations:\n1. The new judgment should be more accurate than the original judgment, and modify some unreasonable places in the original judgment.\n"
            # "2. The new judgment should focus on whether the answer satisfies the query intent of the question.\n"
            
            
            
            # "Given a question, some referenced documents, an answer for the question and its reasons, hallucination types and their explanations, "
            # "your previous judgment, and some hallucination judgments generated by other LLMs, "
            # "follow these steps:\n"
            # "1. Review the question, the documents and the judgments carefully. \n"
            # "2. Based on other judgments' information relevant to the query intent, revise your original judgment. "
            # "3. Include a brief explanation for your revised judgment and provide the correct answer to the question.\n\n"
            # "Key considerations:\n1. The new judgment should focus on the core query intent of the question, and should be consistent with the documents.\n"
            # "2. Output the revised judgment, a brief explanation for the judgment and a correct answer, and do not output any other words. \n"
            
            
            "Act as a critic. Given a question, referenced documents, an answer for the question, "
            "and some hallucination errors with explanations, assess your original judgment by viewing other LLMs' judgments."
            " Follow these steps:\n"
            "1. Carefully read the provided documents and other LLMs' judgments.\n"
            "2. Assess whether your original hallucination judgment was correct and refine it if needed based on valid insights.\n"
            "3. Output only your final revised judgment and a concise explanation. "
            "Do not mention other LLMs' judgments or provide any answers. Keep your output brief and to the point.\n\n"# 新加的
            "The question, documents, answer, original judgment, and hallucination errors and their explanations are given as follows:"
            "\nQuestion: {question}\n\nReferenced Documents: \n{documents}\n\nAnswer: {answer}\n\nHallucination Errors and Explanations:"
            " \n{hallucinations}\n"
        )
        self.user_prompt = (
            "Given Answer: {answer}\n\n"# and Reasons
        )
        self.user_prompt_rethink = (
            "Original Judgment: \n{judgment}\n\n"
            "Other LLMs' Judgments:\n{judgments}"
        )
        
    def make_prompts(self, questions, retrieval_results, answers):
        format_references = [self.format_reference(retrieval_result) for retrieval_result in retrieval_results]
        input_params_system = [{'question': question, 'documents': format_reference, 'hallucinations': self.format_hallucinations()} for question, format_reference in zip(questions, format_references)]
        input_params_user = [{'answer': answer} for answer in answers]
        return [self.system_prompt.format(**input_param_system) for input_param_system in input_params_system], [self.user_prompt.format(**input_param_user) for input_param_user in input_params_user]
        format_reference = self.format_reference(retrieval_results)
        input_params_system = {'question': questions, 'documents': format_reference, 'hallucinations': self.format_hallucinations()}
        input_params_user = {'answer': answers}
        return self.system_prompt.format(**input_params_system), self.user_prompt.format(**input_params_user)
    
    def make_prompts_rethink(self, questions, retrieval_results, answers, judgments, other_judgments):
        format_references = [self.format_reference(retrieval_result) for retrieval_result in retrieval_results]
        format_other_judgments = [self.format_judgments(other_judgment) for other_judgment in other_judgments]
        input_params_system = [{'question': question, 'documents': format_reference, 'answer': answer, 'hallucinations': self.format_hallucinations()} for question, format_reference, answer in zip(questions, format_references, answers)]
        input_params_user = [{'judgment': judgment, 'judgments': format_other_judgment} for judgment, format_other_judgment in zip(judgments, format_other_judgments)]
        return [self.system_prompt_rethink.format(**input_param_system) for input_param_system in input_params_system], [self.user_prompt_rethink.format(**input_param_user) for input_param_user in input_params_user]
        format_reference = self.format_reference(retrieval_results)
        format_other_judgments = self.format_judgments(other_judgments)
        input_params_system = {'question': questions, 'documents': format_reference, 'answer': answers, 'hallucinations': self.format_hallucinations()}
        input_params_user = {'judgment': judgments, 'judgments': format_other_judgments}
        return self.system_prompt_rethink.format(**input_params_system), self.user_prompt_rethink.format(**input_params_user)

class Rethinker(BaseInference):
    def __init__(self, model=None, tokenizer=None, sample_param=None):
        super().__init__(model, tokenizer, sample_param)
        self.system_prompt = (
            "Act as a modifier. Give a question, some referenced documents and an answer. Your task is to reflect on your previous judgment about the answer. "
            "Follow these steps:\n"
            "1. Review the documents and your original judgment carefully. \n"
            "2. According to the document contents, adjust your judgment step by step, making it more accurate if necessary.\n\n"
            "Note that: Be concise in your response; however, capture all of the essential information.\n\n"
            "The question, documents, and answer are given as follows:\n"
            "Question: {question}\n\n"
            "Referenced Documents: \n{documents}\n\n"
            "Answer: {answer}\n"
        )
        self.user_prompt = (
            "Original Judgment: {judgment}\n"
        )
        
    def make_prompts(self, question, retrieval_results, answers, judgment):
        format_references = self.format_reference(retrieval_results)
        input_params_system = {'question': question, 'documents': format_references, 'answer': answers}
        input_params_user = {'judgment': judgment}
        return self.system_prompt.format(**input_params_system), self.user_prompt.format(**input_params_user)
  
class Modifier(BaseInference):
    def __init__(self, model=None, tokenizer=None, sample_param=None):
        super().__init__(model, tokenizer, sample_param)
        self.system_prompt = (
            # "Given a question, some referenced documents, an answer for the question and its reasons, and some judgments for the answer's hallucination problem, "
            # "follow these steps:\n"
            # "1. Identify the query intent of the question and determine what type of correct answer should be (e.g., a name, a time, a place, etc.).\n"
            # "2. Revise the judgments, and answer to match both the query intent and the exact format found in the documents. \n"
            # "3. Revise the answer according to the question, the documents and judgments, and output a final correct answer. \n\n"
            # "Note that:\n"
            # "1. The answer should be strictly based on the documents and the judgments. Can not create the answer that does not exist in the documents.\n"
            # "2. The answer format should be the same as the answer format found in the documents (e.g., order of name, time, place, etc.). \n"
            # "3. Be careful to distinguish between singular and plural answers in the document.\n"
            
            
            "Act as a moderator. Given a question, referenced documents, an answer for the question, combine other LLMs' judgments, "
            "modify the given answer and generate a correct answer. Follow these steps:\n1. Review the reference documents and other "
            "models' judgments.\n2. Based on the judgments, modify the original answer to correctly address the given question.\n3. "
            "Output only the final corrected answer, without any additional explanation or comments. \n\nThe question, documents and "
            "answer are given as follows:\nQuestion: {question}\n\nReferenced Documents: \n{documents}\n\nAnswer: {answer}\n\n"
        )
        self.system_prompt_rethink = (
            "Act as a modifier. Given a question, referenced documents and an original answer for the question, assess other LLMs' judgments "
            "about the original answer and further optimize the revised answer. Follow these steps:\n1. Review the reference documents and "
            "other models' judgments.\n2. Based on the judgments and the original answer, improve the revised answer to accurately address "
            "the question.\n3. If the revised answer is already correct, output only 'The answer is correct' and do not output "
            "any other words. \n\nThe question, documents and the original answer are given as follows:\nQuestion: {question}\n\n"
            "Referenced Documents: \n{documents}\n\nOriginal Answer: {answer}\n\n"
        )
        self.user_prompt = (
            "Other LLMs' Judgment: {judgment}"
        )
        self.user_prompt_rethink = (
            "Judgment: {judgment}\n\n"
            "Modified Answer: {new_answer}"
        )
        
    def make_prompts(self, questions, retrieval_results, answers, judgments):
        format_references = [self.format_reference(retrieval_result) for retrieval_result in retrieval_results]
        format_judgments = [self.format_judgments(judgment) for judgment in judgments]
        input_params_system = [{'question': question, 'documents': format_reference, 'answer': answer} for question, format_reference, answer in zip(questions, format_references, answers)]
        input_params_user = [{'judgment': judgment} for judgment in format_judgments]
        return [self.system_prompt.format(**input_param_system) for input_param_system in input_params_system], [self.user_prompt.format(**input_param_user) for input_param_user in input_params_user]
        format_reference = self.format_reference(retrieval_result)
        format_judgments = self.format_judgments(judgments)
        input_params_system = {'question': questions, 'documents': format_reference, 'answer': answers}
        input_params_user = {'judgment': format_judgments}
        return self.system_prompt.format(**input_params_system), self.user_prompt.format(**input_params_user)
    
    def make_prompts_rethink(self, questions, retrieval_results, answers, judgments, new_answers):
        format_references = [self.format_reference(retrieval_result) for retrieval_result in retrieval_results]
        format_judgments = [self.format_judgments(judgment) for judgment in judgments]
        input_params_system = [{'question': question, 'documents': format_reference, 'answer': answer} for question, format_reference, answer in zip(questions, format_references, answers)]
        input_params_user = [{'judgment': judgment, 'new_answer': new_answer} for judgment, new_answer in zip(format_judgments, new_answers)]
        return [self.system_prompt_rethink.format(**input_param_system) for input_param_system in input_params_system], [self.user_prompt_rethink.format(**input_param_user) for input_param_user in input_params_user]
        format_reference = self.format_reference(retrieval_results)
        format_judgments = self.format_judgments(judgments)
        input_params_system = {'question': questions, 'documents': format_reference, 'answer': answers}
        input_params_user = {'judgment': format_judgments, 'new_answer': new_answers}
        return self.system_prompt_rethink.format(**input_params_system), self.user_prompt_rethink.format(**input_params_user)
    
class Simplifier(BaseInference):
    def __init__(self, model=None, tokenizer=None, sample_param=None):
        super().__init__(model, tokenizer, sample_param)
        self.system_prompt = (
            "Act as a momdifier. Given a question, simplify the given answer by following these steps:\n1. Review the question's "
            "intent and the content of the answer. \n2. Simplify the answer by removing unnecessary parts, keeping only the key "
            "information that satisfies the query.\n3. For yes\/no questions, the simplified answer should only contain 'yes' or 'no."
            "'\n4. For open-ended or fact-based questions, keep only the essential information from the answer.\n5. "
            "If the answer is already concise, output only: 'No need change.'\n\nThe question is given as follows:\nQuestion: {question}"
        )
        self.user_prompt = (
            "Answer to be simplified: {answer}\n"
        )
        
    def make_prompts(self, questions, answers):
        input_params_system = [{'question': question} for question in zip(questions)]
        input_params_user = [{'answer': answer} for answer in answers]
        return [self.system_prompt.format(**input_param_system) for input_param_system in input_params_system], [self.user_prompt.format(**input_param_user) for input_param_user in input_params_user]
        input_params_system = {'question': questions}
        input_params_user = {'answer': answers}
        return self.system_prompt.format(**input_params_system), self.user_prompt.format(**input_params_user)
