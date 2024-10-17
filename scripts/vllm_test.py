from vllm import LLM, SamplingParams
import torch
import os
import argparse
import json
import gc
from vllm.distributed.parallel_state import destroy_model_parallel

# def add(a, b):
#     return a + b

# def mul(a, b):
#     return a * b

# def cal_list(list1, list2, list3, num):
#     return list1 + list2 + list3, num
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--list1', type=list)
#     parser.add_argument('--list2', type=list)
#     parser.add_argument('--list3', type=list)
#     parser.add_argument('--num', type=int)
#     args = parser.parse_args()
#     result, nums = cal_list(args.list1, args.list2, args.list3, args.num)
#     print(json.dumps({'res': result, 'num': nums}))

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
# prompts = [
#     "Hello, my name is",
#     "The president of the United States is",
#     "The capital of France is",
#     "The future of AI is",
# ]
# # sampling_params = SamplingParams(temperature=0.8, top_p=0.8)
llm = LLM(
    model='/data00/yifei_chen/FlashRAG/models/shakechen/Llama-2-7b-chat-hf', 
    dtype=torch.bfloat16,
    enforce_eager=True,
    gpu_memory_utilization=0.5,
    trust_remote_code=True,
    )
tokenizer = llm.get_tokenizer()
system = (
    "Given a question and its answer, your task is to simplify the answer by removing redundant phrases, "
    "such as 'Answer:' or 'The answer is'. Focus on retaining only the essential nouns "
    "(e.g., names, times, places, or strings of numbers) without forming complete sentences. "
    "The simplified answer should be as concise as possible and directly relevant to the question. "
    "For instance, if the question pertains to a time, exclude unrelated details like places or people."
    "Only output the modified answer, and do not output anything else. "
    "If the answer is already sufficiently simple, respond with 'No need change', "
    "and do not include anything else. Ensure that critical information is not lost during simplification. "
    "For example, if your output consists of only one noun (e.g., a person's name, a time, a place, "
    "an object, a title, etc.) or several nouns without any redundant contentand,"
    " only output a 'No need change'. "
    "Note: Do not consider whether the answer correctly meets the query intent. "
    "Just focus on simplifying the given answer according to the format requirements."
)
conversations = tokenizer.apply_chat_template([
            {'role': 'system', 'content': system},
            {'role': 'user', 'content': 'How are you'}],
    tokenize=False,
)
prompts = [conversations] * 3
outputs = llm.generate(
    prompts,
    SamplingParams(
        temperature=0.5,
        top_p=0.9,
        max_tokens=1024,
        stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")],  # KEYPOINT HERE
    )
)
# print(outputs[0].outputs[0].text)
# Print the outputs.
for output in outputs:
    generated_text = output.outputs[0].text
    print(generated_text)
    
destroy_model_parallel()
del llm.llm_engine.model_executor.driver_worker
del llm
gc.collect()
torch.cuda.empty_cache()
del tokenizer
import time
time.sleep(5)
print('\n')
llm = LLM(
    model='/data00/yifei_chen/multi_llms_for_CoT/models/ZhipuAI/glm-4-9b-chat', 
    dtype=torch.bfloat16,
    enforce_eager=True,
    gpu_memory_utilization=0.7,
    trust_remote_code=True,
    )
tokenizer = llm.get_tokenizer()
system = (
    "You are a helpful assistant."
)
conversations = tokenizer.apply_chat_template([
            {'role': 'system', 'content': system},
            {'role': 'user', 'content': 'Hello'}],
    tokenize=False,
)
prompts = [conversations] * 3
outputs = llm.generate(
    prompts,
    SamplingParams(
        temperature=0.5,
        top_p=0.9,
        max_tokens=128,
        stop_token_ids=[tokenizer.eos_token_id],  # KEYPOINT HERE
    )
)
# print(outputs[0].outputs[0].text)
# Print the outputs.
for output in outputs:
    generated_text = output.outputs[0].text
    print(generated_text)