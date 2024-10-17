# screen -S fei_api0 -X stuff "CUDA_VISIBLE_DEVICES=0 vllm serve /data00/LLaMA-3-8b-Instruct/ --port 114514 --gpu_memory_utilization 0.6 --trust_remote_code --max_model_len 8192 --dtype bfloat16 --enforce-eager\n"

# SECONDS=0
# # 等待模型1端口开启
# while ! nc -z localhost 114514; do
#   echo "Waiting for model 1 to start... Total wait time: ${SECONDS} seconds"
#   sleep 5
# done
# screen -S fei_api1 -X stuff "CUDA_VISIBLE_DEVICES=2 vllm serve /data00/jiajie_jin/model/Mistral-7B-Instruct-v0.3 --port 1557 --gpu_memory_utilization 0.45 --trust_remote_code --max_model_len 4096 --dtype bfloat16 --enforce-eager\n"
# 等待模型2端口开启
# while ! nc -z localhost 1557; do
#   echo "Waiting for model 2 to start... Total wait time: ${SECONDS} seconds"
#   sleep 5
# done
# screen -S fei_api2 -X stuff "CUDA_VISIBLE_DEVICES=2 vllm serve /data00/yifei_chen/multi_llms_for_CoT/models/Qwen/Qwen2___5-7B-Instruct --port 1919810 --gpu_memory_utilization 0.6 --trust_remote_code --max_model_len 4096 --dtype bfloat16 --enforce-eager\n"
# screen -S fei_api3 -X stuff "CUDA_VISIBLE_DEVICES=0,1,2 vllm serve /data00/yifei_chen/FlashRAG/models/shakechen/Llama-2-7b-chat-hf --port 4396 --gpu_memory_utilization 0.25 --trust_remote_code --max_model_len 4096\n"
# 等待模型3端口开启
# while ! nc -z localhost 1919810; do
#   echo "Waiting for model 3 to start... Total wait time: ${SECONDS} seconds"
#   sleep 5
# done
# sleep 30

# export CUDA_VISIBLE_DEVICES=0,1
# python run_checkRAG.py --RAG_type check --batch_size 16 --dataset_name nq --judgments_rounds 2 --rethink_rounds 2 --modify_rounds 0 --simplify_rounds 1 --pipeline_type single_hlcn #--is_multiple_choice hotpotqa
# python evaluate.py --eval_type check --dataset_name nq --pipeline_type single_hlcn #hotpotqa
# python run_checkRAG.py --RAG_type naive --batch_size 16 --dataset_name nq --temperature 0 --top_p 0.8 --pipeline_type single_hlcn #--is_multiple_choice hotpotqa 
# python evaluate.py --eval_type naive --dataset_name nq --pipeline_type single_hlcn #hotpotqa



# python run_checkRAG.py --RAG_type check --batch_size 16 --dataset_name hotpotqa --judgments_rounds 2 --rethink_rounds 2 --modify_rounds 0 --simplify_rounds 1 --pipeline_type single_hlcn #--is_multiple_choice hotpotqa
# python evaluate.py --eval_type check --dataset_name hotpotqa --pipeline_type single_hlcn #hotpotqa
# python run_checkRAG.py --RAG_type naive --batch_size 16 --dataset_name hotpotqa --temperature 0 --top_p 0.8 --pipeline_type single_hlcn #--is_multiple_choice hotpotqa 
# python evaluate.py --eval_type naive --dataset_name hotpotqa --pipeline_type single_hlcn #hotpotqa



# python run_checkRAG.py --RAG_type check --batch_size 16 --dataset_name triviaqa --judgments_rounds 1 --rethink_rounds 2 --modify_rounds 0 --simplify_rounds 1 --pipeline_type single_hlcn #--is_multiple_choice hotpotqa
# python evaluate.py --eval_type check --dataset_name triviaqa --pipeline_type single_hlcn #hotpotqa
# python run_checkRAG.py --RAG_type naive --batch_size 16 --dataset_name triviaqa --temperature 0 --top_p 0.8 --pipeline_type single_hlcn #--is_multiple_choice hotpotqa 
# python evaluate.py --eval_type naive --dataset_name triviaqa --pipeline_type single_hlcn #hotpotqa



# python run_checkRAG.py --RAG_type check --batch_size 16 --dataset_name hotpotqa --judgments_rounds 2 --rethink_rounds 2 --modify_rounds 0 --simplify_rounds 0 --pipeline_type single_hlcn #--is_multiple_choice hotpotqa
# python evaluate.py --eval_type check --dataset_name hotpotqa --pipeline_type single_hlcn #hotpotqa
# python run_checkRAG.py --RAG_type naive --batch_size 16 --dataset_name hotpotqa --temperature 0 --top_p 0.8 --pipeline_type single_hlcn #--is_multiple_choice hotpotqa 
# python evaluate.py --eval_type naive --dataset_name hotpotqa --pipeline_type single_hlcn #hotpotqa
#!/bin/bash

# kill -9 $(pidof /data00/yifei_chen/yifei/bin/python)
# kill -9 $(pidof /data00/yifei_chen/yifei/bin/python)
# kill -9 $(pidof /data00/yifei_chen/yifei/bin/python)




python run_checkRAG.py --RAG_type check --batch_size 64 --dataset_name nq --judgments_rounds 1 --rethink_rounds 2 --modify_rounds 0 --simplify_rounds 0 --pipeline_type single_hlcn #--is_multiple_choice hotpotqa
python evaluate.py --eval_type check --dataset_name nq --pipeline_type single_hlcn #hotpotqa
python run_checkRAG.py --RAG_type naive --batch_size 64 --dataset_name nq --temperature 0 --top_p 0.8 --pipeline_type single_hlcn #--is_multiple_choice hotpotqa 
python evaluate.py --eval_type naive --dataset_name nq --pipeline_type single_hlcn #hotpotqa



python run_checkRAG.py --RAG_type check --batch_size 80 --dataset_name hotpotqa --judgments_rounds 1 --rethink_rounds 2 --modify_rounds 0 --simplify_rounds 0 --pipeline_type single_hlcn #--is_multiple_choice hotpotqa
python evaluate.py --eval_type check --dataset_name hotpotqa --pipeline_type single_hlcn #hotpotqa
python run_checkRAG.py --RAG_type naive --batch_size 80 --dataset_name hotpotqa --temperature 0 --top_p 0.8 --pipeline_type single_hlcn #--is_multiple_choice hotpotqa 
python evaluate.py --eval_type naive --dataset_name hotpotqa --pipeline_type single_hlcn #hotpotqa