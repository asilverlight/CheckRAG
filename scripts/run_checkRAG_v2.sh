screen -S fei_api0 -X stuff "CUDA_VISIBLE_DEVICES=0 vllm serve /data00/LLaMA-3-8b-Instruct/ --port 114514 --gpu_memory_utilization 0.5 --trust_remote_code --max_model_len 8192 --dtype bfloat16\n"

screen -S fei_api1 -X stuff "CUDA_VISIBLE_DEVICES=1 vllm serve /data00/jiajie_jin/model/Mistral-7B-Instruct-v0.3 --port 1557 --gpu_memory_utilization 0.5 --trust_remote_code --max_model_len 8192 --dtype bfloat16\n"
 
screen -S fei_api2 -X stuff "CUDA_VISIBLE_DEVICES=2 vllm serve /data00/yifei_chen/multi_llms_for_CoT/models/Qwen/Qwen2___5-7B-Instruct --port 1919810 --gpu_memory_utilization 0.5 --trust_remote_code --max_model_len 8192 --dtype bfloat16\n"

sleep 10


# python run_checkRAG.py --RAG_type check --batch_size 32 --dataset_name nq --judgments_rounds 1 --rethink_rounds 2 --modify_rounds 0 --simplify_rounds 0 --pipeline_type single_hlcn #--is_multiple_choice hotpotqa
# python evaluate.py --eval_type check --dataset_name nq --pipeline_type single_hlcn #hotpotqa
# python run_checkRAG.py --RAG_type naive --batch_size 32 --dataset_name nq --temperature 0 --top_p 0.8 --pipeline_type single_hlcn #--is_multiple_choice hotpotqa 
# python evaluate.py --eval_type naive --dataset_name nq --pipeline_type single_hlcn #hotpotqa



# python run_checkRAG.py --RAG_type check --batch_size 32 --dataset_name hotpotqa --judgments_rounds 1 --rethink_rounds 2 --modify_rounds 0 --simplify_rounds 0 --pipeline_type single_hlcn #--is_multiple_choice hotpotqa
# python evaluate.py --eval_type check --dataset_name hotpotqa --pipeline_type single_hlcn #hotpotqa
# python run_checkRAG.py --RAG_type naive --batch_size 32 --dataset_name hotpotqa --temperature 0 --top_p 0.8 --pipeline_type single_hlcn #--is_multiple_choice hotpotqa 
# python evaluate.py --eval_type naive --dataset_name hotpotqa --pipeline_type single_hlcn #hotpotqa



python run_checkRAG.py --RAG_type check --batch_size 32 --dataset_name 2wikimultihopqa --judgments_rounds 1 --rethink_rounds 2 --modify_rounds 0 --simplify_rounds 0 --pipeline_type single_hlcn #--is_multiple_choice hotpotqa
python evaluate.py --eval_type check --dataset_name 2wikimultihopqa --pipeline_type single_hlcn #hotpotqa
python run_checkRAG.py --RAG_type naive --batch_size 32 --dataset_name 2wikimultihopqa --temperature 0 --top_p 0.8 --pipeline_type single_hlcn #--is_multiple_choice hotpotqa 
python evaluate.py --eval_type naive --dataset_name 2wikimultihopqa --pipeline_type single_hlcn #hotpotqa



kill -9 $(pidof /data00/yifei_chen/yifei/bin/python)