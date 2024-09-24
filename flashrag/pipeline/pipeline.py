from flashrag.evaluator import Evaluator
from flashrag.dataset.utils import split_dataset, merge_dataset
from flashrag.utils import get_retriever, get_generator, get_refiner, get_judger
from flashrag.prompt import PromptTemplate
import datetime, time
import copy
from tqdm import tqdm
from flashrag.dataset.dataset import Dataset

class BasicPipeline:
    """Base object of all pipelines. A pipeline includes the overall process of RAG.
    If you want to implement a pipeline, you should inherit this class.
    """

    def __init__(self, config, prompt_template=None):
        self.config = config
        self.device = config["device"]
        self.retriever = None
        self.evaluator = Evaluator(config)
        self.save_retrieval_cache = config["save_retrieval_cache"]
        # print(config)
        if prompt_template is None:
            prompt_template = PromptTemplate(config)
        self.prompt_template = prompt_template
        self.retrieval_method = config['retrieval_method']
        self.generator_model = config['generator_model']
        self.refiner_name = config['refiner_name']
        self.batch_size = config['generator_batch_size']

    def run(self, dataset):
        """The overall inference process of a RAG framework."""
        pass
    
    def run_multi(self, config, dataset, queries):
        """将其他model的输出作为部分输入"""
        generator_multi = config['generator_multi']
        generator_multi = [{**d, 'device': config['device']} for d in generator_multi]
        result = dict()
        # for refiner_ in refiner_multi:
        #     if refiner_['refiner_name'] == None:
        #         refiner_['refiner_name'] = 'None'
        for i in range(len(generator_multi)):
            config_new = copy.deepcopy(config)
            for key in generator_multi[i]:
                if key in config_new:
                    config_new[key] = generator_multi[i][key]
            generator = get_generator(config_new) 
            # print(input_prompts)
            pred_answer_list = generator.generate(queries)
            result[generator_multi[i]['generator_model']] = pred_answer_list
            del generator
            del config_new
        result = list(map(list, zip(*result.values())))
        # print(result)
        # return
        generator_num = len(generator_multi)
        for i in range(len(dataset.data)):
            current_result = dataset.data[i].output['retrieval_result']
            for j in range(0, generator_num):
                new_dict = dict()
                new_dict['id'] = '_'
                new_dict['title'] = queries[i]
                new_dict['contents'] = result[i][j]
                current_result.append(new_dict)
                
        return dataset
            

    def evaluate(self, dataset, do_eval=True, pred_process_fun=None):
        """The evaluation process after finishing overall generation"""

        if pred_process_fun is not None:
            raw_pred = dataset.pred
            processed_pred = [pred_process_fun(pred) for pred in raw_pred]
            dataset.update_output("raw_pred", raw_pred)
            dataset.update_output("pred", processed_pred)

        if do_eval:
            # evaluate & save result
            eval_result = self.evaluator.evaluate(dataset)
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_entry = f"[{current_time}] Result {eval_result}, Retrieval method: {self.retrieval_method}, Generator model: {self.generator_model}, Refiner name: {self.refiner_name}, Use multi generator: {self.config['multi_model']}\n"
            log_file = open('/data00/yifei_chen/FlashRAG/result.log', 'a')
            log_file.write(log_entry)
            print(eval_result)

        # save retrieval cache
        if self.save_retrieval_cache:
            self.retriever._save_cache()

        return dataset


class SequentialPipeline(BasicPipeline):
    def __init__(self, config, prompt_template=None, retriever=None, generator=None):
        """
        inference stage:
            query -> pre-retrieval -> retriever -> post-retrieval -> generator
        """

        super().__init__(config, prompt_template)
        if retriever is None:
            self.retriever = get_retriever(config)
        else:
            self.retriever = retriever

        # TODO: add rewriter module

        self.use_fid = config["use_fid"]
        self.multi_model = config['multi_model']

        self.generator = None
        if config["refiner_name"] is not None:
            # For refiners other than kg, do not load the generator for now to save memory
            if "kg" in config["refiner_name"].lower():
                self.generator = get_generator(config) if generator is None else generator
            self.refiner = get_refiner(config, self.retriever, self.generator)
        else:
            self.refiner = None
            self.generator = get_generator(config) if generator is None else generator

    def naive_run(self, dataset, do_eval=True, pred_process_fun=None):
        # direct generation without RAG
        input_prompts = [self.prompt_template.get_string(question=q) for q in dataset.question]
        dataset.update_output("prompt", input_prompts)

        pred_answer_list = self.generator.generate(input_prompts)
        dataset.update_output("pred", pred_answer_list)

        dataset = self.evaluate(dataset, do_eval=do_eval, pred_process_fun=pred_process_fun)
        return dataset

    def run(self, dataset, do_eval=True, pred_process_fun=None):
        input_question = dataset.question
        input_subquestions = dataset.subquestion
        """
        input_subquestions:
            [
                [2-5个],
                [2-5个],
                ……
                [2-5个]
            ]20个
        """
        # print("yes")

        retrieval_results = self.retriever.batch_search(input_subquestions)
        # 针对每个question的若干子question，分别进行检索
        # print(retrieval_results[-1])
        
        # 接下来进行generation。首先用LLM1进行生成，之后收集生成结果
        # 然后定义新的prompt，并导入若干其他model，进行ensemble，得到
        # ensemble结果。
        # 然后收集ensemble结果，重新设计prompt，并再次输入LLM1，得到新的结果
        
        # dataset.update_output("retrieval_result", retrieval_results)
        # for data in dataset.data:
        #     print(data.output['retrieval_result'])
        # print(dataset.data[0].output)
        # for data in dataset.data:
        #     print(data.output)
        
        # 关于retrieval_results：一共20个元素，每个元素是一个长为2-5的list，每个list里有5个dict，代表了检索返回的结果
        """
        retrieval_results:
            [
                [
                    [5个{}],[5个{}],……[5个{}]2-5个
                ],
                [
                    [],[],……[]
                ],
                ……
                [
                    [],[],……[]
                ]
            ]20个
        """
        # if self.multi_model:
        #     dataset = self.run_multi(self.config, dataset, input_question)

        # if self.refiner:
        #     input_prompt_flag = self.refiner.input_prompt_flag
        #     if "llmlingua" in self.refiner.name and input_prompt_flag:
        #         # input prompt
        #         input_prompts = [
        #             self.prompt_template.get_string(question=q, retrieval_result=r)
        #             for q, r in zip(dataset.question, dataset.retrieval_result)
        #         ]
        #         dataset.update_output("prompt", input_prompts)
        #         input_prompts = self.refiner.batch_run(dataset)
        #     else:
        #         # input retrieval docs
        #         refine_results = self.refiner.batch_run(dataset)
        #         dataset.update_output("refine_result", refine_results)
        #         input_prompts = [
        #             self.prompt_template.get_string(question=q, formatted_reference=r)
        #             for q, r in zip(dataset.question, refine_results)
        #         ]

        # else:
        # input_prompts = [
        #     self.prompt_template.get_string(question=q, retrieval_result=r)
        #     for q, r in zip(dataset.question, dataset.retrieval_result)
        # ]
        """
        input_prompts:
            [
                [
                    2-5个
                ],
                [
                    2-5个
                ],
                ……
                [
                    2-5个
                ]
            ]20个
            """
        input_prompts = [
            [self.prompt_template.get_string(question=q, retrieval_result=r)
             for q, r in zip(input_subquestion, retrieval_result)]
            for input_subquestion, retrieval_result in zip(input_subquestions, retrieval_results)
        ]
        
        # dataset.update_output("prompt", input_prompts)
        # print(input_prompts)

        # if self.use_fid:
        #     print("Use FiD generation")
        #     input_prompts = []
        #     for item in dataset:
        #         q = item.question
        #         docs = item.retrieval_result
        #         input_prompts.append([q + " " + doc for doc in docs])
        # # delete used refiner to release memory
        # if self.refiner:
        #     if "kg" in self.config["refiner_name"].lower():
        #         self.generator = self.refiner.generator
        #     else:
        #         self.generator = get_generator(self.config)
        #     del self.refiner
        # print(input_prompts)
        pred_answer_lists = []
        for i in tqdm(
            range(0, len(input_prompts), self.batch_size), desc="Generation process: "
        ):
            pred_answer_lists.append(self.generator.generate(input_prompts[i]))
        
        print(pred_answer_lists)
        return
        # pred_answer_lists = [self.generator.generate(input_prompt) for input_prompt in input_prompts]
        dataset.update_output("pred", pred_answer_lists)

        dataset = self.evaluate(dataset, do_eval=do_eval, pred_process_fun=pred_process_fun)

        return dataset


class ConditionalPipeline(BasicPipeline):
    def __init__(self, config, prompt_template=None):
        """
        inference stage:
            query -> judger -> sequential pipeline or naive generate
        """

        super().__init__(config, prompt_template)
        self.judger = get_judger(config)
        self.retriever = get_retriever(config)
        self.generator = get_generator(config)

        self.sequential_pipeline = SequentialPipeline(
            config, prompt_template, retriever=self.retriever, generator=self.generator
        )

        self.zero_shot_templete = PromptTemplate(
            config=config,
            system_prompt="Answer the question based on your own knowledge. \
                            Only give me the answer and do not output any other words.",
            user_prompt="Question: {question}",
        )

    def run(self, dataset, do_eval=True, pred_process_fun=None):
        # judge_result: list of bool element, representing whether to use retrieval
        judge_result = self.judger.judge(dataset)
        dataset.update_output("judge_result", judge_result)

        # split dataset based on judge_result
        dataset_split = split_dataset(dataset, judge_result)
        pos_dataset, neg_dataset = dataset_split[True], dataset_split[False]

        pos_dataset = self.sequential_pipeline.run(pos_dataset, do_eval=False)
        self.sequential_pipeline.prompt_template = self.zero_shot_templete
        neg_dataset = self.sequential_pipeline.naive_run(neg_dataset, do_eval=False)

        # merge datasets into original format
        dataset = merge_dataset(dataset_split, judge_result)

        dataset = self.evaluate(dataset, do_eval=do_eval, pred_process_fun=pred_process_fun)

        return dataset


class AdaptivePipeline(BasicPipeline):
    def __init__(
        self,
        config,
        norag_template=None,
        single_hop_prompt_template=None,
        multi_hop_prompt_template=None,
    ):
        super().__init__(config)
        # load adaptive classifier as judger
        self.judger = get_judger(config)

        retriever = get_retriever(config)
        generator = get_generator(config)

        # Load three pipeline for three types of query: naive/single-hop/multi-hop
        from flashrag.pipeline import IRCOTPipeline

        if norag_template is None:
            norag_templete = PromptTemplate(
                config=config,
                system_prompt="Answer the question based on your own knowledge. Only give me the answer and do not output any other words.",
                user_prompt="Question: {question}",
            )
        self.norag_pipeline = SequentialPipeline(
            config,
            prompt_template=norag_templete,
            retriever=retriever,
            generator=generator,
        )

        self.single_hop_pipeline = SequentialPipeline(
            config,
            prompt_template=single_hop_prompt_template,
            retriever=retriever,
            generator=generator,
        )

        self.multi_hop_pipeline = IRCOTPipeline(
            config,
            prompt_template=multi_hop_prompt_template,
            retriever=retriever,
            generator=generator,
        )

    def run(self, dataset, do_eval=True, pred_process_fun=None):
        # judge_result: choice result representing which pipeline to use(e.g. A, B, C)
        judge_result = self.judger.judge(dataset)
        dataset.update_output("judge_result", judge_result)

        # split dataset based on judge_result
        dataset_split = split_dataset(dataset, judge_result)
        for symbol, symbol_dataset in dataset_split.items():
            if symbol == "A":
                symbol_dataset = self.norag_pipeline.naive_run(symbol_dataset, do_eval=False)
            elif symbol == "B":
                symbol_dataset = self.single_hop_pipeline.run(symbol_dataset, do_eval=False)
            elif symbol == "C":
                symbol_dataset = self.multi_hop_pipeline.run(symbol_dataset, do_eval=False)
            else:
                assert False, "Unknown symbol!"

        # merge datasets into original format
        dataset = merge_dataset(dataset_split, judge_result)

        dataset = self.evaluate(dataset, do_eval=do_eval, pred_process_fun=pred_process_fun)

        return dataset
