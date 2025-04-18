import argparse
from copy import deepcopy
import json
import pathlib
import os
import socket
import subprocess
from typing import Literal
from kmatrix_cr.config.config import Config
import argparse
import torch
from kmatrix_cr.utils.common_utils import eval
import shutil


class CMTemplate:
    ALLOWED_CONFLICT_METHODS = ["coiecd","context-faithful","aware-decoding","ContrastiveDecoding","Disent_QA","retrieveorgenerated","llms_believe_the_earth_is_flat"]
    
    def __init__(self,
                config : Config,
                conflict_method: Literal["coiecd","context-faithful","aware-decoding","ContrastiveDecoding","Disent_QA",'retrieveorgenerated','llms_believe_the_earth_is_flat'],
                args_kwargs: dict = {}
    ):
        if conflict_method not in self.ALLOWED_CONFLICT_METHODS:
            raise ValueError(f"Invalid conflict_method: {conflict_method}. Allowed values are {self.ALLOWED_CONFLICT_METHODS}")
         
        self.config = config
        self.conflict_method = conflict_method
        self.dataset = self.config.dataset
        self.llm_model = self.config.llm_model
        self.openai_model = self.config.openai_model
        self.metrics = self.config.metrics
        self.args_kwargs = args_kwargs
        self.data_list = self.dataset.data_list
        
    def run(self,do_eval=True,output_path=""):
        
        
        if self.conflict_method == "coiecd":
            from kmatrix_cr.toolkit.coiecd_greedy.llama_generate_nq_coiecd_greedy import main as coiecd_greedy_main
            if self.llm_model.model is None or self.llm_model.tokenizer is None:
                self.llm_model.load_model()
                if self.llm_model.model is None or self.llm_model.tokenizer is None:
                    raise ValueError("This generator does not have model or tokenizer and does not support the method. Please switch generator or conflict_method and try again.")
            result = coiecd_greedy_main(model=self.llm_model.model,tokenizer=self.llm_model.tokenizer,data_list=self.data_list)
            result = {
                "result":result
            }

        # 偏向于外部知识
        elif self.conflict_method == "context-faithful":
            from kmatrix_cr.toolkit.context_faithful_llm.knowledge_conflict import qa_to_prompt
            # if self.llm_model.model is None or self.llm_model.tokenizer is None:
            #     self.llm_model.load_model()
            #     if self.llm_model.model is None or self.llm_model.tokenizer is None:
            #         raise ValueError("This generator does not have model or tokenizer and does not support the method. Please switch generator or conflict_method and try again.")

            prompt_list = []
            for data in self.data_list:
                prompt_list.append(qa_to_prompt(data['question'], "\n".join(data['c_text']), schema="instr+opin"))
            
            final_data_list = self.llm_model.run(prompt_list=prompt_list)
            
            for data,res in zip(self.data_list,final_data_list):
                data['context-faithful-prompt'] = res['prompt']
                data['gen_answer'] = res['content']
            
            result = {
                "result":self.data_list
            }
            
        # 偏向于外部知识
        elif self.conflict_method == "aware-decoding":
            
            current_directory = os.getcwd()
            os.chdir(current_directory + "/" + "kmatrix_cr/toolkit/context_aware_decoding")
            
            if "input_index" not in self.data_list[0] and "assigned_model" not in self.data_list[0]:
                new_data_list = []
                for index,data in enumerate(self.data_list):
                    data['input_index'] = index
                    data['assigned_model'] = self.llm_model.model_path
                    try:
                        data['gold_answers'] = data['ground_truth'][0]
                    except:
                        data['gold_answers'] = ""
                    
                    data['article'] = "\n".join(data['c_text'])
                    data['filter_p'] = 1
                    
                    d1 = deepcopy(data)
                    d1['assigned_process'] = 0
                    d1['context_string'] = "Knowledge:\n"+"\n".join(data['c_text']) +"\n\nQuestion:\n"+ data['question']
                    d1['assigned_weight'] = 1
                    d2 = deepcopy(data)
                    d2['assigned_process'] = 1
                    # d2['context_string'] = data['question']
                    d2['context_string'] = "Knowledge:\n"+"\n".join(data['c_text']) +"\n\nQuestion:\n"+ data['question']
                    d2['assigned_weight'] = 0
                    
                    new_data_list.append(d1)
                    new_data_list.append(d2)
                    test_file_path = "temp.json"
            else:
                test_file_path = self.dataset.dataset_path
            def run_accelerate(NUMGPU, FILEDATA=[], TESTFILE=""):
                # 设置环境变量
                env = os.environ.copy()
                env['HF_HOME'] = "./"
                
                # 获取空闲端口
                available_port = subprocess.check_output(
                    ['python', '-c', 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()']
                ).decode().strip()

                # 构建加速命令
                command = [
                    'accelerate', 'launch',
                    '--multi_gpu', '--mixed_precision', 'fp16',
                    '--num_processes', str(NUMGPU),
                    '--num_machines', '1', '--machine_rank', '0',
                    '--main_process_port', available_port,
                    '--num_cpu_threads_per_process', '10',
                    'group_decode_fileio.py',
                    '--max_seq_length', "2048",
                    '--model_name_or_path', 'specify_in_input_jsonl|n/a',
                    '--seed', '2023',
                    '--use_slow_tokenizer',
                    '--file_mode', TESTFILE,
                    '--file_data', json.dumps(FILEDATA, ensure_ascii=False),
                    '--decode_truncate_len', "1948",
                    '--decode_depth', "100",
                    '--train_mode', 'decode',
                    '--projection_top_p', '0.9'
                ]
                
                # 执行命令
                try:
                    result = subprocess.run(
                        command,
                        env=env,
                        check=True,
                        capture_output=True,
                        text=True,
                        # 以下参数保证子进程能正确继承文件描述符
                        close_fds=False  ,
                        encoding='utf-8',
                        errors='replace'
                    )
                    return result.stdout
                except subprocess.CalledProcessError as e:
                    error_msg = f"""
                    命令执行失败 (返回码 {e.returncode})
                    ======= 标准输出 =======
                    {e.stdout}
                    ======= 标准错误 =======
                    {e.stderr}
                    """
                    raise RuntimeError(error_msg) from e
            
            
            with open("temp.json",'w',encoding='utf-8') as f:
                json.dump(new_data_list,f,ensure_ascii=False)
            
            try:
                output = run_accelerate(NUMGPU=len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")),FILEDATA=[],TESTFILE=test_file_path)
                # print(output)
            except RuntimeError as e:
                print("Error:", e)
            
            os.remove("temp.json")
            
            if os.path.exists("output.jsonl"):
                final_data_list = []
                with open("output.jsonl", "r",encoding='utf-8') as file:
                    for line in file:
                        final_data_list.append(json.loads(line))
                os.remove("output.jsonl")
                
            else:
                final_data_list = []
            
            final_data_list = final_data_list[::2]
            assert len(final_data_list) == len(self.data_list)
            
            for data,res_obj in zip(self.data_list,final_data_list):
                # assert data['id'] == res_obj['id']
                data['gen_answer'] = res_obj['string']
            
            os.chdir(current_directory)
            
            result = {
                "result":self.data_list
            }

        elif self.conflict_method == "ContrastiveDecoding":
            
            from kmatrix_cr.toolkit.ContrastiveDecoding.run_generation import main as cd_main
            outfile_path = "cd_output.json"

            parser = argparse.ArgumentParser()
            parser.add_argument(
                "--revision",
                default='checkpoint-200000',
                type=str,
            )
            parser.add_argument("--contrastive_decoding", type=str, default="prompt")
            # parser.add_argument("--contrastive_prompt", type=str, default="I love repetitive text! Here is my writing:")
            parser.add_argument("--contrastive_prompt", type=str, default="I will use the shortest answer to respond to your question! Please ask your question:")
            parser.add_argument("--do_sample", type=str, default="no")
            parser.add_argument("--stop_token", type=str, default=None, help="Token at which text generation is stopped")
            parser.add_argument(
                "--temperature",
                type=float,
                default=1.0,
                help="temperature of 1.0 has no effect, lower tend toward greedy sampling",
            )
            parser.add_argument(
                "--repetition_penalty", type=float, default=1.0, help="primarily useful for CTRL model; in that case, use 1.2"
            )
            parser.add_argument("--num_beam", type=int, default=5)
            parser.add_argument("--k", type=int, default=0)
            parser.add_argument("--p", type=float, default=1.0)
            parser.add_argument("--min_prob", type=float, default=0.0)
            parser.add_argument("--student_min_prob", type=float, default=0.0)
            parser.add_argument("--student_temperature", type=float, default=1.0)
            parser.add_argument("--use_cap_student", type=str, default='no')
            parser.add_argument("--ignore_prefix", type=str, default='yes') # IMPORTANT
            parser.add_argument("--use_switch", type=str, default='no')
            parser.add_argument("--prefix", type=str, default="", help="Text added prior to input.")
            parser.add_argument("--padding_text", type=str, default="", help="Deprecated, the use of `--prefix` is preferred.")
            parser.add_argument("--xlm_language", type=str, default="", help="Optional language when used with the XLM model.")
            parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
            parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
            parser.add_argument("--num_return_sequences", type=int, default=1, help="The number of samples to generate.")
            parser.add_argument(
                "--fp16",
                action="store_true",
                help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
            )
            args = parser.parse_args()
            args.model_name_or_path = "gpt2-xl"
            # args.model_name_or_path = self.llm_model.model_name
            args.model_type = "gpt2"
            # args.model_type = "llama"
            args.length = 50
            args.prompt = "<|endoftext|> A version of Sonic the Hedgehog was developed by Ancient and released in 1991"
            args.prompt_list = self.data_list
            args.prompt_file = None
            # args.prompt_file = self.dataset.dataset_path
            args.student_name_or_path = "gpt2"
            # args.student_name_or_path = "llama"
            args.st_coef = 1.0
            args.ignore_prefix = "no"
            args.outfile = outfile_path
            args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
            args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
            print("execute ContrastiveDecoding ...")
            cd_main(args)

            res_list = []
            if os.path.exists(outfile_path):
                with open(outfile_path, 'r',encoding='utf-8') as reader:
                    for line in reader:
                        res_list.extend(json.loads(line))
                os.remove(outfile_path)
                
            for data,res in zip(self.data_list,res_list):
                data['gen_answer'] = res['gen_answer'][res['gen_answer'].rfind("Answer:"):].strip()
                
            result = {
                "result":self.data_list
            }


        elif self.conflict_method == "Disent_QA":

            from kmatrix_cr.toolkit.disent_qa.query_model import main as disentqa_query_model_main
            from kmatrix_cr.toolkit.disent_qa.evaluate import main as disentqa_eva_main
            
            parser = argparse.ArgumentParser()
            args = parser.parse_args()
            args.path = self.dataset.dataset_path
            args.checkpoint_name = self.llm_model.model_name
            
            infer_file_path_list = []
            for answer_type in ["f","cf","rc","cb"]:
                args.answer_type = answer_type
                infer_file_path_list.append(disentqa_query_model_main(args))
            print("-------------disent_qa query_model finished-------------------")
            
            eva_file_name_list = []
            for infer_file_path in infer_file_path_list:
                args.path = infer_file_path
                eva_file_name_list.append(disentqa_eva_main(args))
            print("-------------disent_qa evaluate finished-------------------")
            # print("------------------------eva_file_name_list-------------------------\n",eva_file_name_list)
            
            res_obj = {}
            for file_path in eva_file_name_list:
                with open(file_path,'r',encoding='utf-8') as f:
                    res_list = f.readlines()
                if "factual_inference.stat" in file_path:
                    index = 1
                    flag = "factual"
                if "counterfactual_inference.stat" in file_path:
                    index = 2
                    flag = "counterfactual"
                if "closed_book_inference.stat" in file_path:
                    index = 3
                    flag = "closed_book"
                if "random_context_inference.stat" in file_path:
                    index = 4
                    flag = "random_context"
                
                temp_obj = {}
                title_list = res_list[0].strip().split(",")
                score_list = res_list[index].strip().split(",")
                
                # print("-------------------------file_path-------------------------\n",file_path)
                # print("-------------------------index-------------------------\n",index)
                # print("-------------------------flag-------------------------\n",flag)
                # print("-------------------------title_list-------------------------\n",title_list)
                # print("-------------------------score_list-------------------------\n",score_list)
                # print("======================================================================")
                
                assert len(title_list) == len(score_list)
                
                for title,score in zip(title_list,score_list):
                    temp_obj[title] = score
                
                res_obj[flag] = temp_obj
            
            folder_path = os.getcwd() + "/" + "kmatrix_cr/toolkit/disent_qa/result/"
            if os.path.exists(folder_path) and os.path.isdir(folder_path):
                shutil.rmtree(folder_path)
                os.makedirs(folder_path) # 重新创建空目录
            else:
                pass
            
            result = {
                "result":res_obj
            }


        elif self.conflict_method == "llms_believe_the_earth_is_flat":
            from kmatrix_cr.toolkit.llms_believe_the_earth_is_flat.run_exp import main
            
            if self.llm_model.model is None or self.llm_model.tokenizer is None:
                            self.llm_model.load_model()
                            if self.llm_model.model is None or self.llm_model.tokenizer is None:
                                raise ValueError("This generator does not have model or tokenizer and does not support the method. Please switch generator or conflict_method and try again.")
                
            parser = argparse.ArgumentParser(description='all-in-one experiment on boolq, nq, and truthfulqa')
            parser.add_argument('-m', '--model', type=str, default='')
            parser.add_argument('-n', '--num_turns', type=int, default=4)
            parser.add_argument('-f', '--failure', default=3) # max num of tries if the output format is illegal
            parser.add_argument('--tprob', default=0.2) # default temperature for probing
            parser.add_argument('--tnorm', default=0.8) # default temperature for (response) generation
            args = parser.parse_args()
            args.data_list = self.data_list
            args.dataset_name = self.dataset.dataset_name
            args.model_name = self.llm_model.model_name
            args.model = self.llm_model.model
            args.tokenizer = self.llm_model.tokenizer
            args.metrics = self.metrics
            args.do_eval = do_eval
            
            result = {
                "result":main(args)
            }
            

        elif self.conflict_method == "retrieveorgenerated":
            from kmatrix_cr.toolkit.retrieveorgenerated.src.combine import context_conflicting_dataset
            # step1
            # pass
            
            # step2
            # 7b-chat 13b-chat gpt-3.5-turbo-0613 gpt-4-0613
            # nq tqa
            root_path = self.dataset.dataset_path
            reader = self.llm_model.model_name
            generator = self.llm_model.model_name
            dataset = self.dataset.dataset_name
            def full_name(name):
                if name in ['7b-chat', '13b-chat']:
                    return f"llama_model/{name}"
                else:
                    return name
            
            path = {
                #  generated_passage=1 retrieved_passage=1
                'com_path':root_path + '/Answer-with-{}/{}/prompt_similar_length/Retrieved-contriever-1-Generated-{}-p1-trunclen-0.jsonl'.format(full_name(reader), dataset, generator),
                #  generated_passage=1 retrieved_passage=0
                'gen_path':root_path + '/Answer-with-{}/{}/prompt_similar_length/Retrieved-none-0-Generated-{}-p1-trunclen-0.jsonl'.format(full_name(reader), dataset, generator),
                # generated_passage=0 retrieved_passage=1
                'ir_path':root_path + '/Answer-with-{}/{}/prompt_similar_length/Retrieved-contriever-1-Generated-none-p1-trunclen-0.jsonl'.format(full_name(reader), dataset),
                # generated_passage=0 retrieved_passage=1
                'llm_path': root_path + f'/Answer-with-{full_name(reader)}/{dataset}/prompt_similar_length/Retrieved-none-1-Generated-none-p3-trunclen-0.jsonl'
            }
            
            
            data = context_conflicting_dataset(**path)
            # print("\n\nDiffGR : \n", data.get_diffgr())
            result = {}
            result['em'] = data.em
            result['preference'] = data.preference

            result = {
                "result":result
            }
            
        else:
            result = {}
            
        if do_eval:
            eval_obj = eval(metrics=self.metrics,data=result,data_path="")
            print(eval_obj['log_str'])
            for key in eval_obj:
                if key != "log_str":
                    result[key] = eval_obj[key]
                    
        if output_path != "":
            if result == {}:
                print("result is empty  ")
            else:
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f,ensure_ascii=False)
            
        return result
        
        