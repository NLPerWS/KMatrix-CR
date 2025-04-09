import argparse
from copy import deepcopy
import json
import pathlib
import os
import socket
import subprocess
from typing import Literal
from kmatrix_cr.config.config import Config
from accelerate import Accelerator
from accelerate.utils import KwargsHandler
import argparse
import torch
from kmatrix_cr.utils.common_utils import eval

class CMTemplate:
    ALLOWED_CONFLICT_METHODS = ["coiecd","context-faithful","aware-decoding","retrieveorgenerated","llms_believe_the_earth_is_flat"]
    
    def __init__(self,
                config : Config,
                conflict_method: Literal["coiecd","context-faithful","aware-decoding",'retrieveorgenerated','llms_believe_the_earth_is_flat'],
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
            if self.llm_model.model is None or self.llm_model.tokenizer is None:
                self.llm_model.load_model()
                if self.llm_model.model is None or self.llm_model.tokenizer is None:
                    raise ValueError("This generator does not have model or tokenizer and does not support the method. Please switch generator or conflict_method and try again.")

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
                    data['gold_answers'] = data['ground_truth'][0]
                    data['article'] = "\n".join(data['c_text'])
                    data['filter_p'] = 1
                    
                    d1 = deepcopy(data)
                    d1['assigned_process'] = 0
                    d1['context_string'] = "\n".join(data['c_text']) +"\n\n"+ data['question']
                    d1['assigned_weight'] = 1
                    d2 = deepcopy(data)
                    d2['assigned_process'] = 1
                    d2['context_string'] = data['question']
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
                assert data['id'] == res_obj['id']
                data['gen_answer'] = res_obj['string']
            
            os.chdir(current_directory)
            
            result = {
                "result":self.data_list
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
                'com_path':root_path + '/Answer-with-{}/{}/prompt_similar_length/Retrieved-contriever-1-Generated-{}-p1-trunclen-0.jsonl'.format(full_name(reader), dataset, generator),
                'gen_path':root_path + '/Answer-with-{}/{}/prompt_similar_length/Retrieved-none-0-Generated-{}-p1-trunclen-0.jsonl'.format(full_name(reader), dataset, generator),
                'ir_path':root_path + '/Answer-with-{}/{}/prompt_similar_length/Retrieved-contriever-1-Generated-none-p1-trunclen-0.jsonl'.format(full_name(reader), dataset),
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
        
        