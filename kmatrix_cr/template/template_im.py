import argparse
import json
import os
from typing import Literal
from kmatrix_cr.config.config import Config
from kmatrix_cr.utils.common_utils import eval

class IMTemplate:
    ALLOWED_CONFLICT_METHODS = ["dola"]
    def __init__(self,
                config : Config,
                conflict_method: Literal["dola"],
                args_kwargs: dict = {}
    ):
        if conflict_method not in self.ALLOWED_CONFLICT_METHODS:
            raise ValueError(f"Invalid conflict_method: {conflict_method}. Allowed values are {self.ALLOWED_CONFLICT_METHODS}")
        
        self.args_kwargs = args_kwargs
        self.config = config
        self.conflict_method = conflict_method
        self.dataset = self.config.dataset
        self.llm_model = self.config.llm_model
        self.openai_model = self.config.openai_model
        
        self.metrics = self.config.metrics
        self.data_list =  self.dataset.data_list
        
    def run(self,do_eval=True,output_path=""):

        
        if self.conflict_method == "dola":
            # from kmatrix_cr.toolkit.dola.gsm8k_eval import main as dola_gsm8k_main
            from kmatrix_cr.toolkit.dola.strqa_eval import main as strategyqa_main
                
            parser = argparse.ArgumentParser()
            parser.add_argument("--model-name", type=str, default="huggyllama/llama-7b")
            parser.add_argument("--num-gpus", type=str, default="1")
            parser.add_argument("--max_gpu_memory", type=int, default=27)
            parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda")
            parser.add_argument("--data-path", type=str, default="./strqa")
            parser.add_argument("--output-path", type=str, default="./strqa_result")
            # parallel mode (split the dataset into multiple parts, inference by separate processes)
            parser.add_argument("--early-exit-layers", type=str, default="-1")
            parser.add_argument("--parallel", action="store_true")
            parser.add_argument("--total-shard", type=int, default=8)
            parser.add_argument("--shard-id", type=int, default=None)
            parser.add_argument("--max-new-tokens", type=int, default=256)
            parser.add_argument("--top_p", type=float, default=0.95)
            parser.add_argument("--top_k", type=int, default=0)
            parser.add_argument("--temperature", type=float, default=0.9)
            parser.add_argument("--repetition_penalty", type=float, default=None)
            parser.add_argument("--relative_top", type=float, default=0.1)
            parser.add_argument("--do_sample", action="store_true")
            parser.add_argument("--do_shuffle", action="store_true")
            parser.add_argument("--debug", action="store_true")
            parser.add_argument("--seed", type=int, default=42)
            parser.add_argument("--retry", type=int, default=3)
            args = parser.parse_args()
            
            args.model_name=self.llm_model.model_name
            args.data_list=self.data_list
            args.early_exit_layers = "0,2,4,6,8,10,12,14,32"
            result = strategyqa_main(args=args)
            
            for data,res,is_correct,model_completion,full_input_text in zip(self.data_list,result['model_answer'],result['is_correct'],result['model_completion'],result['full_input_text']):
                data['gen_answer'] = res
                data['is_correct'] = is_correct
                data['model_completion'] = model_completion
                data['full_input_text'] = full_input_text
            
            result = {"result":self.data_list}
    
        else:
            result = {}
        
        if do_eval:
            eval_obj = eval(metrics=self.metrics,data=result,data_path="")
            print(eval_obj)
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
        