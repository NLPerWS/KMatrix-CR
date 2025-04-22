import traceback
from typing import Any, Dict, List, Optional
from transformers import AutoTokenizer,LlamaForCausalLM
from transformers import AutoModelForCausalLM, AutoModel
from vllm import RequestOutput, SamplingParams,LLM
import torch
from kmatrix_cr.generator.root_generator import RootGenerator
from root_config import RootConfig


class LLmGenerator(RootGenerator):
    def __init__(
        self,
        model_path:str,
        generation_kwargs: Dict[str, Any] = {
            # "max_tokens":4096,
            # "temperature":0
        },
        load_model: bool = False,
        load_model_mode: str = "llm"
    ):
        self.model_path = model_path
        self.generation_kwargs = generation_kwargs
        self.model_name = model_path
        self.model = None
        self.tokenizer = None
        self.load_model_mode = load_model_mode
        self.token = None
        if load_model:
            self.load_model(mode=self.load_model_mode)
        
    def RequestOutputToDict(self,pred):
        if not isinstance(pred,RequestOutput):
            return pred
        pred_dict = {
            "request_id":pred.request_id,
            "prompt":pred.prompt,
            "prompt_token_ids":pred.prompt_token_ids,
            "prompt_logprobs":pred.prompt_logprobs,
            "outputs":[{
                "index":pred.outputs[0].index,
                "text":pred.outputs[0].text,
                "token_ids":pred.outputs[0].token_ids,
                "cumulative_logprob":pred.outputs[0].cumulative_logprob,
                "logprobs":pred.outputs[0].logprobs,
                "finish_reason":pred.outputs[0].finish_reason,
                }],
            "finished":pred.finished
        }
        return pred_dict
        
    def load_model(self,mode="llama"):
        
        catch_flag = False
        
        for catch in RootConfig.tempModelCatch:
            if catch["path"] == self.model_path + mode:
                catch_flag = True
                self.model = catch['model']
                self.tokenizer = catch['tokenizer']
                self.model_name = self.model_path
                break
            
        if catch_flag == False:
        
            try:
                if self.model_path == "":
                    raise ValueError("Model path is not specified")
                
                if mode == "llm":
                    # self.model = LLM(model=self.model_path,dtype="half",trust_remote_code=True,tensor_parallel_size=torch.cuda.device_count())
                    self.model = LLM(model=self.model_path,dtype="half",trust_remote_code=True,tensor_parallel_size=1)
                    self.tokenizer = AutoTokenizer.from_pretrained(self.model_path,trust_remote_code=True)
                else:
                    self.model = LlamaForCausalLM.from_pretrained(self.model_path, device_map="auto", low_cpu_mem_usage=True, torch_dtype=torch.float32,token=self.token)
                    self.tokenizer = AutoTokenizer.from_pretrained(self.model_path,trust_remote_code=True,token=self.token)
                    
                self.model_name = self.model_path
                
                RootConfig.tempModelCatch.append(
                    {
                        "path":self.model_path + mode,
                        "model":self.model,
                        "tokenizer":self.tokenizer
                    }
                )
                
            except Exception as e:
                traceback.print_exc()
        
    def run(
        self,
        prompt: str = "",
        prompt_list: List[str] = [],
        sampling_params: Any = None
    ):
        param_sampling_params = SamplingParams()
        for k in self.generation_kwargs:
            param_sampling_params.__setattr__(k, self.generation_kwargs[k])
    
        if sampling_params is not None:
            for k in sampling_params:
                param_sampling_params.__setattr__(k, sampling_params[k])
            
        if self.model==None:
            raise ValueError("The model is empty.")
        
        if prompt != "" and len(prompt_list) == 0:
            prompt_list = [prompt]
        
        final_result = []
        preds = self.model.generate(prompts=prompt_list,sampling_params=param_sampling_params)
        for one_prompt,pred in zip(prompt_list, preds):
            pred_dict =  self.RequestOutputToDict(pred)
            content = pred_dict['outputs'][0]['text']
            final_result.append({
                "prompt":one_prompt,
                "content":content,
                "meta":{"pred":pred_dict},    
            })    
        
        return final_result
    
        
    def run_llama(
        self,
        prompt: str = "",
        prompt_list: List[str] = [],
        sampling_params: Any = None
    ):
        
        param_sampling_params = SamplingParams()
        for k in self.generation_kwargs:
            param_sampling_params.__setattr__(k, self.generation_kwargs[k])
    
        if sampling_params is not None:
            for k in sampling_params:
                param_sampling_params.__setattr__(k, sampling_params[k])
            
            
        if self.model==None:
            raise ValueError("The model is empty.")
        
        if prompt != "" and len(prompt_list) == 0:
            prompt_list = [prompt]
        
        final_result = []

        for index,input_text in enumerate(prompt_list):
            inputs = self.tokenizer(input_text, return_tensors="pt")
            inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
            with torch.no_grad():
                outputs = self.model.generate(**inputs,sampling_params=param_sampling_params)
            output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            final_result.append({
                "prompt":input_text,
                "content":output_text,
                "meta":{"pred":{}},    
            })    
            print(f"{index+1}/{len(prompt_list)} ok ...")
        
        return final_result
        
    