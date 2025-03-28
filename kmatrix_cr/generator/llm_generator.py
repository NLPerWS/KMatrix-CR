import traceback
from typing import Any, Dict, List, Optional
from transformers import AutoTokenizer,LlamaForCausalLM
from transformers import AutoModelForCausalLM, AutoModel
from vllm import RequestOutput, SamplingParams,LLM
import torch
from torch.nn import DataParallel

from kmatrix_cr.generator.root_generator import RootGenerator

class LLmGenerator(RootGenerator):
    def __init__(
        self,
        model_path:str,
        generation_kwargs: Dict[str, Any] = {},
        load_model: bool = False,
        load_model_mode: str = "llama"
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
        try:
            if self.model_path == "":
                raise ValueError("Model path is not specified")
            
            if mode == "llm":
                self.model = LLM(model=self.model_path,tensor_parallel_size=1,dtype="half",trust_remote_code=True)
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path,trust_remote_code=True)
            else:
                
                self.model = LlamaForCausalLM.from_pretrained(self.model_path, device_map="auto", low_cpu_mem_usage=True, torch_dtype=torch.float32,token=self.token)
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path,trust_remote_code=True,token=self.token)
                
            self.model_name = self.model_path
            
        except Exception as e:
            traceback.print_exc()
        
    def run(
        self,
        prompt: str = "",
        prompt_list: List[str] = [],
        sampling_params: Any = None
    ):
        
        if sampling_params is None:
            sampling_params = SamplingParams()
            for k in self.generation_kwargs:
                sampling_params.__setattr__(k, self.generation_kwargs[k])
        else:
            sampling_params = sampling_params
            
        if self.model==None:
            raise ValueError("The model is empty.")
        
        if prompt != "" and len(prompt_list) == 0:
            prompt_list = [prompt]
        
        final_result = []
        preds = self.model.generate(prompts=prompt_list)
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
        
        if sampling_params is None:
            sampling_params = SamplingParams()
            for k in self.generation_kwargs:
                sampling_params.__setattr__(k, self.generation_kwargs[k])
        else:
            sampling_params = sampling_params
            
        if self.model==None:
            raise ValueError("The model is empty.")
        
        if prompt != "" and len(prompt_list) == 0:
            prompt_list = [prompt]
        
        final_result = []

        for index,input_text in enumerate(prompt_list):
            inputs = self.tokenizer(input_text, return_tensors="pt")
            inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
            with torch.no_grad():
                outputs = self.model.generate(**inputs)
            output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            final_result.append({
                "prompt":input_text,
                "content":output_text,
                "meta":{"pred":{}},    
            })    
            print(f"{index+1}/{len(prompt_list)} ok ...")
        
        return final_result
        
    